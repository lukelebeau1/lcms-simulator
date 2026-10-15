"""The trace: one compound followed from its ampoule to the reported area, nominal against actual."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from lcms_sim.faults.library import LIBRARY, Scenario
from lcms_sim.io.loaders import load_input_set, load_prep_recipe
from lcms_sim.io.schemas import TransitionRole
from lcms_sim.pipeline import Kind, RunConditions, run_sequence
from lcms_sim.quant.ledger import molecules, run_recipe, with_actual
from lcms_sim.trace import STATIONS, Actual, Station, build_trace, path_steps, spray_fraction

EXAMPLES = Path(__file__).resolve().parents[1] / "data" / "examples"


@pytest.fixture(scope="module")
def inputs():
    return load_input_set(EXAMPLES)


@pytest.fixture(scope="module")
def recipe():
    return load_prep_recipe(EXAMPLES / "prep_recipe.yaml")


@pytest.fixture(scope="module")
def expected_run(inputs):
    return run_sequence(inputs, np.random.default_rng(0), conditions=RunConditions(detector="expected"))


def calibrator(run, level):
    return next(r for r in run.injections if r.item.kind is Kind.calibrator and r.item.ref == level)


def stop_named(trace, prefix):
    return next(s for s in trace.stops if s.name.startswith(prefix))


def test_nominal_area_is_the_pipelines_expected_area(inputs, recipe, expected_run):
    for level in ("Cal 1", "Cal 8"):
        for name in ("Myclobutanil", "Ochratoxin A", "Bifenthrin"):
            t = build_trace(inputs, recipe, f"Column: {level}", name)
            assert t.nominal_area == pytest.approx(calibrator(expected_run, level).areas[name], rel=1e-6)
            assert t.actual_area == pytest.approx(t.nominal_area)
            assert t.first_departure() is None


def test_qualifier_trace_carries_its_relative_response(inputs, recipe, expected_run):
    tr = next(t for t in inputs.transitions if t.analyte == "Myclobutanil" and t.role is TransitionRole.qualifier)
    t = build_trace(inputs, recipe, "Column: Cal 4", "Myclobutanil", transition=tr)
    assert t.nominal_area == pytest.approx(calibrator(expected_run, "Cal 4").qualifier_areas[inputs.transitions.index(tr)],
                                           rel=1e-6)
    assert stop_named(t, "Collision cell").nominal_fraction == pytest.approx(0.3 * (tr.relative_response or 1.0))


def test_the_path_runs_from_the_ampoule_to_the_area(inputs, recipe):
    t = build_trace(inputs, recipe, "Column: Cal 3", "Ochratoxin A")
    assert t.stops[0].station is Station.storage and "Ochratoxin" in t.stops[0].name
    names = [s.name for s in t.stops]
    assert any("Mycotoxin master 0.2 ppm" in n for n in names)
    assert any(n.startswith("Inject") for n in names) and names[-1].startswith("Area reported")
    order = [STATIONS.index(s.station) for s in t.stops]
    assert order == sorted(order)                           # stations in physical order
    assert [s.unit for s in t.stops] == ["ng"] * sum(s.unit == "ng" for s in t.stops) + \
        ["ions"] * sum(s.unit == "ions" for s in t.stops) + ["area units"]
    _, log = run_recipe(recipe, "nominal")
    for s in t.stops:
        if s.step is not None:                             # bench stops are the ledger's own amounts
            assert s.nominal == pytest.approx(log[s.step].vessel.ng.get("Ochratoxin A", 0.0))
    first_ng, last_ng = t.stops[0].nominal, [s for s in t.stops if s.unit == "ng"][-1].nominal
    assert last_ng == pytest.approx(first_ng * math.prod(s.nominal_fraction for s in t.stops if s.unit == "ng"))


def test_path_includes_every_step_that_touched_a_vessel_on_the_way(inputs, recipe):
    _, log = run_recipe(recipe, "nominal")
    path = path_steps(recipe, log, "Column: LCS", "Myclobutanil")
    ops = [recipe.steps[k].op for k in path]
    assert ops[0] == "stock" and ops[-1] == "inject"
    assert "solvent" in ops and "extract" in ops
    assert "sample" not in ops                          # the blank flower was weighed before the spike arrived
    assert path == sorted(path)
    native = path_steps(recipe, log, "Column: PT sample", "Myclobutanil")
    assert recipe.steps[native[0]].op == "sample"       # a native residue enters with the weighed sample


def test_a_short_draw_of_one_component_is_charged_to_that_step(inputs, recipe):
    k = next(k for k, s in enumerate(recipe.steps) if s.op == "transfer" and s.source == "Ochratoxin A (2 ug/mL)")
    bad = with_actual(recipe, {k: recipe.steps[k].volume_ul.nominal * 0.5})
    t = build_trace(inputs, recipe, "Column: Cal 8", "Ochratoxin A", actual=Actual(recipe=bad))
    assert t.first_departure().step == k
    assert next(s for s in t.stops if s.step == k).loss == pytest.approx(0.5)
    # The stock lost 50 uL of volume too, so every later draw from it takes a larger share: a small gain for
    # everything in it, ochratoxin included. The product of the losses is the final ratio either way.
    k2 = next(k2 for k2 in range(k + 1, len(recipe.steps))
              if recipe.steps[k2].op == "transfer" and recipe.steps[k2].source == "Mycotoxin master 0.2 ppm")
    _, log_n = run_recipe(recipe, "nominal")
    _, log_a = run_recipe(bad, "actual")
    v = recipe.steps[k2].volume_ul.nominal
    gain = (log_n[k2].source.volume_ul + v) / (log_a[k2].source.volume_ul + v)
    assert t.ratio == pytest.approx(0.5 * gain)
    assert t.ratio == pytest.approx(math.prod(s.loss for s in t.stops))
    other = build_trace(inputs, recipe, "Column: Cal 8", "Aflatoxin B1", actual=Actual(recipe=bad))   # same master
    assert other.first_departure().step == k2 and other.ratio == pytest.approx(gain)


def test_instrument_losses_are_charged_where_they_act(inputs, recipe):
    run = RunConditions(carryover_fraction=0.02, drift_per_injection=-0.03)
    t = build_trace(inputs, recipe, "Column: Cal 2", "Myclobutanil",
                    actual=Actual(run=run, injection_index=5, previous_injection="Column: Cal 8"))
    carry = stop_named(t, "Carried over")
    assert carry.station is Station.injection and carry.loss > 1.0
    drift = stop_named(t, "Instrument sensitivity")
    assert drift.station is Station.source and drift.loss == pytest.approx(0.85)
    assert t.ratio == pytest.approx(math.prod(s.loss for s in t.stops))
    assert [s.name for s in t.losses()] == [carry.name, drift.name]


def test_a_dwell_volume_change_is_charged_to_the_column_window(inputs, recipe):
    sc = LIBRARY["dwell_volume"].apply(Scenario.baseline(inputs), 3.0)
    traces = [build_trace(inputs, recipe, "Column: Cal 5", a.name, actual=Actual.from_scenario(sc))
              for a in inputs.analytes if not a.is_internal_standard and a.molecular_weight]
    worst = min(traces, key=lambda t: t.ratio)
    assert worst.ratio < 0.9
    assert worst.first_departure().station is Station.column
    for t in traces:
        assert t.ratio == pytest.approx(math.prod(s.loss for s in t.stops))
        assert all(s.loss == pytest.approx(1.0) for s in t.stops if s.station in (Station.storage, Station.bench))


def test_dead_time_loss_grows_with_the_amount(inputs, recipe):
    acq = inputs.acquisition.model_copy(update={"detector_dead_time_s": 1e-4})
    inp = dataclasses.replace(inputs, acquisition=acq)
    lo = build_trace(inputs, recipe, "Column: Cal 1", "Myclobutanil", actual=Actual(inputs=inp))
    hi = build_trace(inputs, recipe, "Column: Cal 8", "Myclobutanil", actual=Actual(inputs=inp))
    assert stop_named(hi, "Registered within").loss < stop_named(lo, "Registered within").loss < 1.0
    assert hi.first_departure().station is Station.detector


def test_interference_adds_counts_only_for_a_sample_quantifier(inputs, recipe):
    smp = next(s for s in inputs.samples if s.id == "PT sample")
    quantified = {t.analyte for t in inputs.transitions if t.role is TransitionRole.quantifier}
    name = next(c for c, v in smp.true_concentration_ug_kg.items() if v > 0 and c in quantified)
    run = RunConditions(interference_ug_kg={name: 50.0})
    t = build_trace(inputs, recipe, "Column: PT sample", name, actual=Actual(run=run))
    assert stop_named(t, "Interference").station is Station.detector and t.ratio > 1.0
    c = build_trace(inputs, recipe, "Column: Cal 3", name, actual=Actual(run=run))
    assert not any(s.name.startswith("Interference") for s in c.stops) and c.ratio == pytest.approx(1.0)


def test_units_convert_between_ng_molecules_and_ions(inputs, recipe):
    t = build_trace(inputs, recipe, "Column: Cal 3", "Myclobutanil")
    mw = next(a.molecular_weight for a in inputs.analytes if a.name == "Myclobutanil")
    inject = next(s for s in t.stops if s.station is Station.injection)
    assert t.molecules(inject) == pytest.approx(molecules(inject.nominal, mw))
    spray = stop_named(t, "Ionised in the spray")
    assert t.ng(spray) == pytest.approx(spray.nominal / molecules(1.0, mw))
    assert spray.nominal_fraction == pytest.approx(spray_fraction(
        next(a for a in inputs.analytes if a.name == "Myclobutanil"), inputs.acquisition))
    assert t.molecules(t.stops[-1]) is None and t.ng(t.stops[-1]) is None
    rows = t.rows()
    assert len(rows) == len(t.stops) and rows[-1]["unit"] == "area units"


def test_unknown_injection_or_compound_is_an_error(inputs, recipe):
    with pytest.raises(ValueError):
        build_trace(inputs, recipe, "Column: nowhere", "Myclobutanil")
    with pytest.raises(ValueError):
        build_trace(inputs, recipe, "Column: Cal 1", "Unobtainium")
