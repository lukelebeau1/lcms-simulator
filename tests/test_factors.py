"""Loss factors: settings become the trace's actual run, and each loss lands on its own container."""

import math
from pathlib import Path

import pytest

from lcms_sim.factors import (BENCH, FACTORS, STAGE, Factor, apply, bench_efficiencies, bench_key, factor_of,
                              is_own_label, istd_note, parse_key, transfer_efficiency, transfer_multipliers)
from lcms_sim.io.loaders import load_containers, load_input_set, load_prep_recipe
from lcms_sim.quant.ledger import lineage, run_recipe
from lcms_sim.trace import build_trace

EXAMPLES = Path(__file__).resolve().parents[1] / "data" / "examples"


@pytest.fixture(scope="module")
def inputs():
    return load_input_set(EXAMPLES)


@pytest.fixture(scope="module")
def recipe():
    return load_prep_recipe(EXAMPLES / "prep_recipe.yaml")


def stop(trace, key):
    return next(s for s in trace.stops if s.key == key)


def test_every_factor_has_a_theory_and_a_provenance():
    for f in BENCH + STAGE:
        assert f.theory and f.provenance and f.key in FACTORS


def test_every_factor_has_a_scope_and_only_compound_scope_says_what_it_follows():
    for f in BENCH + STAGE:
        assert f.scope in ("uniform", "compound")
        assert (f.scope == "compound") == bool(f.varies_with), f.key
        assert ("Compound-specific" in f.scope_text) == (f.scope == "compound")
    assert FACTORS["pipette"].scope == "uniform" and FACTORS["precipitated"].scope == "compound"
    with pytest.raises(ValueError):
        Factor("x", "x", "transfer", "number", 0.0, scope="compound")
    with pytest.raises(ValueError):
        Factor("x", "x", "transfer", "number", 0.0, varies_with="m/z")


def test_setting_keys_carry_step_and_compound():
    assert bench_key(12, "pipette") == "12:pipette"
    assert bench_key(12, "precipitated", "Bifenthrin") == "12:precipitated@Bifenthrin"
    assert parse_key("12:precipitated@Bifenthrin") == (12, "precipitated", "Bifenthrin")
    assert parse_key("12:pipette") == (12, "pipette", None)
    assert parse_key("q1_share") == (None, "q1_share", None)
    assert factor_of("12:precipitated@Bifenthrin") is FACTORS["precipitated"]
    assert factor_of("q1_share") is FACTORS["q1_share"] and factor_of("3:nothing") is None


def test_bench_efficiencies_split_uniform_from_per_compound():
    settings = {"3:pipette": 50.0, "3:unmixed@A": 0.5, "3:adsorbed": 10.0, "5:precipitated@A": 0.0, "q1_share": 0.2}
    uniform, own = bench_efficiencies(settings, "B")
    assert uniform == {}                                      # the pipette is a volume, not an efficiency
    assert own == {3: {"A": 0.5, "B": pytest.approx(0.9)}}    # no compound named: charged to the followed one
    assert transfer_efficiency(settings, 3, "A") == pytest.approx(0.45)
    assert transfer_efficiency(settings, 3, "B") == pytest.approx(0.9)
    assert transfer_efficiency(settings, 3) == pytest.approx(0.9)


def test_transfer_multipliers_combine():
    k = 7
    settings = {bench_key(k, "unmixed"): 0.5, bench_key(k, "adsorbed"): 10.0}
    assert dict((f.key, m) for f, m in transfer_multipliers(settings, k)) == {"unmixed": 0.5, "adsorbed": 0.9}
    assert transfer_efficiency(settings, k) == pytest.approx(0.45)
    assert transfer_efficiency({bench_key(k, "precipitated"): 0.3}, k) == pytest.approx(0.7)
    assert transfer_efficiency({}, k) == 1.0


def test_an_unmixed_component_is_charged_to_that_transfer_only(inputs, recipe):
    k = next(k for k, s in enumerate(recipe.steps) if s.op == "transfer" and s.source == "Ochratoxin A (2 ug/mL)")
    actual = apply(inputs, recipe, {bench_key(k, "unmixed"): 0.4}, "Ochratoxin A")
    t = build_trace(inputs, recipe, "Column: Cal 8", "Ochratoxin A", actual=actual)
    assert stop(t, f"step:{k}").loss == pytest.approx(0.4)
    assert t.ratio == pytest.approx(0.4)                       # the volume is unchanged, so nothing else moves
    assert [s.key for s in t.losses()] == [f"step:{k}"]
    other = build_trace(inputs, recipe, "Column: Cal 8", "Myclobutanil", actual=actual)
    assert other.ratio == pytest.approx(1.0)                   # the rest of the stock is untouched
    # Conservation: what the tip did not pick up stayed in the standard vial.
    vessels, _ = run_recipe(actual.recipe, "actual")
    s0 = next(s for s in recipe.steps if s.op == "stock" and s.vessel == "Ochratoxin A (2 ug/mL)")
    initial = s0.contents_ng_ul["Ochratoxin A"] * s0.volume_ul.nominal
    moved_nominal = initial * recipe.steps[k].volume_ul.nominal / s0.volume_ul.nominal
    assert vessels["Ochratoxin A (2 ug/mL)"].ng["Ochratoxin A"] == pytest.approx(
        initial - 0.4 * moved_nominal, rel=1e-9)


def test_a_pipette_setting_changes_the_volume_moved_for_every_compound(inputs, recipe):
    k = next(k for k, s in enumerate(recipe.steps) if s.op == "inject" and s.vessel == "Column: Cal 3")
    actual = apply(inputs, recipe, {bench_key(k, "pipette"): 5.0}, "Myclobutanil")
    t = build_trace(inputs, recipe, "Column: Cal 3", "Myclobutanil", actual=actual)
    assert stop(t, f"step:{k}").loss == pytest.approx(0.5) and t.ratio == pytest.approx(0.5)
    other = build_trace(inputs, recipe, "Column: Cal 3", "Bifenthrin", actual=actual)
    assert other.ratio == pytest.approx(0.5)                   # uniform: the injection carries half of everything


def mixed_transfer(recipe):
    """The first transfer step that draws from a vessel holding many compounds, Bifenthrin among them."""
    vessels, _ = run_recipe(recipe, "nominal")
    return next(k for k, s in enumerate(recipe.steps)
                if s.op == "transfer" and len(vessels[s.source].ng) > 10 and "Bifenthrin" in vessels[s.source].ng)


def test_a_compound_scoped_factor_is_charged_to_that_compound_only(inputs, recipe):
    k = mixed_transfer(recipe)
    dest, src = recipe.steps[k].vessel, recipe.steps[k].source
    injection = next(s.vessel for s in recipe.steps if s.op == "inject" and k in lineage(recipe, s.vessel))
    settings = {bench_key(k, "precipitated", "Bifenthrin"): 0.3}
    actual = apply(inputs, recipe, settings, "Bifenthrin")
    t = build_trace(inputs, recipe, injection, "Bifenthrin", actual=actual)
    assert stop(t, f"step:{k}").loss == pytest.approx(0.7) and t.ratio == pytest.approx(0.7)
    # Every other compound in the same draw is untouched, in the destination and in the source.
    nominal, _ = run_recipe(recipe, "nominal")
    moved, _ = run_recipe(actual.recipe, "actual")
    for c, ng in nominal[dest].ng.items():
        assert moved[dest].ng[c] == pytest.approx(ng * (0.7 if c == "Bifenthrin" else 1.0), rel=1e-9), c
    for c, ng in nominal[src].ng.items():
        if c != "Bifenthrin":
            assert moved[src].ng[c] == pytest.approx(ng, rel=1e-9), c
    assert moved[src].ng["Bifenthrin"] > nominal[src].ng["Bifenthrin"]     # the crystals stayed behind
    assert build_trace(inputs, recipe, injection, "Myclobutanil", actual=actual).ratio == pytest.approx(1.0)
    # The same setting while following another compound: still Bifenthrin's, so Myclobutanil does not move.
    assert build_trace(inputs, recipe, injection, "Myclobutanil",
                       actual=apply(inputs, recipe, settings, "Myclobutanil")).ratio == pytest.approx(1.0)


def test_matrix_scale_is_a_property_of_the_sample(inputs, recipe):
    actual = apply(inputs, recipe, {"matrix_scale": 2.0}, "Myclobutanil")
    assert set(actual.run.mf_scale) == {a.name for a in inputs.analytes}
    assert all(v == 2.0 for v in actual.run.mf_scale.values())


def test_own_label_by_name():
    assert is_own_label("Myclobutanil", "Myclobutanil-D4")
    assert not is_own_label("Chlormequat chloride", "Chlormequat-D4")       # the salt name differs: not matched
    assert not is_own_label("Acephate", "Daminozide-D4") and not is_own_label("Avermectin B1a", "Ivermectin-D2")
    assert not is_own_label("Myclobutanil", None) and not is_own_label("Myclobutanil", "Myclobutanil")


def test_istd_note_says_whether_the_ratio_cancels_the_loss():
    u, c = FACTORS["pipette"], FACTORS["precipitated"]
    assert "No internal standard" in istd_note(u, None, True, False)
    assert "not in this vessel yet" in istd_note(u, "X-D4", False, True)
    assert "cancels it" in istd_note(u, "X-D4", True, False)
    assert "cancels most" in istd_note(c, "X-D4", True, True)
    assert "reaches the result" in istd_note(c, "Y-D4", True, False)


def test_stage_factors_land_on_their_stage(inputs, recipe):
    settings = {"q1_share": 0.15, "spray_share": 0.5, "carryover": 0.01, "dead_time_us": 50.0, "drift": -0.02}
    actual = apply(inputs, recipe, settings, "Myclobutanil", injection_index=4, previous_injection="Column: Cal 8")
    t = build_trace(inputs, recipe, "Column: Cal 2", "Myclobutanil", actual=actual)
    assert stop(t, "q1").loss == pytest.approx(0.5)
    assert stop(t, "spray").loss == pytest.approx(0.5)
    assert stop(t, "carryover").loss > 1.0
    assert stop(t, "dead_time").loss < 1.0
    assert stop(t, "drift").loss == pytest.approx(0.92)
    assert t.ratio == pytest.approx(math.prod(s.loss for s in t.stops))
    assert all(s.loss == pytest.approx(1.0) for s in t.stops if s.step is not None)


def test_column_factors_change_the_window_share_only_on_the_column(inputs, recipe):
    actual = apply(inputs, recipe, {"dwell_volume": 3.0}, "Myclobutanil")
    t = build_trace(inputs, recipe, "Column: Cal 5", "Myclobutanil", actual=actual)
    assert stop(t, "window").loss < 1.0
    assert all(s.loss == pytest.approx(1.0) for s in t.stops if s.step is not None or s.key in ("spray", "q1", "cell", "q3"))
    aged = apply(inputs, recipe, {"column_age": 0.2}, "Myclobutanil")
    t2 = build_trace(inputs, recipe, "Column: Cal 5", "Myclobutanil", actual=aged)
    assert stop(t2, "window").loss <= 1.0                       # wider peaks never gain window share


def test_extra_transitions_count_fewer_ions_but_report_the_same_area(inputs, recipe):
    actual = apply(inputs, recipe, {"extra_transitions": 60.0}, "Myclobutanil")
    t = build_trace(inputs, recipe, "Column: Cal 4", "Myclobutanil", actual=actual)
    assert stop(t, "dwell").loss < 0.9
    assert stop(t, "area").loss > 1.1                         # scaled back by cycle / dwell
    assert t.ratio == pytest.approx(1.0, rel=1e-6)


def test_default_settings_change_nothing(inputs, recipe):
    settings = {f.key: f.default for f in STAGE}
    actual = apply(inputs, recipe, settings, "Myclobutanil")
    t = build_trace(inputs, recipe, "Column: Cal 6", "Myclobutanil", actual=actual)
    assert t.first_departure() is None


def test_every_recipe_vessel_has_a_container(recipe):
    cat = load_containers(EXAMPLES / "containers.yaml")
    for vessel in dict.fromkeys(s.vessel for s in recipe.steps):
        assert cat.for_vessel(vessel) is not None, vessel
    assert cat.for_vessel("Column: Cal 1").kind == "loop" and cat.for_vessel("Tube: LCS").kind == "tube"
