"""Crosstalk: co-eluting compounds whose ions pass both of a transition's mass filters."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from lcms_sim.io.loaders import load_input_set
from lcms_sim.io.schemas import AcquisitionMethod, Transition
from lcms_sim.ms.acquisition import build_schedule
from lcms_sim.ms.crosstalk import crosstalk_map, window_transmission
from lcms_sim.ms.noise import PeakSpec, acquire
from lcms_sim.pipeline import Kind, RunConditions, run_sequence
from lcms_sim.rng import make_rng

EXAMPLES = Path(__file__).resolve().parents[1] / "data" / "examples"


def tr(analyte, prec, prod, role="quantifier", rel=1.0, polarity="positive"):
    return Transition.model_validate(dict(analyte=analyte, precursor_mz=prec, product_mz=prod, role=role,
                                          retention_time_min=5.0, window_min=1.0, relative_response=rel,
                                          polarity=polarity))


PAIR = [tr("A", 294.1, 125.0), tr("A", 294.1, 70.1, "qualifier", 0.5),
        tr("B", 294.1, 130.0), tr("B", 294.1, 70.2, "qualifier", 0.5)]


def test_window_is_half_height_at_half_fwhm():
    assert window_transmission(0.35, 0.7) == pytest.approx(0.5)
    assert window_transmission(0.0, 0.7) == 1.0


def test_shared_precursor_and_close_products_leak_both_ways():
    m = crosstalk_map(PAIR)
    t01 = window_transmission(0.1, 0.7)
    assert [(c.source, c.factor) for c in m[1]] == [("B", pytest.approx(0.5 * t01))]   # A's 70.1 gets B's 70.2
    assert [(c.source, c.factor) for c in m[3]] == [("A", pytest.approx(0.5 * t01))]
    assert 0 not in m and 2 not in m        # 125 vs 130: Q3 separates the quantifiers


def test_no_leak_across_polarity_or_far_precursor():
    assert crosstalk_map([tr("A", 294.1, 70.1), tr("B", 294.1, 70.1, polarity="negative")]) == {}
    assert crosstalk_map([tr("A", 294.1, 70.1), tr("B", 296.1, 70.1)]) == {}


def test_example_method_has_no_strong_pair():
    """The application note's transitions share precursors only between peaks that do not co-elute: every
    transition's leak stays below 1% of the source's listed response."""
    i = load_input_set(EXAMPLES)
    m = crosstalk_map(i.transitions)
    assert all(c.factor < 0.01 for cs in m.values() for c in cs)
    assert {(i.transitions[j].analyte, c.source) for j, cs in m.items() for c in cs} == {
        ("Bifenazate", "Prallethrin"), ("Prallethrin", "Bifenazate")}


def with_a_pair(i):
    """The example with Myclobutanil-D4 moved onto paclobutrazol: same precursor, a 70.2 product within Q3's window
    of paclobutrazol's 70.1 qualifier, and the same retention. A labelled standard placed like this is the case the
    crosstalk model is for."""
    pac = next(a for a in i.analytes if a.name == "Paclobutrazol")
    rt = next(t.retention_time_min for t in i.transitions if t.analyte == "Paclobutrazol")
    analytes = [a.model_copy(update={"log_kw": pac.log_kw, "S": pac.S}) if a.name == "Myclobutanil-D4" else a
                for a in i.analytes]
    transitions = [t.model_copy(update={"precursor_mz": 294.1, "product_mz": 70.2, "retention_time_min": rt})
                   if t.analyte == "Myclobutanil-D4" else t for t in i.transitions]
    return dataclasses.replace(i, analytes=analytes, transitions=transitions)


def test_a_labelled_standard_on_the_same_ions_is_found_both_ways():
    i = with_a_pair(load_input_set(EXAMPLES))
    strong = {(i.transitions[j].analyte, c.source) for j, cs in crosstalk_map(i.transitions).items()
              for c in cs if c.factor > 0.01}
    assert strong == {("Paclobutrazol", "Myclobutanil-D4"), ("Myclobutanil-D4", "Paclobutrazol")}


def test_acquired_area_adds_the_leak():
    acq = AcquisitionMethod(cycle_time_ms=100.0, inter_scan_delay_ms=1.0, gain_area_per_ion=1e-3)
    s = build_schedule(PAIR, acq, 10.0)
    pa, pb = PeakSpec("A", 5.0, 0.03, 2e4), PeakSpec("B", 5.0, 0.03, 2e5)
    m = crosstalk_map(PAIR)
    leaks = {j: [({"A": pa, "B": pb}[c.source], {"A": pa, "B": pb}[c.source].quant_area * c.factor) for c in cs]
             for j, cs in m.items()}
    areas = np.array([[x.area for x in acquire([pa, pb], PAIR, s, acq, make_rng(k), crosstalk=leaks)]
                      for k in range(20)]).mean(axis=0)
    t01 = window_transmission(0.1, 0.7)
    assert areas[1] == pytest.approx(2e4 * 0.5 + 2e5 * 0.5 * t01, rel=0.02)
    assert areas[0] == pytest.approx(2e4, rel=0.02)


def test_internal_standard_inflates_paclobutrazol_ion_ratio_most_at_low_levels():
    i = with_a_pair(load_input_set(EXAMPLES))
    j = next(k for k, t in enumerate(i.transitions) if t.analyte == "Paclobutrazol" and t.role.value == "qualifier")
    expected = i.transitions[j].relative_response

    def ratio(inputs, level):
        r = run_sequence(inputs, make_rng(0), conditions=RunConditions(detector="expected"))
        rec = next(x for x in r.injections if x.item.kind is Kind.calibrator and x.item.ref == level)
        return rec.qualifier_areas[j] / rec.areas["Paclobutrazol"] / expected

    def with_xt(on: bool, y: float = 1.0):
        return dataclasses.replace(i, acquisition=i.acquisition.model_copy(update={"crosstalk": on,
                                                                                   "crosstalk_yield": y}))

    assert not i.acquisition.crosstalk            # off by default: its size is unknown until measured
    assert ratio(i, "Cal 1") == pytest.approx(1.0) and ratio(i, "Cal 8") == pytest.approx(1.0)
    low, high = ratio(with_xt(True), "Cal 1"), ratio(with_xt(True), "Cal 8")
    assert low > high > 1.0
    assert low > 1.3     # outside the +-30% ion ratio tolerance at the lowest calibrator
    # The excess scales with the yield.
    assert ratio(with_xt(True, 0.01), "Cal 1") - 1 == pytest.approx(0.01 * (low - 1), rel=1e-6)
