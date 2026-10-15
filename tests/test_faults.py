"""Phase 4: every fault reproduces its listed signature."""

import dataclasses
import math

import numpy as np
import pytest

from lcms_sim.faults.library import LIBRARY, Scenario
from lcms_sim.faults.signatures import FEATURES, extract
from lcms_sim.io.loaders import load_input_set
from lcms_sim.io.schemas import ISTDStage, NominalActual, SampleDefinition
from lcms_sim.paths import EXAMPLES
from lcms_sim.pipeline import Kind, SequenceItem
from lcms_sim.rng import make_rng


@pytest.fixture(scope="module")
def inputs():
    return load_input_set(EXAMPLES)


def run(scn, seed=0):
    r = scn.simulate(make_rng(seed))
    return r, extract(r, scn.inputs)


@pytest.fixture(scope="module")
def base(inputs):
    return run(Scenario.baseline(inputs, detector="expected"))


def rep(r, sid, a):
    return r.reported_ug_kg[sid][a]


PT = "PT sample"


def pt_bias(r, inputs):
    true = next(s for s in inputs.samples if s.id == PT).true_concentration_ug_kg
    return {a: rep(r, PT, a) / c for a, c in true.items()}


def test_every_fault_documented():
    assert len(LIBRARY) >= 16
    for f in LIBRARY.values():
        assert f.signature and f.stage and f.low <= f.default <= f.high
        assert f.scope in ("uniform", "compound") and (f.scope == "compound") == bool(f.varies_with), f.key
        if "Uniform" in f.signature or "All raw areas" in f.signature:
            assert f.scope == "uniform", f.key
        if "specific" in f.signature or "depends on" in f.signature or "analytes only" in f.signature:
            assert f.scope == "compound", f.key


@pytest.mark.parametrize("key", [k for k, f in LIBRARY.items() if f.scope == "uniform"])
def test_uniform_faults_move_every_compound_by_the_same_share(inputs, key):
    _, f = run(LIBRARY[key].apply(Scenario.baseline(inputs, detector="expected")))
    assert f["sample_bias_spread"] < 1e-9, key


def test_baseline_is_clean(base):
    _, f = base
    assert set(f) == set(FEATURES)
    assert f["sample_bias"] == pytest.approx(1.0, rel=1e-6)
    assert f["qc_recovery"] == pytest.approx(1.0, rel=1e-6)
    assert f["istd_area_ratio"] == pytest.approx(1.0, rel=1e-6)
    assert f["istd_missing"] == 0 and f["blank_positive"] == 0 and f["unreportable"] == 0
    # Method RTs are the SOP's; the lab batch elutes a median 0.03 min later than the SOP table.
    assert abs(f["rt_shift"]) < 0.05


# --- calibration -------------------------------------------------------------


def test_stock_strong_hides_from_qcs(inputs):
    _, f = run(LIBRARY["stock_strong"].apply(Scenario.baseline(inputs, detector="expected"), 1.5))
    assert f["sample_bias"] == pytest.approx(1 / 1.5, rel=1e-6)
    assert f["qc_recovery"] == pytest.approx(1.0, rel=1e-6)       # QCs come from the same stock
    assert f["istd_area_ratio"] == pytest.approx(1.0, rel=1e-6)


def test_stock_degraded_hits_listed_analytes_only(inputs):
    fault = LIBRARY["stock_degraded"].with_analytes(["Bifenazate"])
    r, _ = run(fault.apply(Scenario.baseline(inputs, detector="expected"), 0.6))
    b = pt_bias(r, inputs)
    assert b["Bifenazate"] == pytest.approx(1 / 0.6, rel=1e-6)
    assert b["Imidacloprid"] == pytest.approx(1.0, rel=1e-6)


def test_wrong_weighting_fails_low_levels(inputs):
    good, bad = [], []
    for seed in range(3):
        _, f0 = run(Scenario.baseline(inputs), seed)
        _, f1 = run(LIBRARY["wrong_weighting"].apply(Scenario.baseline(inputs)), seed)
        good.append(f0["cal_low_error"])
        bad.append(f1["cal_low_error"])
        assert f1["cal_high_error"] < 5
    assert np.mean(bad) > 2 * np.mean(good)


def test_isomer_share_biases_one_component(inputs):
    fault = LIBRARY["isomer_share"].with_analytes(["Spinosyn A"])
    r, _ = run(fault.apply(Scenario.baseline(inputs, detector="expected"), 0.8))
    b = pt_bias(r, inputs)
    assert b["Spinosyn A"] == pytest.approx(1 / 0.8, rel=1e-6)
    assert b["Spinosyn D"] == pytest.approx(1.0, rel=1e-6)
    assert r.reported_ug_kg["CCV 1"]["Spinosyn A"] == pytest.approx(
        next(s for s in inputs.samples if s.id == "CCV 1").true_concentration_ug_kg["Spinosyn A"], rel=1e-6)


# --- internal standard ---------------------------------------------------------


def test_istd_overspiked(inputs):
    _, f = run(LIBRARY["istd_overspiked"].apply(Scenario.baseline(inputs, detector="expected"), 1.5))
    assert f["sample_bias"] == pytest.approx(1 / 1.5, rel=1e-6)
    assert f["istd_area_ratio"] == pytest.approx(1.5, rel=1e-3)


def test_istd_late_leaves_recovery_uncorrected(inputs):
    """Method with IS before extraction and solvent calibrators; each IS recovers like its analytes."""
    rec = {}
    for a in inputs.analytes:
        key = a.name if a.is_internal_standard else a.internal_standard
        rec[a.name] = 0.5 + 0.45 * (hash(key) % 1000) / 1000
    prep = inputs.prep.model_copy(update={"istd_stage": ISTDStage.before_extraction, "recovery": rec})
    cal = inputs.calibration.model_copy(update={"preparation": "vial", "matrix_matched": False})
    method = dataclasses.replace(inputs, prep=prep, calibration=cal)
    r0, _ = run(Scenario.baseline(method, detector="expected"))
    r1, _ = run(LIBRARY["istd_late"].apply(Scenario.baseline(method, detector="expected")))
    b0, b1 = pt_bias(r0, method), pt_bias(r1, method)
    for a in b1:
        assert b1[a] / b0[a] == pytest.approx(rec[a], rel=1e-6)
    assert np.std(list(b1.values())) > 0.05


def test_istd_window_miss(inputs):
    fault = LIBRARY["istd_window_miss"].with_analytes(["Carbaryl-D7"])
    r, f = run(fault.apply(Scenario.baseline(inputs, detector="expected"), 2.0))
    assert f["istd_missing"] > 0
    assert not math.isfinite(r.reported_ug_kg["CCV 1"]["Carbaryl"])
    assert math.isfinite(r.reported_ug_kg["CCV 1"]["Imidacloprid"])


# --- prep ----------------------------------------------------------------------


def test_low_recovery_is_analyte_specific_and_qcs_pass(inputs):
    _, f = run(LIBRARY["low_recovery"].apply(Scenario.baseline(inputs, detector="expected"), 0.6))
    assert f["sample_bias"] < 0.9
    assert f["sample_bias_spread"] > 0.02
    assert f["qc_recovery"] == pytest.approx(1.0, rel=1e-6)


def test_volume_error_is_uniform(inputs):
    _, f = run(LIBRARY["volume_error"].apply(Scenario.baseline(inputs, detector="expected"), 1.5))
    assert f["sample_bias"] == pytest.approx(1 / 1.5, rel=1e-6)
    assert f["sample_bias_spread"] < 1e-6
    assert f["qc_recovery"] == pytest.approx(1 / 1.5, rel=1e-6)


def test_low_injection_volume_corrected_by_istd(inputs):
    _, f = run(LIBRARY["low_injection"].apply(Scenario.baseline(inputs, detector="expected"), 0.5))
    assert f["sample_bias"] == pytest.approx(1.0, rel=1e-6)
    assert f["istd_area_ratio"] == pytest.approx(0.5, rel=1e-3)


# --- ionisation and detection ------------------------------------------------------


def test_matrix_suppression_depends_on_retention_and_istd(inputs, base):
    r, f = run(LIBRARY["matrix_suppression"].apply(Scenario.baseline(inputs, detector="expected"), 1.5))
    b = pt_bias(r, inputs)
    assert b["Imidacloprid"] == pytest.approx(1.0, abs=0.01)     # co-eluting labelled IS
    assert b["Pyridaben"] == pytest.approx(1.0, abs=0.04)        # its own labelled IS, 0.01 min earlier
    assert abs(b["Acequinocyl"] - 1) > 0.05                      # IS (Permethrin-D5) 0.9 min earlier, less suppressed
    assert f["sample_bias_spread"] > base[1]["sample_bias_spread"]
    assert f["istd_area_ratio"] < 1.0


def test_interference_in_samples_only(inputs, base):
    fault = LIBRARY["interference"].with_analytes(["Imidacloprid"])
    r, f = run(fault.apply(Scenario.baseline(inputs, detector="expected"), 50.0))
    assert rep(r, "Flower A", "Imidacloprid") > 40       # blank flower reads ~50 ug/kg
    assert rep(r, "CCV 1", "Imidacloprid") == pytest.approx(rep(base[0], "CCV 1", "Imidacloprid"))
    assert f["ion_ratio_fail_sample"] > base[1]["ion_ratio_fail_sample"]
    # Calibrators are untouched by a sample-only interference.
    assert f["ion_ratio_fail_cal"] == base[1]["ion_ratio_fail_cal"]


def test_saturation_bends_curve_top(inputs):
    r0, _ = run(Scenario.baseline(inputs), 1)
    r1, _ = run(LIBRARY["saturation"].apply(Scenario.baseline(inputs), 2e-7), 1)

    def top_over_low(r):
        out = []
        for fit in r.fits.values():
            i_lo, i_hi = int(np.argmin(fit.x)), int(np.argmax(fit.x))
            out.append((fit.y[i_hi] / fit.x[i_hi]) / (fit.y[i_lo] / fit.x[i_lo]))
        return np.median(out)

    assert top_over_low(r1) < top_over_low(r0) - 0.05


def test_drift_trends_istd_but_results_hold(inputs):
    _, f = run(LIBRARY["drift"].apply(Scenario.baseline(inputs, detector="expected"), -0.03))
    assert f["istd_order_corr"] < -0.8
    assert f["sample_bias"] == pytest.approx(1.0, rel=1e-6)


def test_carryover_makes_blanks_positive(inputs):
    _, f = run(LIBRARY["carryover"].apply(Scenario.baseline(inputs, detector="expected"), 0.01))
    assert f["blank_positive"] > 0.5


def test_too_many_transitions_costs_precision_not_mean(inputs):
    """At the LOQ: recoveries scatter more across analytes, centred on 100%."""
    def loq_spread(scn):
        out, centre = [], []
        for seed in range(3):
            r, _ = run(scn, seed)
            true = next(s for s in inputs.samples if s.id == "LOQ").true_concentration_ug_kg
            x = np.array([r.reported_ug_kg["LOQ"][a] / c for a, c in true.items()])
            out.append(np.std(x))
            centre.append(np.median(x))
        return np.mean(out), np.mean(centre)

    s0, c0 = loq_spread(Scenario.baseline(inputs))
    s1, c1 = loq_spread(LIBRARY["too_many_transitions"].apply(Scenario.baseline(inputs), 300))
    # 300 extra transitions cut the median analyte's dwell 2.5-fold: counting alone would scale the spread by
    # about 1.58; injection and point noise do not scale, and three seeds leave the estimate +-10%.
    assert s1 > 1.3 * s0
    assert c1 == pytest.approx(1.0, abs=0.05)


# --- LC --------------------------------------------------------------------------


def test_ph_error_moves_ionisable_analytes(inputs):
    r, f = run(LIBRARY["ph_error"].apply(Scenario.baseline(inputs, detector="expected"), 1.0))
    r0 = run(Scenario.baseline(inputs, detector="expected"))[0]
    shift = {a: r.retention_min[a] - r0.retention_min[a] for a in r.retention_min}
    assert abs(shift["Ochratoxin A"]) > 0.05     # pKa 4.4: a unit up from pH 3 ionises a quarter of it
    assert abs(shift["Tebuconazole"]) < 1e-6


def test_organic_error_shifts_everything_earlier(inputs):
    _, f = run(LIBRARY["organic_error"].apply(Scenario.baseline(inputs, detector="expected"), 0.03))
    assert f["rt_shift"] < -0.05


def test_dwell_volume_shifts_later(inputs):
    _, f = run(LIBRARY["dwell_volume"].apply(Scenario.baseline(inputs, detector="expected"), 2.0))
    assert f["rt_shift"] > 0.3


def test_large_rt_shift_clips_peaks_and_reads_low(inputs):
    """Calibrators and samples shift together, so the clipped fraction mostly cancels in the result,
    but the clipped analytes lose signal (IS-normalised areas fall where only the analyte clips)."""
    r, f = run(LIBRARY["dwell_volume"].apply(Scenario.baseline(inputs, detector="expected"), 3.0))
    cal_top = [x for x in r.injections if x.item.kind is Kind.calibrator][-1]
    base_top = [x for x in run(Scenario.baseline(inputs, detector="expected"))[0].injections
                if x.item.kind is Kind.calibrator][-1]
    lost = [a for a in base_top.areas if cal_top.areas.get(a, 0) < 0.9 * base_top.areas[a]]
    assert lost
