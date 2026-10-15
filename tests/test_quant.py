"""Phase 3: calibration fits and the quantitation chain against the bias relation."""

import dataclasses
import itertools
import math

import numpy as np
import pytest

from lcms_sim.io.loaders import load_input_set
from lcms_sim.io.schemas import (
    Analyte,
    CalibrationLevel,
    CalibrationModel,
    ISTDStage,
    NominalActual,
    OriginHandling,
    SampleDefinition,
    Transition,
    Weighting,
)
from lcms_sim.paths import EXAMPLES
from lcms_sim.pipeline import Kind, RunConditions, SequenceItem, run_sequence
from lcms_sim.quant.calibration import fit_calibration
from lcms_sim.quant.prep import walk
from lcms_sim.rng import make_rng

# --- calibration -----------------------------------------------------------

X = np.array([0.05, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0])


@pytest.mark.parametrize("weighting,origin", list(itertools.product(Weighting, OriginHandling)))
def test_noise_free_linear_recovered(weighting, origin):
    intercept = 0.0 if origin is not OriginHandling.ignore else 37.0
    y = intercept + 2500.0 * X
    fit = fit_calibration(X, y, CalibrationModel.linear, weighting, origin)
    assert fit.slope == pytest.approx(2500.0, rel=1e-12)
    assert fit.intercept == pytest.approx(intercept, abs=1e-9)
    assert fit.r2 == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(fit.accuracy_pct, 100.0, atol=1e-9)


@pytest.mark.parametrize("weighting", list(Weighting))
def test_noise_free_quadratic_recovered(weighting):
    y = 12.0 + 2500.0 * X - 150.0 * X * X
    fit = fit_calibration(X, y, CalibrationModel.quadratic, weighting)
    assert fit.coef == pytest.approx((12.0, 2500.0, -150.0), rel=1e-9)
    assert fit.inverse(float(fit.predict(3.0))) == pytest.approx(3.0, rel=1e-12)


def exact_wls(x, y, w, degree):
    """Independent reference: weighted normal equations solved in exact rational arithmetic."""
    from fractions import Fraction as F

    A = [[F(float(xi)) ** p for p in range(degree + 1)] for xi in x]
    W, Y = [F(float(v)) for v in w], [F(float(v)) for v in y]
    n = degree + 1
    M = [[sum(W[k] * A[k][i] * A[k][j] for k in range(len(x))) for j in range(n)] for i in range(n)]
    b = [sum(W[k] * A[k][i] * Y[k] for k in range(len(x))) for i in range(n)]
    for i in range(n):
        for r in range(i + 1, n):
            f = M[r][i] / M[i][i]
            M[r] = [M[r][c] - f * M[i][c] for c in range(n)]
            b[r] -= f * b[i]
    out = [F(0)] * n
    for i in reversed(range(n)):
        out[i] = (b[i] - sum(M[i][c] * out[c] for c in range(i + 1, n))) / M[i][i]
    return tuple(float(v) for v in out)


@pytest.mark.parametrize("weighting,model", list(itertools.product(Weighting, CalibrationModel)))
def test_weighted_fit_matches_independent_routine(weighting, model):
    rng = np.random.default_rng(0)
    y = (5.0 + 2500.0 * X - 40.0 * X * X) * (1 + rng.normal(0, 0.05, X.size))
    fit = fit_calibration(X, y, model, weighting)
    w = {Weighting.none: np.ones_like(X), Weighting.inv_x: 1 / X, Weighting.inv_x2: 1 / X**2}[weighting]
    ref = exact_wls(X, y, w, 1 if model is CalibrationModel.linear else 2)
    assert fit.coef == pytest.approx(ref, rel=1e-9)


def test_forced_origin_has_no_intercept():
    fit = fit_calibration(X, 30 + 2500 * X, origin=OriginHandling.force)
    assert fit.intercept == 0.0


# --- chain fixture ---------------------------------------------------------

TARGETS = ["Acephate", "Imidacloprid", "Tebuconazole", "Bifenthrin"]
TRUE = 100.0  # ug/kg


@pytest.fixture(scope="module")
def base():
    return load_input_set(EXAMPLES)


def make_inputs(base, *, istd=True, stage=ISTDStage.before_extraction, coelute=True, **prep_kw):
    """Four targets, optionally each with a co-eluting labelled internal standard."""
    keep = [a for a in base.analytes if a.name in TARGETS]
    analytes, transitions = [], []
    for i, a in enumerate(keep):
        is_name = f"{a.name}-IS"
        src = a if coelute else keep[(i + 2) % len(keep)]  # a non-matching IS elutes elsewhere
        analytes.append(a.model_copy(update={"internal_standard": is_name if istd else None}))
        transitions += [t for t in base.transitions if t.analyte == a.name]
        if istd:
            analytes.append(Analyte(name=is_name, is_internal_standard=True, log_kw=src.log_kw, S=src.S,
                                    response_factor=a.response_factor))
            q = next(t for t in base.transitions if t.analyte == src.name and t.role.value == "quantifier")
            transitions.append(q.model_copy(update={"analyte": is_name, "precursor_mz": q.precursor_mz + 4}))
    prep = base.prep.model_copy(update={
        "istd_spike_ul": NominalActual(nominal=25.0) if istd else None,
        "istd_mix_ng_ul": {f"{a}-IS": 4.0 for a in TARGETS} if istd else {},
        "istd_stage": stage if istd else None,
        "vial_volume_ml": NominalActual(nominal=0.5),
        **prep_kw,
    })
    cal = base.calibration.model_copy(update={
        "use_internal_standard": istd,
        "istd_spike_ul": NominalActual(nominal=25.0) if istd else None,
        "preparation": "vial",
    })
    levels = [CalibrationLevel(level=l.level, concentrations_ng_ml={a: l.concentrations_ng_ml[a] for a in TARGETS})
              for l in base.calibration_design]
    samples = [SampleDefinition(id="blank", matrix="cannabis flower"),
               SampleDefinition(id="s1", matrix="cannabis flower", true_concentration_ug_kg={a: TRUE for a in TARGETS})]
    return dataclasses.replace(base, analytes=analytes, transitions=transitions, prep=prep, calibration=cal,
                               calibration_design=levels, samples=samples)


def reported(inputs, **cond):
    r = run_sequence(inputs, make_rng(0), conditions=RunConditions(detector="expected", **cond))
    return r, {a: r.reported_ug_kg["s1"][a] / TRUE for a in TARGETS}


def na(n, a):
    return NominalActual(nominal=n, actual=a)


# --- chain: no faults ------------------------------------------------------


@pytest.mark.parametrize("istd", [True, False])
def test_no_faults_reports_true_value(base, istd):
    _, bias = reported(make_inputs(base, istd=istd))
    for a in TARGETS:
        assert bias[a] == pytest.approx(1.0, rel=1e-9), a


def test_blank_reports_zero(base):
    r, _ = reported(make_inputs(base))
    assert all(abs(v) < 1e-9 for v in r.reported_ug_kg["blank"].values())


def test_no_faults_with_noise_is_unbiased(base):
    inputs = make_inputs(base, istd=False)
    rel = []
    for seed in range(8):
        r = run_sequence(inputs, make_rng(seed))
        rel += [r.reported_ug_kg["s1"][a] / TRUE for a in TARGETS]
    assert np.mean(rel) == pytest.approx(1.0, abs=0.04)


# --- chain: the bias relation ----------------------------------------------


def bias_relation(R_a=1.0, MF_as=1.0, R_i=1.0, MF_is=1.0, MF_ic=1.0, MF_ac=1.0,
                  std_nom_over_act=1.0, I_c=1.0, I_s=1.0, f_act_over_nom=1.0):
    return (R_a * MF_as) / (R_i * MF_is) * (MF_ic / MF_ac) * std_nom_over_act * (I_c / I_s) * f_act_over_nom


def test_stock_stronger_than_labelled(base):
    inputs = make_inputs(base)
    inputs = dataclasses.replace(inputs, calibration=inputs.calibration.model_copy(
        update={"stock_concentration_ng_ml": na(1000.0, 1250.0)}))
    _, bias = reported(inputs)
    for a in TARGETS:
        assert bias[a] == pytest.approx(bias_relation(std_nom_over_act=1000 / 1250), rel=1e-3)


def test_stock_degraded_for_one_analyte(base):
    inputs = make_inputs(base)
    inputs = dataclasses.replace(inputs, calibration=inputs.calibration.model_copy(
        update={"stock_factor_actual": {"Tebuconazole": 0.5}}))
    _, bias = reported(inputs)
    assert bias["Tebuconazole"] == pytest.approx(bias_relation(std_nom_over_act=2.0), rel=1e-3)
    assert bias["Acephate"] == pytest.approx(1.0, rel=1e-3)


def test_istd_overspiked_in_samples(base):
    _, bias = reported(make_inputs(base, istd_spike_ul=na(25.0, 32.5)))
    for a in TARGETS:
        assert bias[a] == pytest.approx(bias_relation(I_s=1.3), rel=1e-3)


def test_istd_overspiked_in_calibrators(base):
    inputs = make_inputs(base)
    inputs = dataclasses.replace(inputs, calibration=inputs.calibration.model_copy(
        update={"istd_spike_ul": na(25.0, 27.5)}))
    _, bias = reported(inputs)
    for a in TARGETS:
        assert bias[a] == pytest.approx(bias_relation(I_c=1.1), rel=1e-3)


@pytest.mark.parametrize("stage,R_i,expected_R_i", [
    (ISTDStage.before_extraction, 0.6, 0.6),   # matched IS corrects recovery
    (ISTDStage.before_extraction, 1.0, 1.0),   # IS recovers fully, analyte does not
    (ISTDStage.after_extraction, 0.6, 1.0),    # IS never sees extraction
])
def test_recovery(base, stage, R_i, expected_R_i):
    rec = {a: 0.6 for a in TARGETS} | {f"{a}-IS": R_i for a in TARGETS}
    _, bias = reported(make_inputs(base, stage=stage, recovery=rec))
    for a in TARGETS:
        assert bias[a] == pytest.approx(bias_relation(R_a=0.6, R_i=expected_R_i), rel=1e-3)


def test_recovery_without_istd(base):
    _, bias = reported(make_inputs(base, istd=False, recovery={a: 0.6 for a in TARGETS}))
    for a in TARGETS:
        assert bias[a] == pytest.approx(0.6, rel=1e-3)


@pytest.mark.parametrize("stage,expected", [
    (ISTDStage.before_extraction, 1.0),        # dilution after IS: cancels
    (ISTDStage.in_vial, 10 / 12),              # dilution before IS: analyte only
])
def test_dilution_error(base, stage, expected):
    _, bias = reported(make_inputs(base, stage=stage, dilution_factor=na(10.0, 12.0)))
    for a in TARGETS:
        assert bias[a] == pytest.approx(bias_relation(f_act_over_nom=expected), rel=1e-3)


def test_mass_error_always_biases(base):
    _, bias = reported(make_inputs(base, sample_mass_g=na(1.0, 0.9)))
    for a in TARGETS:
        assert bias[a] == pytest.approx(bias_relation(f_act_over_nom=0.9), rel=1e-3)


@pytest.mark.parametrize("istd", [True, False])
def test_low_injection_volume(base, istd):
    _, bias = reported(make_inputs(base, istd=istd, injection_volume_ul=na(10.0, 8.0)))
    for a in TARGETS:
        assert bias[a] == pytest.approx(1.0 if istd else 0.8, rel=1e-3)


def _solvent_cal(inputs):
    return dataclasses.replace(inputs, calibration=inputs.calibration.model_copy(update={"matrix_matched": False}))


def test_matrix_suppression_solvent_calibration_no_istd(base):
    r, bias = reported(_solvent_cal(make_inputs(base, istd=False)))
    mf = r.record("s1").matrix_factor
    for a in TARGETS:
        assert bias[a] == pytest.approx(bias_relation(MF_as=mf[a]), rel=1e-3)
    assert min(mf[a] for a in TARGETS) < 0.9  # the example profile does suppress something


def test_matrix_suppression_corrected_by_coeluting_istd(base):
    _, bias = reported(_solvent_cal(make_inputs(base)))
    for a in TARGETS:
        assert bias[a] == pytest.approx(1.0, rel=1e-3)


def test_matrix_suppression_with_istd_eluting_elsewhere(base):
    r, bias = reported(_solvent_cal(make_inputs(base, coelute=False)))
    mf = r.record("s1").matrix_factor
    for a in TARGETS:
        assert bias[a] == pytest.approx(bias_relation(MF_as=mf[a], MF_is=mf[f"{a}-IS"]), rel=1e-3)


def test_matrix_matched_calibration_corrects_without_istd(base):
    _, bias = reported(make_inputs(base, istd=False))  # example calibration is matrix-matched
    for a in TARGETS:
        assert bias[a] == pytest.approx(1.0, rel=1e-3)


# --- sequence effects ------------------------------------------------------


def test_carryover_into_blank(base):
    inputs = make_inputs(base, istd=False)
    seq = [SequenceItem(Kind.calibrator, l.level) for l in inputs.calibration_design]
    seq += [SequenceItem(Kind.sample, "s1"), SequenceItem(Kind.blank, "blank")]
    r = run_sequence(inputs, make_rng(0), seq, RunConditions(detector="expected", carryover_fraction=0.01))
    s1, blank = r.record("s1"), r.record("blank")
    for a in TARGETS:
        assert blank.on_column_ng[a] == pytest.approx(0.01 * s1.on_column_ng[a])
        assert r.reported_ug_kg["blank"][a] > 0.005 * TRUE


def test_drift_trends_raw_areas_but_ratio_holds(base):
    inputs = make_inputs(base)
    seq = [SequenceItem(Kind.calibrator, l.level) for l in inputs.calibration_design]
    seq += [SequenceItem(Kind.sample, "s1")] * 5
    r = run_sequence(inputs, make_rng(0), seq, RunConditions(detector="expected", drift_per_injection=-0.03))
    s1 = [x for x in r.injections if x.item.ref == "s1"]
    assert s1[-1].areas["Imidacloprid"] < s1[0].areas["Imidacloprid"]
    assert r.reported_ug_kg["s1"]["Imidacloprid"] == pytest.approx(TRUE, rel=1e-9)


# --- walk-through ----------------------------------------------------------


def test_walk_shows_nominal_and_actual(base):
    inputs = make_inputs(base, dilution_factor=na(10.0, 12.0))
    stages = walk(TRUE, inputs.prep, recovery=0.8)
    vial = next(s for s in stages if s.stage == "Dilution")
    assert vial.nominal == pytest.approx(TRUE * 1.0 / (25 * 10))
    assert vial.actual == pytest.approx(TRUE * 1.0 * 0.8 / (25 * 12))
    assert math.isclose(stages[1].nominal, 25.0)  # IS addition shown


# --- SOP example: procedural calibration and isomers -----------------------


def test_isomer_totals_reported_correctly(base):
    r = run_sequence(base, make_rng(0), conditions=RunConditions(detector="expected"))
    ccv = next(s for s in base.samples if s.id == "CCV 1")
    reg = {a.name: a.regulated_as for a in base.analytes}
    true_tot = {}
    for a, c in ccv.true_concentration_ug_kg.items():
        true_tot[reg[a]] = true_tot.get(reg[a], 0.0) + c
    got = r.totals(base, "CCV 1")
    for name in ("Pyrethrins (sum of 6)", "Spinosad", "Abamectin", "Bifenazate"):
        assert got[name] == pytest.approx(true_tot[name], rel=1e-9), name


def test_procedural_calibration_cancels_recovery_without_istd(base):
    inputs = make_inputs(base, istd=False, recovery={a: 0.7 for a in TARGETS})
    inputs = dataclasses.replace(inputs, calibration=inputs.calibration.model_copy(update={"preparation": "procedural"}))
    _, bias = reported(inputs)
    for a in TARGETS:
        assert bias[a] == pytest.approx(1.0, rel=1e-9)
