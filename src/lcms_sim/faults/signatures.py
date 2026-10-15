"""Reduce a run to a feature vector, so simulated and observed results can be compared.

Every feature is a plain number. An observation from the lab may supply any subset of them.

| Feature | Meaning |
|---|---|
| sample_bias | median reported / assigned over analytes in real samples (kind "sample") |
| sample_bias_spread | robust spread (1.4826 x MAD) of log10(reported / assigned) across those analytes |
| bias_rt_corr | Spearman correlation of log bias with retention time |
| qc_recovery | median reported / expected over QCs (LCS, ICV, CCV) |
| low_high_ratio | median recovery at the LOQ QC / median recovery at the CCV |
| istd_area_ratio | median over IS of mean IS area in samples and QCs / mean in calibrators |
| istd_missing | fraction of IS with essentially no area in calibrators |
| istd_order_corr | Spearman correlation of IS area with injection order |
| rt_shift | median (actual - method) retention time, min |
| rt_shift_spread | spread of that shift across analytes, min |
| ion_ratio_fail_sample | fraction of ion ratios in samples outside +-30% of expected |
| ion_ratio_fail_cal | the same in calibrators |
| blank_positive | fraction of targets above their LOQ in any blank after the first injection |
| cal_low_error | median |accuracy - 100| (%) at the lowest calibrator |
| cal_high_error | the same at the highest calibrator |
| unreportable | fraction of target results that are not finite (injections that received IS) |
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from scipy.stats import spearmanr

from lcms_sim.io.loaders import InputSet
from lcms_sim.io.schemas import TransitionRole
from lcms_sim.pipeline import Kind, RunResult

FEATURES = (
    "sample_bias", "sample_bias_spread", "bias_rt_corr", "qc_recovery", "low_high_ratio",
    "istd_area_ratio", "istd_missing", "istd_order_corr", "rt_shift", "rt_shift_spread",
    "ion_ratio_fail_sample", "ion_ratio_fail_cal", "blank_positive", "cal_low_error", "cal_high_error",
    "unreportable",
)

# One line per feature, from the table above (the matcher's help text).
FEATURE_MEANING: dict[str, str] = {
    row[0].strip(): row[1].strip()
    for row in (line.strip().strip("|").split("|", 1) for line in __doc__.splitlines() if line.startswith("| "))
    if row[0].strip() in FEATURES
}


def _median(x: Iterable[float]) -> float:
    a = np.array([v for v in x if np.isfinite(v)])
    return float(np.median(a)) if a.size else math.nan


def _robust_sd(x: Iterable[float]) -> float:
    a = np.array([v for v in x if np.isfinite(v)])
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if a.size > 2 else math.nan


def _spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return math.nan
    return float(spearmanr(x[ok], y[ok]).statistic)


def extract(result: RunResult, inputs: InputSet) -> dict[str, float]:
    samples = {s.id: s for s in inputs.samples}
    targets = [a.name for a in inputs.analytes if not a.is_internal_standard]
    istds = [a.name for a in inputs.analytes if a.is_internal_standard]
    method_rt = {t.analyte: t.retention_time_min for t in inputs.transitions if t.role is TransitionRole.quantifier}
    f: dict[str, float] = {}

    def ratios(kind: Kind, ids: set[str] | None = None) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for r in result.injections:
            if r.item.kind is not kind or (ids and r.item.ref not in ids):
                continue
            true = samples[r.item.ref].true_concentration_ug_kg
            rep = result.reported_ug_kg.get(r.item.ref, {})
            for a, c in true.items():
                if c > 0 and a in rep:
                    out.setdefault(a, []).append(rep[a] / c)
        return out

    s_rat = {a: _median(v) for a, v in ratios(Kind.sample).items()}
    f["sample_bias"] = _median(s_rat.values())
    logs = {a: math.log10(v) for a, v in s_rat.items() if v > 0 and np.isfinite(v)}
    f["sample_bias_spread"] = _robust_sd(logs.values())
    f["bias_rt_corr"] = _spearman([result.retention_min[a] for a in logs], list(logs.values()))

    qc_ids = {s.id for s in inputs.samples if s.kind == "qc" and s.id.upper() != "LOQ"}
    f["qc_recovery"] = _median(v for vs in ratios(Kind.qc, qc_ids).values() for v in vs)
    loq = _median(v for vs in ratios(Kind.qc, {"LOQ"}).values() for v in vs)
    ccv = _median(v for vs in ratios(Kind.qc, {i for i in qc_ids if i.upper().startswith("CCV")}).values() for v in vs)
    f["low_high_ratio"] = loq / ccv if ccv and np.isfinite(ccv) else math.nan

    cal = [r for r in result.injections if r.item.kind is Kind.calibrator]
    non_cal = [r for r in result.injections if r.item.kind in (Kind.sample, Kind.qc)]
    is_ratio, missing = [], 0
    for i in istds:
        c = np.mean([r.areas.get(i, 0.0) for r in cal]) if cal else math.nan
        n = np.mean([r.areas.get(i, 0.0) for r in non_cal]) if non_cal else math.nan
        if not c or c <= 0:
            missing += 1
            continue
        is_ratio.append(n / c)
    f["istd_area_ratio"] = _median(is_ratio)
    f["istd_missing"] = missing / len(istds) if istds else math.nan
    with_is = [r for r in result.injections if any(r.areas.get(i, 0) > 0 for i in istds)]
    if istds and len(with_is) > 3:
        norm = [_median(r.areas.get(i, 0.0) / max(np.mean([q.areas.get(i, 0.0) for q in with_is]), 1e-30)
                        for i in istds) for r in with_is]
        f["istd_order_corr"] = _spearman([r.index for r in with_is], norm)
    else:
        f["istd_order_corr"] = math.nan

    shifts = [result.retention_min[a] - method_rt[a] for a in targets if a in method_rt]
    f["rt_shift"] = _median(shifts)
    f["rt_shift_spread"] = _robust_sd(shifts)

    expected = {j: t.relative_response for j, t in enumerate(inputs.transitions)
                if t.role is TransitionRole.qualifier and t.relative_response}
    quant_of = {j: t.analyte for j, t in enumerate(inputs.transitions) if j in expected}

    def ion_fail(kind: Kind, tolerance: float = 0.30) -> float:
        devs = []
        for r in result.injections:
            if r.item.kind is not kind:
                continue
            for j, q in r.qualifier_areas.items():
                a = quant_of.get(j)
                base = r.areas.get(a, 0.0) if a else 0.0
                if a and base > 0:
                    devs.append(abs(q / base / expected[j] - 1) > tolerance)
        return float(np.mean(devs)) if devs else math.nan

    f["ion_ratio_fail_sample"] = ion_fail(Kind.sample)
    f["ion_ratio_fail_cal"] = ion_fail(Kind.calibrator)

    levels = sorted(inputs.calibration_design, key=lambda lv: np.mean(list(lv.concentrations_ng_ml.values())))
    low = levels[0].concentrations_ng_ml if levels else {}
    from lcms_sim.quant.prep import back_calculate, prep_values
    nom = prep_values(inputs.prep, "nominal")
    frac = {a.name: a.isomer_fraction or 1.0 for a in inputs.analytes}
    loq_ug_kg = {a: back_calculate(c * frac.get(a, 1.0), nom) for a, c in low.items()}
    blanks_after = [r for r in result.injections if r.item.kind is Kind.blank and r.index > 0]
    pos = set()
    for r in blanks_after:
        for a, v in result.reported_ug_kg.get(r.item.ref, {}).items():
            if np.isfinite(v) and a in loq_ug_kg and v > loq_ug_kg[a]:
                pos.add(a)
    f["blank_positive"] = len(pos) / len(targets) if targets else math.nan

    acc_low, acc_high = [], []
    for fit in result.fits.values():
        if fit.accuracy_pct:
            i_lo, i_hi = int(np.argmin(fit.x)), int(np.argmax(fit.x))
            acc_low.append(abs(fit.accuracy_pct[i_lo] - 100))
            acc_high.append(abs(fit.accuracy_pct[i_hi] - 100))
    f["cal_low_error"] = _median(acc_low)
    f["cal_high_error"] = _median(acc_high)

    with_istd = {s.id for s in inputs.samples if s.add_istd}
    vals = [v for sid, rep in result.reported_ug_kg.items() if sid in with_istd for v in rep.values()]
    f["unreportable"] = float(np.mean([not np.isfinite(v) for v in vals])) if vals else math.nan
    return f
