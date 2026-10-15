"""Glue between layers.

    inputs -> LC peaks -> acquired signals            (one injection)
    sequence of injections -> calibration -> reported concentrations   (one run)

A run is simulated as a whole, in order, because carryover and sensitivity drift depend on
what was injected before.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np

from lcms_sim.io.loaders import InputSet
from lcms_sim.io.schemas import ISTDStage, SampleDefinition, TransitionRole
from lcms_sim.lc.gradient import LCConditions
from lcms_sim.lc.peaks import peak_sigma
from lcms_sim.lc.retention import retention_time
from lcms_sim.ms.acquisition import Schedule, build_schedule
from lcms_sim.ms.crosstalk import Crosstalk, crosstalk_map
from lcms_sim.ms.noise import PeakSpec, TransitionSignal, acquire, integration_range
from lcms_sim.quant.calibration import CalibrationFit, fit_calibration
from lcms_sim.quant.istd import response
from lcms_sim.quant.matrix import MatrixProfile, profile_from_traces, unity_profile
from lcms_sim.quant.prep import analyte_vial_conc, back_calculate, istd_vial_conc, on_column_ng, prep_values

# ---------------------------------------------------------------------------
# One injection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Injection:
    peaks: list[PeakSpec]
    schedule: Schedule
    signals: list[TransitionSignal]


def lc_peaks(
    inputs: InputSet,
    amounts_ng: Mapping[str, float],
    cond: LCConditions | None = None,
    matrix_factors: Mapping[str, float] | None = None,
    tau_min: float = 0.0,
    sensitivity: float = 1.0,
) -> list[PeakSpec]:
    """Where each analyte elutes, how wide it is, and its expected quantifier area."""
    cond = cond or LCConditions.from_method(inputs.system, inputs.lc_method)
    mf = matrix_factors or {}
    peaks = []
    for a in inputs.analytes:
        ng = amounts_ng.get(a.name, 0.0)
        tR = retention_time(a.log_kw, a.S, cond)
        sigma = peak_sigma(a.log_kw, a.S, tR, cond, inputs.system.plate_number,
                           inputs.system.extra_column_sigma_s)
        area = a.response_factor * ng * mf.get(a.name, 1.0) * sensitivity
        peaks.append(PeakSpec(a.name, tR, sigma, area, tau_min))
    return peaks


def simulate_injection(
    inputs: InputSet,
    amounts_ng: Mapping[str, float],
    rng: np.random.Generator,
    cond: LCConditions | None = None,
    matrix_factors: Mapping[str, float] | None = None,
    tau_min: float = 0.0,
    schedule: Schedule | None = None,
) -> Injection:
    peaks = lc_peaks(inputs, amounts_ng, cond, matrix_factors, tau_min)
    schedule = schedule or build_schedule(inputs.transitions, inputs.acquisition, inputs.lc_method.run_time_min)
    signals = acquire(peaks, inputs.transitions, schedule, inputs.acquisition, rng)
    return Injection(peaks, schedule, signals)


# ---------------------------------------------------------------------------
# A run
# ---------------------------------------------------------------------------


class Kind(str, Enum):
    blank = "blank"
    calibrator = "calibrator"
    qc = "qc"
    sample = "sample"


@dataclass(frozen=True)
class SequenceItem:
    kind: Kind
    ref: str  # calibration level for calibrators, sample id otherwise


def default_sequence(inputs: InputSet) -> list[SequenceItem]:
    """Solvent blank, calibrators low to high, then every defined sample in file order."""
    seq = []
    blanks = [s for s in inputs.samples if not s.true_concentration_ug_kg]
    if blanks:
        seq.append(SequenceItem(Kind.blank, blanks[0].id))
    seq += [SequenceItem(Kind.calibrator, lvl.level) for lvl in inputs.calibration_design]
    for s in inputs.samples:
        if s.kind:
            kind = Kind(s.kind)
        else:
            kind = Kind.blank if not s.true_concentration_ug_kg else Kind.qc if s.id.lower().startswith("qc") else Kind.sample
        seq.append(SequenceItem(kind, s.id))
    return seq


@dataclass(frozen=True)
class RunConditions:
    """How the run is simulated, beyond the input files."""

    detector: Literal["simulated", "expected"] = "simulated"  # expected = noise-free areas
    carryover_fraction: float = 0.0       # of the previous injection's on-column amount
    drift_per_injection: float = 0.0      # relative sensitivity change per injection
    mf_scale: Mapping[str, float] = field(default_factory=dict)
    tau_min: float = 0.0
    # Differences between real samples (kind "sample") and the calibration matrix:
    sample_recovery_factor: Mapping[str, float] = field(default_factory=dict)
    sample_matrix_scale: float = 1.0      # stretches (1 - MF) for samples only
    sample_istd_stage: ISTDStage | None = None   # where the IS actually went in, for samples
    interference_ug_kg: Mapping[str, float] = field(default_factory=dict)  # added to the quantifier, samples only
    isomer_fraction_actual: Mapping[str, float] = field(default_factory=dict)  # true share in the standard


@dataclass(frozen=True)
class InjectionRecord:
    index: int
    item: SequenceItem
    on_column_ng: dict[str, float]
    matrix_factor: dict[str, float]
    areas: dict[str, float]                  # quantifier area per analyte (targets and IS)
    qualifier_areas: dict[int, float]        # by transition index
    signals: list[TransitionSignal] | None


@dataclass(frozen=True)
class RunResult:
    injections: list[InjectionRecord]
    fits: dict[str, CalibrationFit]
    reported_ug_kg: dict[str, dict[str, float]]   # sample id -> analyte -> reported
    retention_min: dict[str, float]

    def record(self, ref: str) -> InjectionRecord:
        return next(r for r in self.injections if r.item.ref == ref and r.item.kind is not Kind.calibrator)

    def totals(self, inputs: InputSet, sample_id: str) -> dict[str, float]:
        """Reported results summed into regulated analytes (isomers and components added)."""
        reg = {a.name: a.regulated_as or a.name for a in inputs.analytes}
        out: dict[str, float] = {}
        for a, v in self.reported_ug_kg.get(sample_id, {}).items():
            out[reg[a]] = out.get(reg[a], 0.0) + v
        return out


def run_sequence(
    inputs: InputSet,
    rng: np.random.Generator,
    sequence: Sequence[SequenceItem] | None = None,
    conditions: RunConditions | None = None,
    cond: LCConditions | None = None,
    profile: MatrixProfile | None = None,
) -> RunResult:
    conditions = conditions or RunConditions()
    sequence = list(sequence or default_sequence(inputs))
    cond = cond or LCConditions.from_method(inputs.system, inputs.lc_method)
    profile = profile or profile_from_traces(inputs.infusion_matrix, inputs.infusion_solvent)
    tR = {a.name: retention_time(a.log_kw, a.S, cond) for a in inputs.analytes}
    schedule = build_schedule(inputs.transitions, inputs.acquisition, inputs.lc_method.run_time_min)
    samples = {s.id: s for s in inputs.samples}
    levels = {lvl.level: lvl for lvl in inputs.calibration_design}

    acq = inputs.acquisition
    xt = {}
    if acq.crosstalk and acq.crosstalk_yield > 0:
        xt = {j: [dataclasses.replace(c, factor=c.factor * acq.crosstalk_yield) for c in cs]
              for j, cs in crosstalk_map(inputs.transitions, acq.q1_fwhm_mz, acq.q3_fwhm_mz).items()}

    records: list[InjectionRecord] = []
    prev: dict[str, float] = {}
    for k, item in enumerate(sequence):
        if item.kind is Kind.calibrator:
            ng, in_matrix = _calibrator_amounts(inputs, levels[item.ref], conditions), inputs.calibration.matrix_matched
        else:
            smp = samples[item.ref]
            ng, in_matrix = _sample_amounts(inputs, smp, item.kind, conditions), smp.matrix.lower() != "none"
        if conditions.carryover_fraction:
            ng = {a: ng.get(a, 0.0) + conditions.carryover_fraction * prev.get(a, 0.0) for a in set(ng) | set(prev)}
        extra = conditions.sample_matrix_scale if item.kind is Kind.sample else 1.0
        mf = {a: float(profile.factor(tR[a], conditions.mf_scale.get(a, 1.0) * extra)) if in_matrix else 1.0
              for a in tR}
        sens = 1.0 + conditions.drift_per_injection * k
        peaks = lc_peaks(inputs, ng, cond, mf, conditions.tau_min, sens)
        interf = _interference_areas(inputs, conditions) if item.kind is Kind.sample else {}
        areas, qual, signals = _measure(inputs, peaks, schedule, conditions, rng, interf, xt)
        records.append(InjectionRecord(k, item, ng, mf, areas, qual, signals))
        prev = ng

    fits, reported = _quantitate(inputs, records)
    return RunResult(records, fits, reported, tR)


def _istd_vials(inputs: InputSet, v, recovery: bool = True) -> dict[str, float]:
    """Vial concentration of every IS for prep values v (actual mix content, IS recovery)."""
    p = inputs.prep
    out = {}
    for a in inputs.analytes:
        if a.is_internal_standard and a.name in p.istd_mix_ng_ul:
            mix = p.istd_mix_ng_ul[a.name] * p.istd_mix_factor_actual.get(a.name, 1.0)
            c = istd_vial_conc(v, mix, p.recovery.get(a.name, 1.0) if recovery else 1.0)
            if c is not None:
                out[a.name] = c
    return out


def _sample_amounts(inputs: InputSet, smp: SampleDefinition, kind: Kind = Kind.sample,
                    conditions: RunConditions | None = None) -> dict[str, float]:
    conditions = conditions or RunConditions()
    act = prep_values(inputs.prep, "actual")
    rec = dict(inputs.prep.recovery)
    if kind is Kind.sample:
        rec = {a: rec.get(a, 1.0) * f for a, f in conditions.sample_recovery_factor.items()} | {
            a: v for a, v in rec.items() if a not in conditions.sample_recovery_factor}
    conc = dict(smp.true_concentration_ug_kg)
    if smp.spiked_from_calibration_stock:
        cal = inputs.calibration
        stock = cal.stock_concentration_ng_ml.actual / cal.stock_concentration_ng_ml.nominal
        frac = {a.name: a.isomer_fraction or 1.0 for a in inputs.analytes}
        conc = {a: c * stock * cal.stock_factor_actual.get(a, 1.0)
                * conditions.isomer_fraction_actual.get(a, frac.get(a, 1.0)) / frac.get(a, 1.0)
                for a, c in conc.items()}
    ng = {a: on_column_ng(analyte_vial_conc(c, rec.get(a, 1.0), act), act) for a, c in conc.items()}
    if smp.add_istd:
        v = act
        if kind is Kind.sample and conditions.sample_istd_stage is not None:
            v = dataclasses.replace(act, istd_stage=conditions.sample_istd_stage)
        ng |= {k: on_column_ng(c, act) for k, c in _istd_vials(inputs, v).items()}
    return ng


def _interference_areas(inputs: InputSet, conditions: RunConditions) -> dict[str, float]:
    """Extra quantifier area equivalent to the given ug/kg, through the nominal prep."""
    if not conditions.interference_ug_kg:
        return {}
    nom = prep_values(inputs.prep, "nominal")
    rf = {a.name: a.response_factor for a in inputs.analytes}
    return {a: rf[a] * on_column_ng(analyte_vial_conc(c, 1.0, nom), nom)
            for a, c in conditions.interference_ug_kg.items()}


def _calibrator_amounts(inputs: InputSet, level, conditions: RunConditions | None = None) -> dict[str, float]:
    """Calibrator vial: nominal level x stock error x isomer share (x recovery if extracted)."""
    cal = inputs.calibration
    inj_ul = inputs.prep.injection_volume_ul.nominal
    stock = cal.stock_concentration_ng_ml.actual / cal.stock_concentration_ng_ml.nominal
    procedural = cal.preparation == "procedural"
    frac = {a.name: a.isomer_fraction or 1.0 for a in inputs.analytes}
    if conditions is not None:
        frac |= dict(conditions.isomer_fraction_actual)
    rec = inputs.prep.recovery
    ng = {a: c * stock * cal.stock_factor_actual.get(a, 1.0) * frac.get(a, 1.0)
          * (rec.get(a, 1.0) if procedural else 1.0) * inj_ul / 1000.0
          for a, c in level.concentrations_ng_ml.items()}
    if cal.use_internal_standard and cal.istd_spike_ul is not None:
        v = _calibrator_prep(inputs, cal.istd_spike_ul.actual)
        ng |= {k: c * inj_ul / 1000.0 for k, c in _istd_vials(inputs, v, recovery=procedural).items()}
    return ng


def _calibrator_prep(inputs: InputSet, spike_ul: float):
    """Prep values for a calibrator's IS: the sample procedure if extracted, else straight into the vial."""
    v = dataclasses.replace(prep_values(inputs.prep, "nominal"), istd_spike_ul=spike_ul)
    if inputs.calibration.preparation != "procedural":
        v = dataclasses.replace(v, istd_stage=ISTDStage.in_vial)
    return v


def _measure(inputs, peaks, schedule, conditions, rng, interference=None,
             xt: Mapping[int, list[Crosstalk]] | None = None):
    interference, xt = interference or {}, xt or {}
    by_name = {p.analyte: p for p in peaks}
    if conditions.detector == "expected":
        # Noise-free areas, reduced by the share of each peak that falls inside its acquisition window.
        areas, qual = {}, {}
        for j, t in enumerate(inputs.transitions):
            p = by_name.get(t.analyte)
            if p is None:
                continue
            lo = t.retention_time_min - t.window_min / 2
            hi = t.retention_time_min + t.window_min / 2
            inside = _normal_cdf((hi - p.center_min) / p.sigma_min) - _normal_cdf((lo - p.center_min) / p.sigma_min)
            inside = inside if inside > 1e-9 else 0.0
            # Crosstalk: the part of another compound's peak inside both this window and this peak's integration.
            i_lo, i_hi = integration_range(p)
            a, b = max(lo, i_lo), min(hi, i_hi)
            leak = sum(q.quant_area * c.factor * max(_normal_cdf((b - q.center_min) / q.sigma_min)
                                                     - _normal_cdf((a - q.center_min) / q.sigma_min), 0.0)
                       for c in xt.get(j, ()) if (q := by_name.get(c.source)) is not None and b > a)
            if t.role is TransitionRole.quantifier:
                areas[t.analyte] = p.quant_area * inside + interference.get(t.analyte, 0.0) + leak
            else:
                qual[j] = p.quant_area * (t.relative_response or 1.0) * inside + leak
        return areas, qual, None
    quant_idx = {t.analyte: j for j, t in enumerate(inputs.transitions) if t.role is TransitionRole.quantifier}
    extra = {quant_idx[a]: x for a, x in interference.items() if a in quant_idx}
    leaks = {j: [(by_name[c.source], by_name[c.source].quant_area * c.factor) for c in cs if c.source in by_name]
             for j, cs in xt.items()}
    signals = acquire(peaks, inputs.transitions, schedule, inputs.acquisition, rng, extra_area=extra, crosstalk=leaks)
    areas = {s.analyte: s.area for s in signals if s.role is TransitionRole.quantifier}
    qual = {s.transition_index: s.area for s in signals if s.role is TransitionRole.qualifier}
    return areas, qual, signals


def _normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _quantitate(inputs: InputSet, records: Sequence[InjectionRecord]):
    cal = inputs.calibration
    use_is = cal.use_internal_standard
    targets = [a for a in inputs.analytes if not a.is_internal_standard]
    levels = {lvl.level: lvl for lvl in inputs.calibration_design}
    cal_records = [r for r in records if r.item.kind is Kind.calibrator]
    nominal = prep_values(inputs.prep, "nominal")
    mix = inputs.prep.istd_mix_ng_ul
    frac = {a.name: a.isomer_fraction or 1.0 for a in inputs.analytes}
    is_vial_sample, is_vial_cal = {}, {}
    if use_is and cal.istd_spike_ul is not None:
        is_vial_sample = {k: istd_vial_conc(nominal, m) for k, m in mix.items()}
        v = _calibrator_prep(inputs, cal.istd_spike_ul.nominal)
        is_vial_cal = {k: istd_vial_conc(v, m) for k, m in mix.items()}

    fits: dict[str, CalibrationFit] = {}
    reported: dict[str, dict[str, float]] = {}
    for a in targets:
        istd = a.internal_standard if use_is and a.internal_standard else None
        xs, ys = [], []
        for r in cal_records:
            c = levels[r.item.ref].concentrations_ng_ml.get(a.name)
            if c is None:
                continue
            xs.append(c / is_vial_cal[istd] if istd else c)
            ys.append(response(r.areas, a.name, istd))
        if len(xs) < 2:
            continue
        fit = fit_calibration(xs, ys, cal.model, cal.weighting, cal.origin)
        fits[a.name] = fit
        for r in records:
            if r.item.kind is Kind.calibrator:
                continue
            y = response(r.areas, a.name, istd)
            x = fit.inverse(y) if not math.isnan(y) else math.nan
            vial = (x * is_vial_sample[istd] if istd else x) * frac[a.name]
            reported.setdefault(r.item.ref, {})[a.name] = back_calculate(vial, nominal)
    return fits, reported
