"""The fault library: named, sized changes to one stage's actual value.

A fault never changes the pipeline. It returns a new Scenario whose inputs or run conditions
differ from the method in one place, and the ordinary simulation shows the consequence.
Each fault has a magnitude with a documented meaning and range, so the matcher can sweep it.

Each fault also carries its scope, the troubleshooter's first split: "uniform" when every
compound's result moves by the same share (a volume, a stock, a drift), "compound" when the
bias differs between compounds, with `varies_with` naming what it follows (recovery chemistry,
retention time, pKa, m/z, level). Classified by the effect on the results, not the cause.
The feature vector reads the same split as sample_bias_spread and bias_rt_corr.
"""

from __future__ import annotations

import dataclasses
import math
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from lcms_sim.io.loaders import InputSet
from lcms_sim.io.schemas import AcidBase, ISTDStage, NominalActual, Transition, TransitionRole, Weighting
from lcms_sim.lc.gradient import LCConditions
from lcms_sim.pipeline import RunConditions, RunResult, SequenceItem, run_sequence

# Assumed retention drop of an ionised form when no measured value exists (log10 k units).
ASSUMED_IONISED_SHIFT = -1.5


@dataclass(frozen=True)
class Scenario:
    inputs: InputSet
    run: RunConditions = field(default_factory=RunConditions)
    lc: LCConditions | None = None
    sequence: tuple[SequenceItem, ...] | None = None

    @classmethod
    def baseline(cls, inputs: InputSet, **run_kw) -> Scenario:
        return cls(inputs, RunConditions(**run_kw))

    def lc_conditions(self) -> LCConditions:
        return self.lc or LCConditions.from_method(self.inputs.system, self.inputs.lc_method)

    def simulate(self, rng: np.random.Generator) -> RunResult:
        return run_sequence(self.inputs, rng, self.sequence, self.run, self.lc_conditions())


def _na(x: NominalActual, factor: float) -> NominalActual:
    return NominalActual(nominal=x.nominal, actual=x.nominal * factor)


def _with_prep(s: Scenario, **upd) -> Scenario:
    return dataclasses.replace(s, inputs=dataclasses.replace(s.inputs, prep=s.inputs.prep.model_copy(update=upd)))


def _with_cal(s: Scenario, **upd) -> Scenario:
    cal = s.inputs.calibration.model_copy(update=upd)
    return dataclasses.replace(s, inputs=dataclasses.replace(s.inputs, calibration=cal))


def _with_run(s: Scenario, **upd) -> Scenario:
    return dataclasses.replace(s, run=dataclasses.replace(s.run, **upd))


def _with_acq(s: Scenario, **upd) -> Scenario:
    acq = s.inputs.acquisition.model_copy(update=upd)
    return dataclasses.replace(s, inputs=dataclasses.replace(s.inputs, acquisition=acq))


def _targets(s: Scenario) -> list[str]:
    return [a.name for a in s.inputs.analytes if not a.is_internal_standard]


def _spread(name: str) -> float:
    """A fixed pseudo-random number in [0, 1) per analyte, so analyte-specific faults are reproducible."""
    return (zlib.crc32(name.encode()) % 10_000) / 10_000


@dataclass(frozen=True)
class Fault:
    key: str
    name: str
    stage: str
    signature: str
    magnitude_label: str
    low: float
    high: float
    default: float
    analytes: tuple[str, ...] = ()   # for faults that hit a subset
    scope: str = "uniform"           # uniform: every compound by the same share; compound: differs between compounds
    varies_with: str = ""            # compound scope: what the bias follows
    sources: tuple[str, ...] = ()    # keys into data/reference/references.yaml; empty = the model's own arithmetic

    def __post_init__(self) -> None:
        if self.scope not in ("uniform", "compound") or (self.scope == "compound") != bool(self.varies_with):
            raise ValueError(f"{self.key}: scope is uniform or compound, and only a compound fault names what it varies with")

    def apply(self, s: Scenario, magnitude: float | None = None) -> Scenario:
        raise NotImplementedError

    def sweep(self, n: int = 7) -> np.ndarray:
        if self.low > 0 and self.high / self.low > 5:
            return np.geomspace(self.low, self.high, n)
        return np.linspace(self.low, self.high, n)

    def with_analytes(self, analytes: Sequence[str]) -> Fault:
        return dataclasses.replace(self, analytes=tuple(analytes))


# --- Calibration -------------------------------------------------------------


class StockStrong(Fault):
    def apply(self, s, m=None):
        m = self.default if m is None else m
        return _with_cal(s, stock_concentration_ng_ml=_na(s.inputs.calibration.stock_concentration_ng_ml, m))


class StockDegraded(Fault):
    def apply(self, s, m=None):
        m = self.default if m is None else m
        hit = self.analytes or tuple(_targets(s)[:3])
        return _with_cal(s, stock_factor_actual={a: m for a in hit})


class WrongWeighting(Fault):
    def apply(self, s, m=None):
        return _with_cal(s, weighting=Weighting.none)


class IsomerShareWrong(Fault):
    def apply(self, s, m=None):
        m = self.default if m is None else m
        frac = {a.name: a.isomer_fraction for a in s.inputs.analytes if a.isomer_fraction}
        hit = self.analytes or tuple(frac)[:1]
        return _with_run(s, isomer_fraction_actual={a: frac[a] * m for a in hit if a in frac})


# --- Internal standard -------------------------------------------------------


class IstdOverspiked(Fault):
    def apply(self, s, m=None):
        m = self.default if m is None else m
        return _with_prep(s, istd_spike_ul=_na(s.inputs.prep.istd_spike_ul, m))


class IstdLate(Fault):
    def apply(self, s, m=None):
        return _with_run(s, sample_istd_stage=ISTDStage.after_extraction)


class IstdWindowMissesPeak(Fault):
    """The method's RT for an IS is wrong by `m` minutes, so its window misses the real peak."""

    def apply(self, s, m=None):
        m = self.default if m is None else m
        istds = self.analytes or tuple(a.name for a in s.inputs.analytes if a.is_internal_standard)[:1]
        trs = [t.model_copy(update={"retention_time_min": t.retention_time_min + m}) if t.analyte in istds else t
               for t in s.inputs.transitions]
        return dataclasses.replace(s, inputs=dataclasses.replace(s.inputs, transitions=trs))


# --- Prep --------------------------------------------------------------------


class LowRecovery(Fault):
    """Samples extract worse than the calibration matrix; the loss differs between analytes."""

    def apply(self, s, m=None):
        m = self.default if m is None else m
        hit = self.analytes or tuple(_targets(s))
        return _with_run(s, sample_recovery_factor={a: m ** (0.5 + _spread(a)) for a in hit})


class VolumeError(Fault):
    def apply(self, s, m=None):
        m = self.default if m is None else m
        return _with_prep(s, extraction_volume_ml=_na(s.inputs.prep.extraction_volume_ml, m))


class LowInjectionVolume(Fault):
    def apply(self, s, m=None):
        m = self.default if m is None else m
        return _with_prep(s, injection_volume_ul=_na(s.inputs.prep.injection_volume_ul, m))


# --- Ionisation and detection --------------------------------------------------


class MatrixSuppression(Fault):
    def apply(self, s, m=None):
        m = self.default if m is None else m
        return _with_run(s, sample_matrix_scale=m)


class Interference(Fault):
    def apply(self, s, m=None):
        m = self.default if m is None else m
        hit = self.analytes or tuple(_targets(s)[:1])
        return _with_run(s, interference_ug_kg={a: m for a in hit})


class Saturation(Fault):
    def apply(self, s, m=None):
        m = self.default if m is None else m
        return _with_acq(s, detector_dead_time_s=m)


class Drift(Fault):
    def apply(self, s, m=None):
        m = self.default if m is None else m
        return _with_run(s, drift_per_injection=m)


class Carryover(Fault):
    def apply(self, s, m=None):
        m = self.default if m is None else m
        return _with_run(s, carryover_fraction=m)


class TooManyTransitions(Fault):
    """`m` extra transitions from other methods, spread evenly over the run with 2 min windows."""

    def apply(self, s, m=None):
        m = int(round(self.default if m is None else m))
        run = s.inputs.lc_method.run_time_min
        extra = [Transition(analyte=f"_extra{i}", precursor_mz=500, product_mz=100, role=TransitionRole.quantifier,
                            retention_time_min=run * (i + 0.5) / m, window_min=2.0) for i in range(m)]
        return dataclasses.replace(s, inputs=dataclasses.replace(s.inputs, transitions=[*s.inputs.transitions, *extra]))


# --- LC ----------------------------------------------------------------------


class OrganicError(Fault):
    """Mobile phase B is m (volume fraction) stronger than intended at every point of the gradient."""

    def apply(self, s, m=None):
        m = self.default if m is None else m
        c = s.lc_conditions()
        lc = dataclasses.replace(c, phis=tuple(min(max(p + m, 0.0), 1.0) for p in c.phis))
        return dataclasses.replace(s, lc=lc)


class PhError(Fault):
    """Mobile phase pH off by m units. Ionisable analytes shift; others do not.

    With equal S for both forms, log k changes by a constant at every %B:
    shift = log10[(1 + 10^d r) / (1 + r)], r = 10^(pH - pKa) (acid) or 10^(pKa - pH) (base),
    d = log kw(ionised) - log kw(neutral), assumed -1.5 when not measured.
    """

    def apply(self, s, m=None):
        m = self.default if m is None else m
        ph0 = s.inputs.lc_method.ph or 3.0
        out = []
        for a in s.inputs.analytes:
            if a.acid_base is AcidBase.neutral or a.pka is None or a.pka > 13:
                out.append(a)
                continue
            d = (a.log_kw_ionised - a.log_kw) if a.log_kw_ionised is not None else ASSUMED_IONISED_SHIFT

            def logk_shift(ph):
                r = 10 ** ((ph - a.pka) if a.acid_base is AcidBase.acid else (a.pka - ph))
                return math.log10((1 + 10 ** d * r) / (1 + r))

            out.append(a.model_copy(update={"log_kw": a.log_kw + logk_shift(ph0 + m) - logk_shift(ph0)}))
        return dataclasses.replace(s, inputs=dataclasses.replace(s.inputs, analytes=out))


class DwellVolumeChange(Fault):
    def apply(self, s, m=None):
        m = self.default if m is None else m
        c = s.lc_conditions()
        return dataclasses.replace(s, lc=dataclasses.replace(c, tD=c.tD * m))


LIBRARY: dict[str, Fault] = {f.key: f for f in [
    StockStrong("stock_strong", "Calibration stock stronger than labelled", "Calibration",
                "Uniform low bias on samples. Curve, IS areas and QCs from the same stock look normal; "
                "an independent QC or PT sample fails.", "actual / labelled stock", 1.05, 4.0, 1.5, scope="uniform",
                sources=("sante_11312_2021",)),
    StockDegraded("stock_degraded", "Calibration stock degraded, some analytes", "Calibration",
                  "High bias on the affected analytes only.", "remaining fraction", 0.2, 0.95, 0.6,
                  scope="compound", varies_with="stability of each compound in the stock",
                  sources=("sante_11312_2021", "restek_2021")),
    IstdOverspiked("istd_overspiked", "Internal standard over-spiked in samples", "Internal standard",
                   "Uniform low bias on IS-corrected analytes; IS areas higher in samples than calibrators.",
                   "actual / nominal spike", 1.05, 3.0, 1.5, scope="uniform", sources=("stokvis_2005",)),
    IstdLate("istd_late", "Internal standard added after extraction", "Internal standard",
             "Recovery no longer corrected: bias varies by analyte and tracks recovery.", "(on/off)", 1, 1, 1,
             scope="compound", varies_with="each compound's extraction recovery",
             sources=("matuszewski_2003", "stokvis_2005")),
    IstdWindowMissesPeak("istd_window_miss", "IS acquisition window misses its peak", "Acquisition",
                         "IS area near zero; analytes paired to it report nothing or nonsense.",
                         "method RT error (min)", 0.5, 3.0, 2.0,
                         scope="compound", varies_with="which compounds are paired to that internal standard",
                         sources=("agilent_5990_3595",)),
    LowRecovery("low_recovery", "Low extraction recovery in samples", "Extraction",
                "Analyte-specific low bias in samples; QCs in the calibration matrix pass.",
                "typical recovery", 0.3, 0.95, 0.6,
                scope="compound", varies_with="each compound's extraction chemistry",
                sources=("matuszewski_2003", "sante_11312_2021")),
    MatrixSuppression("matrix_suppression", "Matrix suppression stronger in samples", "Ionisation",
                      "Raw areas drop in samples. Bias depends on retention time and is worst where the IS "
                      "elutes elsewhere.", "suppression scale vs calibration matrix", 1.2, 3.0, 2.0,
                      scope="compound", varies_with="retention time: what co-elutes, and whether the IS co-elutes",
                      sources=("king_2000", "bonfiglio_1999", "matuszewski_2003", "stokvis_2005")),
    VolumeError("volume_error", "Extraction volume error", "Extraction",
                "Uniform bias when the step is before the IS is added.", "actual / nominal volume",
                0.5, 3.0, 1.5, scope="uniform"),
    LowInjectionVolume("low_injection", "Low injection volume", "Injection",
                       "All raw areas drop together; ratio-based results unchanged.", "actual / nominal",
                       0.2, 0.95, 0.5, scope="uniform"),
    PhError("ph_error", "Mobile phase pH prepared wrong", "LC",
            "Retention shifts, largest for ionisable analytes; peaks clipped by windows read low.",
            "pH error (units)", -1.0, 1.5, 1.0, scope="compound", varies_with="pKa: only ionisable compounds move",
            sources=("horvath_1977", "snyder_kirkland_dolan_2010")),
    OrganicError("organic_error", "Mobile phase %B prepared wrong", "LC",
                 "Retention shifts for every analyte; peaks near window edges are clipped.",
                 "%B error (fraction)", -0.05, 0.05, 0.03,
                 scope="compound", varies_with="each compound's S and how near its window edge it elutes",
                 sources=("snyder_dolan_2007",)),
    DwellVolumeChange("dwell_volume", "Dwell volume or proportioning change", "LC",
                      "Similar retention shift across analytes, largest for early eluters.",
                      "actual / nominal dwell volume", 0.3, 3.0, 2.0,
                      scope="compound", varies_with="retention time: early eluters and narrow windows clip first",
                      sources=("dolan_2006_dwell", "snyder_dolan_2007")),
    TooManyTransitions("too_many_transitions", "Too many concurrent transitions", "Acquisition",
                       "Poor precision and scattered ion ratios at low level; mean result unbiased.",
                       "extra transitions", 20, 300, 150, scope="uniform",
                       sources=("agilent_5990_3595", "zeng_2023", "waters_720001168")),
    WrongWeighting("wrong_weighting", "Wrong calibration weighting", "Calibration",
                   "Low-level accuracy fails while high levels pass.", "(on/off)", 1, 1, 1, scope="uniform",
                   sources=("almeida_2002", "gu_2014")),
    Saturation("saturation", "Detector saturation", "Detection",
               "Curve bends at the top; high samples read low.", "dead time (s)", 1e-8, 1e-6, 2e-7,
               scope="compound", varies_with="ion rate: the strongest responders and the highest levels first",
               sources=("knoll_2010",)),
    Interference("interference", "Interference on one transition", "Detection",
                 "Ion ratio off in samples only; the affected analyte reads high at low level.",
                 "interference (ug/kg equivalent)", 5.0, 200.0, 50.0,
                 scope="compound", varies_with="one transition: the compound whose quantifier the interferent shares",
                 sources=("sante_11312_2021",)),
    Carryover("carryover", "Carryover", "Injection",
              "Blanks positive after high injections; low samples biased high.", "fraction carried", 1e-4, 0.02, 0.005,
              scope="compound", varies_with="the previous injection's level of each compound",
              sources=("dolan_2006_carryover",)),
    Drift("drift", "Sensitivity drift over the sequence", "Detection",
          "Raw areas trend with injection order; ratio results hold until signal nears the LOQ.",
          "relative change per injection", -0.05, 0.05, -0.03, scope="uniform"),
    IsomerShareWrong("isomer_share", "Isomer share in the standard differs from the SOP", "Calibration",
                     "One component of a summed analyte biased; the total shifts by that component's share.",
                     "actual / stated share", 0.5, 1.5, 0.8,
                     scope="compound", varies_with="the components of the summed analyte", sources=("mn_ocm_cta_v2",)),
]}
