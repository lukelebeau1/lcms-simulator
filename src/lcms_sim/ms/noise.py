"""Sampling a chromatographic peak through MRM: counting statistics, noise, saturation, integration.

For each sample of transition j in a cycle (time t_i, dwell d_i):

    expected abundance  a(t)   = A_j x shape(t) + background        (area units / min)
    ion flux            phi(t) = a(t) / gain                         (ions / min)
    counts              c_i ~ Poisson(phi(t_i) x d_i x m_inj x m_i)  m = proportional noise factors
    reported abundance  a_i    = c_i x gain / d_i

Area is the trapezoid of (a_i - baseline) over the sampled points in the integration range.
Because reported abundance is normalised by dwell, the expected area does not depend on dwell;
only its precision does: RSD ~ 1 / sqrt(counted ions), and counted ions scale with d / T_cycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lcms_sim.io.schemas import AcquisitionMethod, Transition, TransitionRole
from lcms_sim.lc.peaks import emg
from lcms_sim.ms.acquisition import MS_PER_MIN, Schedule


@dataclass(frozen=True)
class PeakSpec:
    """One analyte's peak as the LC delivers it."""

    analyte: str
    center_min: float
    sigma_min: float
    quant_area: float        # expected quantifier area (area units)
    tau_min: float = 0.0     # EMG tailing


@dataclass(frozen=True)
class TransitionSignal:
    analyte: str
    transition_index: int
    role: TransitionRole
    times: np.ndarray        # min
    abundance: np.ndarray    # area units / min, as reported
    dwell_ms: np.ndarray
    counts: np.ndarray       # ions counted per point (signal + background)
    baseline: float
    area: float
    expected_area: float
    integration_range: tuple[float, float]
    points_in_peak: int      # sampled points within +-2 sigma of the centre
    clipped: bool            # integration range extends outside the acquired window


def integration_range(p: PeakSpec, n_sigma: float = 4.0) -> tuple[float, float]:
    return p.center_min - n_sigma * p.sigma_min, p.center_min + n_sigma * p.sigma_min + 6.0 * p.tau_min


def saturate(rate_per_s, dead_time_s: float | None):
    """Non-paralysable dead time: observed = true / (1 + true x dead time)."""
    if not dead_time_s:
        return rate_per_s
    return rate_per_s / (1.0 + rate_per_s * dead_time_s)


def acquire(
    peaks: Sequence[PeakSpec],
    transitions: Sequence[Transition],
    schedule: Schedule,
    acq: AcquisitionMethod,
    rng: np.random.Generator,
    transmission: float = 1.0,
    extra_area: dict[int, float] | None = None,
    crosstalk: dict[int, list[tuple[PeakSpec, float]]] | None = None,
) -> list[TransitionSignal]:
    """Simulate one injection. Returns a signal for every transition whose analyte has a peak.

    `extra_area` adds an interfering peak at the same retention to a transition index
    (for the interference fault). `crosstalk` adds other compounds' peaks, at their own retention
    and shape, to the transitions whose Q1 and Q3 windows pass them (ms.crosstalk).
    `transmission` is the hook for the Q1 simulator (default 1).
    """
    by_name = {p.analyte: p for p in peaks}
    m_inj = _prop_factor(rng, acq.injection_rsd)
    gain = acq.gain_area_per_ion
    out: list[TransitionSignal] = []
    for j, tr in enumerate(transitions):
        p = by_name.get(tr.analyte)
        if p is None:
            continue
        cyc = np.flatnonzero(schedule.active[:, j])
        t = schedule.sample_times[cyc, j]
        d_ms = schedule.dwell_ms[cyc]
        d_min = d_ms / MS_PER_MIN
        rel = tr.relative_response if tr.relative_response is not None else 1.0
        area_j = p.quant_area * rel * transmission + (extra_area or {}).get(j, 0.0)
        signal = emg(t, p.center_min, p.sigma_min, p.tau_min, area_j)
        for q, area_q in (crosstalk or {}).get(j, ()):   # another compound's ions through both filters
            signal = signal + emg(t, q.center_min, q.sigma_min, q.tau_min, area_q * transmission)
        a_true = signal + acq.background_rate
        flux = a_true / gain                                   # ions / min
        m_pt = _prop_factor(rng, acq.point_rsd, size=t.size)
        rate_s = saturate(flux * m_inj * m_pt / 60.0, acq.detector_dead_time_s)
        counts = rng.poisson(rate_s * 60.0 * d_min)
        abundance = counts * gain / np.where(d_min > 0, d_min, np.inf)

        lo, hi = integration_range(p)
        inside = (t >= lo) & (t <= hi)
        outside = ~inside
        baseline = float(abundance[outside].mean()) if outside.sum() >= 3 else acq.background_rate
        area = _trapezoid(t[inside], abundance[inside] - baseline)
        acquired_lo = tr.retention_time_min - tr.window_min / 2
        acquired_hi = tr.retention_time_min + tr.window_min / 2
        out.append(TransitionSignal(
            analyte=tr.analyte, transition_index=j, role=tr.role, times=t, abundance=abundance,
            dwell_ms=d_ms, counts=counts, baseline=baseline, area=area, expected_area=area_j,
            integration_range=(lo, hi),
            points_in_peak=int(((t >= p.center_min - 2 * p.sigma_min) & (t <= p.center_min + 2 * p.sigma_min)).sum()),
            clipped=bool(lo < acquired_lo or hi > acquired_hi),
        ))
    return out


def expected_counted_ions(area: float, gain: float, dwell_ms: float, cycle_time_ms: float) -> float:
    """Ions counted over a peak of `area` at constant dwell: A / gain x dwell / cycle."""
    return area / gain * dwell_ms / cycle_time_ms


def _prop_factor(rng: np.random.Generator, rsd: float, size=None):
    if rsd <= 0:
        return 1.0 if size is None else np.ones(size)
    return np.maximum(rng.normal(1.0, rsd, size=size), 0.0)


def _trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))
