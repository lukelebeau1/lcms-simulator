"""Dynamic MRM scheduling: which transitions run in each cycle, their dwell, and when they sample.

Agilent dMRM keeps the cycle time fixed and shares it among the transitions whose retention
windows are open:

    t_dwell = (T_cycle - polarity overhead) / n(t) - t_delay

Within a cycle the active transitions run one after another, so each samples at its own
offset. A peak that drifts out of its window is simply not sampled there, which is how window
clipping loses area.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lcms_sim.io.schemas import AcquisitionMethod, Polarity, Transition

MS_PER_MIN = 60_000.0


@dataclass(frozen=True)
class Schedule:
    cycle_starts: np.ndarray   # (C,) min
    active: np.ndarray         # (C, J) bool
    n_active: np.ndarray       # (C,) int
    dwell_ms: np.ndarray       # (C,) dwell for every active transition in that cycle
    sample_times: np.ndarray   # (C, J) min, nan where inactive
    cycle_time_ms: float

    @property
    def cycle_min(self) -> float:
        return self.cycle_time_ms / MS_PER_MIN

    def concurrency(self, t) -> np.ndarray:
        """n(t): transitions active in the cycle containing t."""
        i = np.clip(np.searchsorted(self.cycle_starts, t, side="right") - 1, 0, len(self.cycle_starts) - 1)
        return self.n_active[i]

    def dwell_at(self, t) -> np.ndarray:
        i = np.clip(np.searchsorted(self.cycle_starts, t, side="right") - 1, 0, len(self.cycle_starts) - 1)
        return self.dwell_ms[i]


def dwell_time_ms(cycle_time_ms: float, n: int, delay_ms: float, overhead_ms: float = 0.0) -> float:
    """The scheduling formula, for one cycle with n active transitions."""
    if n <= 0:
        return 0.0
    return (cycle_time_ms - overhead_ms) / n - delay_ms


def build_schedule(transitions: Sequence[Transition], acq: AcquisitionMethod, run_time_min: float) -> Schedule:
    tc = acq.cycle_time_ms
    starts = np.arange(0.0, run_time_min, tc / MS_PER_MIN)
    rt = np.array([t.retention_time_min for t in transitions])
    half = np.array([t.window_min for t in transitions]) / 2.0
    active = (starts[:, None] >= rt - half) & (starts[:, None] < rt + half)
    n = active.sum(axis=1)

    pos = np.array([t.polarity is Polarity.positive for t in transitions])
    both = active[:, pos].any(axis=1) & active[:, ~pos].any(axis=1)
    # Positive and negative transitions are grouped: two switches per cycle when both run.
    overhead = np.where(both, 2.0 * acq.polarity_switch_ms, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dwell = np.where(n > 0, (tc - overhead) / np.maximum(n, 1) - acq.inter_scan_delay_ms, 0.0)
    if np.any(dwell[n > 0] <= 0):
        raise ValueError("cycle time too short for the number of concurrent transitions")

    # Order inside a cycle: positive group first, then negative, then table order.
    order_key = np.where(pos, 0, len(transitions)) + np.arange(len(transitions))
    perm = np.argsort(order_key)
    act_sorted = active[:, perm]
    slot_sorted = np.cumsum(act_sorted, axis=1) - 1
    slot = np.empty_like(slot_sorted)
    slot[:, perm] = slot_sorted
    slot_ms = np.where(n > 0, (tc - overhead) / np.maximum(n, 1), 0.0)
    t_sample = starts[:, None] + (overhead[:, None] / 2 + (slot + 0.5) * slot_ms[:, None]) / MS_PER_MIN
    t_sample = np.where(active, t_sample, np.nan)
    return Schedule(starts, active, n, dwell, t_sample, tc)


@dataclass(frozen=True)
class Slot:
    """One transition's turn inside a cycle: dwell first, then the inter-scan delay."""

    transition_index: int
    start_ms: float   # from the start of the cycle
    dwell_ms: float
    delay_ms: float


@dataclass(frozen=True)
class CycleLayout:
    cycle_start_min: float
    cycle_time_ms: float
    slots: list[Slot]                       # in measurement order
    switches_ms: list[tuple[float, float]]  # polarity-switch intervals, (start, end) from cycle start


def cycle_layout(schedule: Schedule, acq: AcquisitionMethod, t: float) -> CycleLayout:
    """What the quadrupoles do during the cycle that contains time t (min).

    Read back from the schedule, so it is the same timing the signal model samples at:
    each slot is centred on its transition's sample time. When both polarities run, the
    schedule places half the switching overhead before the first slot and half after the last.
    """
    k = int(np.clip(np.searchsorted(schedule.cycle_starts, t, side="right") - 1, 0, len(schedule.cycle_starts) - 1))
    tc = schedule.cycle_time_ms
    js = np.flatnonzero(schedule.active[k])
    if js.size == 0:
        return CycleLayout(float(schedule.cycle_starts[k]), tc, [], [])
    dwell = float(schedule.dwell_ms[k])
    slot_ms = dwell + acq.inter_scan_delay_ms
    overhead = tc - slot_ms * js.size
    mid = (schedule.sample_times[k, js] - schedule.cycle_starts[k]) * MS_PER_MIN
    slots = sorted((Slot(int(j), float(m - slot_ms / 2), dwell, acq.inter_scan_delay_ms) for j, m in zip(js, mid)),
                   key=lambda s: s.start_ms)
    switches = [(0.0, overhead / 2), (tc - overhead / 2, tc)] if overhead > 1e-9 else []
    return CycleLayout(float(schedule.cycle_starts[k]), tc, slots, switches)


def points_across_peak(sigma_min: float, cycle_time_ms: float) -> float:
    """4 sigma / T_cycle: the plan's measure of how well a peak is sampled."""
    return 4.0 * sigma_min / (cycle_time_ms / MS_PER_MIN)


@dataclass(frozen=True)
class ScheduleReport:
    max_concurrent: int
    min_dwell_ms: float
    low_dwell_cycles: int


def report(schedule: Schedule, acq: AcquisitionMethod) -> ScheduleReport:
    busy = schedule.n_active > 0
    d = schedule.dwell_ms[busy]
    return ScheduleReport(
        max_concurrent=int(schedule.n_active.max(initial=0)),
        min_dwell_ms=float(d.min()) if d.size else float("nan"),
        low_dwell_cycles=int((d < acq.min_dwell_ms).sum()),
    )
