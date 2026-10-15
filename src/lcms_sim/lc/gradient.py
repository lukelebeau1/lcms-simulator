"""Gradient program: organic fraction phi (0-1) versus time at the mixer and at the column inlet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lcms_sim.io.schemas import GradientProgram, LCMethod, SystemConstants


@dataclass(frozen=True)
class LCConditions:
    """Everything retention needs: program, void time, and dwell time, all at one flow."""

    times: tuple[float, ...]   # min, non-decreasing; equal times are steps
    phis: tuple[float, ...]    # volume fraction, 0-1
    t0: float                  # min
    tD: float                  # min, dwell volume / flow
    run_time: float | None = None

    @classmethod
    def from_program(cls, program: GradientProgram, t0: float, tD: float, run_time: float | None = None):
        return cls(
            times=tuple(p.time_min for p in program.points),
            phis=tuple(p.percent_b / 100.0 for p in program.points),
            t0=t0, tD=tD, run_time=run_time,
        )

    @classmethod
    def from_method(cls, system: SystemConstants, method: LCMethod):
        return cls.from_program(
            method.gradient,
            t0=system.t0_at(method.flow_ml_min),
            tD=system.dwell_volume_ml / method.flow_ml_min,
            run_time=method.run_time_min,
        )

    def phi_programmed(self, t):
        return phi_at(self.times, self.phis, t)

    def phi_inlet(self, t):
        """Composition reaching the column inlet at time t (programmed, delayed by the dwell time)."""
        return phi_at(self.times, self.phis, np.asarray(t, dtype=float) - self.tD)

    def inlet_segments(self) -> list[tuple[float, float, float]]:
        """Piecewise-linear phi at the inlet as (start, phi_start, slope) from t = 0 onward.

        The last segment runs to infinity at constant composition.
        """
        xs = [0.0] + [t + self.tD for t in self.times if t + self.tD > 0]
        segs = []
        for a, b in zip(xs, xs[1:]):
            if b > a:
                pa = float(self.phi_inlet(a))
                # Left-limit at b, so a step at b does not leak into this segment.
                pb = _phi_left(self.times, self.phis, b - self.tD)
                segs.append((a, pa, (pb - pa) / (b - a)))
        segs.append((xs[-1], float(self.phi_inlet(xs[-1])), 0.0))
        return segs


def phi_at(times, phis, t):
    """Linear interpolation. At a step (two equal times) the later value applies from that instant."""
    times = np.asarray(times, dtype=float)
    phis = np.asarray(phis, dtype=float)
    t = np.asarray(t, dtype=float)
    i = np.searchsorted(times, t, side="right")  # first index with time > t
    i = np.clip(i, 1, len(times) - 1) if len(times) > 1 else np.zeros_like(i)
    if len(times) == 1:
        return np.full_like(t, phis[0], dtype=float)
    t_a, t_b = times[i - 1], times[i]
    p_a, p_b = phis[i - 1], phis[i]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(t_b > t_a, (t - t_a) / (t_b - t_a), 1.0)
    out = p_a + np.clip(frac, 0.0, 1.0) * (p_b - p_a)
    out = np.where(t < times[0], phis[0], out)
    out = np.where(t >= times[-1], phis[-1], out)
    return out if out.ndim else float(out)


def _phi_left(times, phis, t: float) -> float:
    """Left limit of phi at t: the value approached from earlier times."""
    times = list(times)
    phis = list(phis)
    if t <= times[0]:
        return phis[0]
    for (ta, pa), (tb, pb) in zip(zip(times, phis), zip(times[1:], phis[1:])):
        if ta < t <= tb:
            return pa + (t - ta) / (tb - ta) * (pb - pa)
    return phis[-1]
