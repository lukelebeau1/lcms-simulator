"""Retention under any gradient, by the linear solvent strength (LSS) model.

    log10 k(phi) = log10 kw - S phi
    integral_0^(tR - t0) dt / (t0 k(phi_inlet(t))) = 1

Two solvers:
  * `retention_time`: the reference. The inlet composition is piecewise linear, so on each
    segment k is a single exponential in t and the integral has an exact antiderivative.
    Summing segments and inverting inside the crossing segment is exact for multi-segment
    gradients, holds and steps, with no grid error.
  * `retention_time_numeric`: brute-force cumulative quadrature for any k(phi), used for the
    optional pH extension and as an independent check of the reference.

`retention_time_closed_form` (single linear gradient) is a test oracle only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares

from lcms_sim.lc.gradient import LCConditions

LN10 = math.log(10.0)


def k_lss(phi, log_kw: float, S: float):
    return 10.0 ** (log_kw - S * np.asarray(phi, dtype=float))


def retention_time(log_kw: float, S: float, cond: LCConditions) -> float:
    """Exact LSS retention time (min). Returns inf if the analyte never elutes (S <= 0 is rejected)."""
    if S <= 0:
        raise ValueError("S must be positive")
    remaining = 1.0
    segs = cond.inlet_segments()
    ends = [s for s, _, _ in segs[1:]] + [math.inf]
    for (start, phi0, slope), end in zip(segs, ends):
        c =10.0 ** (S * phi0 - log_kw) / cond.t0          # integrand at segment start, 1/min
        r = S * slope * LN10                                # integrand growth rate, 1/min
        length = end - start
        if r == 0.0:
            area = c * length
        else:
            area = c * math.expm1(r * length) / r if math.isfinite(length) else math.inf
        if area >= remaining:
            if r == 0.0:
                x = remaining / c
            else:
                u = remaining * r / c
                if u <= -1.0:  # decreasing integrand that cannot reach `remaining`
                    remaining -= area
                    continue
                # log1p, not log(1 + u): a hold whose slope comes out as 1e-16 makes u tiny, and log(1 + u)
                # would quantise the retention time in steps of about 0.2 min.
                x = math.log1p(u) / r
            return start + x + cond.t0
        remaining -= area
    return math.inf


def retention_time_numeric(
    k_of_phi: Callable[[np.ndarray], np.ndarray], cond: LCConditions,
    t_max: float = 200.0, dt: float = 2e-4,
) -> float:
    """Cumulative trapezoid of 1/(t0 k) on a fine grid, crossing found by linear interpolation."""
    t = np.arange(0.0, t_max + dt, dt)
    f = 1.0 / (cond.t0 * k_of_phi(cond.phi_inlet(t)))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * dt)])
    i = int(np.searchsorted(cum, 1.0))
    if i >= len(t):
        return math.inf
    frac = (1.0 - cum[i - 1]) / (cum[i] - cum[i - 1])
    return float(t[i - 1] + frac * dt + cond.t0)


def retention_time_closed_form(log_kw: float, S: float, phi0: float, dphi: float,
                               tG: float, t0: float, tD: float) -> float:
    """Single linear gradient from phi0 over tG, starting at t = 0. Test oracle only.

    tR = (t0/b) log10[2.303 k0 b (1 - tD/(t0 k0)) + 1] + t0 + tD,  b = S dphi t0 / tG
    Valid when tD < t0 k0; otherwise the analyte elutes isocratically at t0 (1 + k0).
    """
    k0 = 10.0 ** (log_kw - S * phi0)
    if tD >= t0 * k0:
        return t0 * (1.0 + k0)
    b = S * dphi * t0 / tG
    return (t0 / b) * math.log10(LN10 * k0 * b * (1.0 - tD / (t0 * k0)) + 1.0) + t0 + tD


# ---------------------------------------------------------------------------
# pH extension
# ---------------------------------------------------------------------------


def k_ph(phi, ph: float, pka: float, acid: bool,
         log_kw_neutral: float, S_neutral: float, log_kw_ionised: float, S_ionised: float):
    """Retention factor as the fraction-weighted average of neutral and ionised forms.

    Acid:  k = (k_HA + k_A- 10^(pH - pKa)) / (1 + 10^(pH - pKa))
    Base:  k = (k_B  + k_BH+ 10^(pKa - pH)) / (1 + 10^(pKa - pH))
    """
    r = 10.0 ** ((ph - pka) if acid else (pka - ph))
    kn = k_lss(phi, log_kw_neutral, S_neutral)
    ki = k_lss(phi, log_kw_ionised, S_ionised)
    return (kn + ki * r) / (1.0 + r)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LSSFit:
    log_kw: float
    S: float
    residuals_pct: tuple[float, ...]   # (predicted - measured) / measured * 100, per run

    @property
    def flagged(self) -> bool:
        return any(abs(r) > 2.0 for r in self.residuals_pct)


def fit_lss(runs: Sequence[tuple[LCConditions, float]], init: tuple[float, float] = (3.0, 4.0)) -> LSSFit:
    """Fit log kw and S from two or more runs of different gradient (relative residuals)."""
    if len(runs) < 2:
        raise ValueError("need at least two scouting runs")

    def resid(p):
        lkw, S = p
        return [(retention_time(lkw, S, c) - tr) / tr for c, tr in runs]

    sol = least_squares(resid, init, bounds=([-2.0, 0.1], [12.0, 20.0]), x_scale=[1.0, 1.0],
                        xtol=1e-12, ftol=1e-12, gtol=1e-12)
    lkw, S = sol.x
    return LSSFit(float(lkw), float(S), tuple(100.0 * r for r in resid(sol.x)))


def fit_log_kw(tR: float, S: float, cond: LCConditions) -> float:
    """log kw that reproduces one measured retention time, with S assumed.

    Used when only one gradient is available (literature retention times).
    """
    if tR <= cond.t0:
        raise ValueError(f"retention time {tR} is not after t0 {cond.t0}")
    return float(brentq(lambda lkw: retention_time(lkw, S, cond) - tR, -3.0, 15.0, xtol=1e-12))


def estimate_S(molecular_weight: float) -> float:
    """Rough S for small molecules from molecular weight: S ~ 0.25 MW^0.5.

    Snyder and Dolan, High-Performance Gradient Elution (Wiley, 2007). An assumption,
    used only when scouting runs on two gradients are not available.
    """
    return 0.25 * math.sqrt(molecular_weight)
