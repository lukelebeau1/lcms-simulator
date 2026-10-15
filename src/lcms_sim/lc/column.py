"""Inside the column: where each band is, and what composition surrounds it, at any time.

A band at fraction x of the column length (0 inlet, 1 outlet) moves at the mobile-phase speed
slowed by retention:

    dx/dt = 1 / (t0 (1 + k(phi(x, t))))

and the mobile phase at x carries the composition that entered the inlet x t0 earlier:

    phi(x, t) = phi_inlet(t - x t0)

Writing the motion in terms of tau = t - x t0, the inlet time of the mobile phase the band
sits in, gives dx/dtau = 1 / (t0 k(phi_inlet(tau))). So

    x(tau) = integral_0^tau dtau' / (t0 k(phi_inlet(tau'))),   t = tau + x t0

which is the retention equation in `retention.py` read part way: x reaches 1 at tR. The
integral is evaluated with the same exact per-segment antiderivative, so the band leaves the
outlet at exactly `retention_time`.

Band width: plate height H = L / N gives sigma_x = sqrt(x / N) in column fractions. Gradient
compression (G = 1) and extra-column broadening are left out, as in `peaks.py` by default.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from lcms_sim.lc.gradient import LCConditions
from lcms_sim.lc.retention import LN10, k_lss


def migrated_fraction(log_kw: float, S: float, cond: LCConditions, tau: float) -> float:
    """x(tau): column fraction travelled once mobile phase that entered at inlet time tau reaches the band."""
    if tau <= 0:
        return 0.0
    total = 0.0
    segs = cond.inlet_segments()
    ends = [s for s, _, _ in segs[1:]] + [math.inf]
    for (start, phi0, slope), end in zip(segs, ends):
        if start >= tau:
            break
        length = min(end, tau) - start
        c = 10.0 ** (S * phi0 - log_kw) / cond.t0
        r = S * slope * LN10
        total += c * length if r == 0.0 else c * math.expm1(r * length) / r
    return total


def band_position(log_kw: float, S: float, cond: LCConditions, t: float) -> float:
    """Column fraction (0 inlet, 1 outlet) of the band centre at time t after injection.

    Returns 1.0 once the band has eluted (t >= tR).
    """
    if t <= 0:
        return 0.0

    def clock(tau: float) -> float:   # elapsed time when the band sits in mobile phase from inlet time tau
        return tau + cond.t0 * migrated_fraction(log_kw, S, cond, tau)

    # x(tau) only grows, so clock(tau) is increasing; find the tau where x reaches 1.
    hi = t
    if migrated_fraction(log_kw, S, cond, hi) >= 1.0:
        tau_out = brentq(lambda u: migrated_fraction(log_kw, S, cond, u) - 1.0, 0.0, hi, xtol=1e-12)
        if t >= tau_out + cond.t0:
            return 1.0
        hi = tau_out
    if clock(hi) <= t:   # numerically at the limit
        return min(migrated_fraction(log_kw, S, cond, hi), 1.0)
    tau = brentq(lambda u: clock(u) - t, 0.0, hi, xtol=1e-12)
    return min(migrated_fraction(log_kw, S, cond, tau), 1.0)


def phi_in_column(cond: LCConditions, x, t: float):
    """Organic fraction at column fraction x and time t: what entered the inlet x t0 earlier."""
    return cond.phi_inlet(t - np.asarray(x, dtype=float) * cond.t0)


def band_sigma_fraction(x: float, plate_number: float) -> float:
    """Band standard deviation along the column, as a fraction of its length: sqrt(x / N)."""
    return math.sqrt(max(x, 0.0) / plate_number)


def local_k(log_kw: float, S: float, cond: LCConditions, x: float, t: float) -> float:
    """Retention factor the band feels at its current position."""
    return float(k_lss(phi_in_column(cond, x, t), log_kw, S))


def band_trajectory(log_kw: float, S: float, cond: LCConditions, times, dtau: float = 1e-3) -> np.ndarray:
    """Band centre (column fraction) at each of `times`, 1.0 once eluted. Vectorised for animation.

    x(tau) is evaluated exactly on a fine grid of inlet times, mapped to clock time
    t = tau + t0 x, and interpolated; agrees with `band_position` to well under 1e-4.
    """
    times = np.asarray(times, dtype=float)
    tau = np.arange(0.0, max(float(times.max(initial=0.0)), 0.0) + 2 * dtau, dtau)
    x = np.zeros_like(tau)
    segs = cond.inlet_segments()
    ends = [s for s, _, _ in segs[1:]] + [math.inf]
    for (start, phi0, slope), end in zip(segs, ends):
        length = np.clip(tau - start, 0.0, end - start)
        c = 10.0 ** (S * phi0 - log_kw) / cond.t0
        r = S * slope * LN10
        x += c * length if r == 0.0 else c * np.expm1(r * length) / r
    x = np.minimum(x, 1.0)
    clock = tau + cond.t0 * x
    out = np.interp(times, clock, x, left=0.0, right=float(x[-1]))
    return np.where(times <= 0, 0.0, out)
