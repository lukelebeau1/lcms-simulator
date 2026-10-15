"""Quadrupole stability diagram for the Q1 close-up: why a mass filter passes a band of m/z.

An ion's sideways motion between the rods obeys the Mathieu equation
    u'' + (a - 2 q cos 2 xi) u = 0,   a = 8 z e U / (m r0^2 Omega^2),  q = 4 z e V / (m r0^2 Omega^2)
with a -> -a, q -> -q for the other rod pair. Both directions are stable inside the first stability
region, bounded on the left by a = -a0(q) and on the right by a = b1(q) (characteristic values of
the Mathieu functions, series from Abramowitz and Stegun 20.2.25; checked against a direct
Floquet calculation in tests). Fixed U and V put every ion on the line a / q = 2U/V, at q ~ 1/m;
the stretch of that line inside the region is the pass window.

The geometry (r0 4.0 mm) and RF frequency (1.0 MHz) used for the voltage readout are assumed,
typical orders of magnitude, not the Ultivo's specification. Illustration only; not used by the
signal model.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

R0_M, RF_HZ = 4.0e-3, 1.0e6          # assumed
E_CHARGE, DALTON = 1.602176634e-19, 1.66053906660e-27


def a0(q):
    """Left boundary is a = -a0(q) (y-direction); series in q."""
    q = np.asarray(q, dtype=float)
    return -q**2 / 2 + 7 * q**4 / 128 - 29 * q**6 / 2304 + 68687 * q**8 / 18874368


def b1(q):
    """Right boundary a = b1(q) (x-direction); series in q."""
    q = np.asarray(q, dtype=float)
    return 1 - q - q**2 / 8 + q**3 / 64 - q**4 / 1536 - 11 * q**5 / 36864


Q_END = brentq(lambda q: float(b1(q)), 0.8, 1.0)                     # b1 = 0, ~0.908
APEX_Q = brentq(lambda q: float(-a0(q) - b1(q)), 0.6, 0.8)          # ~0.706
APEX_A = float(b1(APEX_Q))                                            # ~0.237
APEX_RATIO = APEX_A / APEX_Q                                          # a/q at the apex = 2U/V


def upper_boundary(q):
    """Top of the stable region at each q (0 outside it)."""
    q = np.asarray(q, dtype=float)
    return np.where((q >= 0) & (q <= Q_END), np.clip(np.minimum(-a0(q), b1(q)), 0, None), 0.0)


def segment(ratio: float) -> tuple[float, float] | None:
    """Where the line a = ratio x q enters and leaves the stable region (q_left, q_right); None above the apex."""
    if ratio <= 0:
        return 0.0, Q_END
    if ratio >= APEX_RATIO:
        return None
    ql = brentq(lambda q: float(-a0(q)) - ratio * q, 1e-9, APEX_Q)
    qr = brentq(lambda q: float(b1(q)) - ratio * q, APEX_Q, Q_END)
    return ql, qr


def window(ratio: float, mz_set: float) -> tuple[float, float, float] | None:
    """Pass window (m/z low, m/z high) with the setpoint at the middle of the segment, and that q."""
    seg = segment(ratio)
    if seg is None:
        return None
    ql, qr = seg
    q_mid = 0.5 * (ql + qr)
    return mz_set * q_mid / qr, mz_set * q_mid / ql, q_mid


def ratio_for_width(width_mz: float, mz_set: float) -> float:
    """The a/q ratio (2U/V) that gives an ideal window `width_mz` wide at `mz_set`."""
    def width(r):
        w = window(r, mz_set)
        return (w[1] - w[0]) if w else 0.0
    return brentq(lambda r: width(r) - width_mz, 0.5 * APEX_RATIO, APEX_RATIO * (1 - 1e-12))


def rf_amplitude_v(mz: float, q: float) -> float:
    """RF amplitude V (0-peak) that puts a singly charged ion of `mz` at `q` (assumed r0 and frequency)."""
    omega = 2 * math.pi * RF_HZ
    return q * mz * DALTON * R0_M**2 * omega**2 / (4 * E_CHARGE)


def floquet_trace(a: float, q: float) -> float:
    """Trace of the one-period transfer matrix of the Mathieu equation; |trace| < 2 means stable."""
    def f(xi, y):
        return [y[1], -(a - 2 * q * math.cos(2 * xi)) * y[0]]
    s1 = solve_ivp(f, (0, math.pi), [1.0, 0.0], rtol=1e-10, atol=1e-12)
    s2 = solve_ivp(f, (0, math.pi), [0.0, 1.0], rtol=1e-10, atol=1e-12)
    return float(s1.y[0, -1] + s2.y[1, -1])


def stable(a: float, q: float) -> bool:
    """Both rod pairs stable: x at (a, q) and y at (-a, -q)."""
    return abs(floquet_trace(a, q)) < 2 and abs(floquet_trace(-a, -q)) < 2


def paths(a: float, q: float, n_cycles: float, u0: float = 0.01, per_cycle: int = 24) -> dict:
    """Sideways position in both planes (units of the rod gap r0) over `n_cycles` RF cycles.

    x is the plane of the +U pair (Mathieu a, q), y the plane of the -U pair (-a, -q); both start
    u0 off axis at rest. Positions stop (NaN) once the ion reaches a rod, |u| = 1.
    """
    cycles = np.arange(0.0, n_cycles + 1e-9, 1.0 / per_cycle)
    xi = cycles * math.pi                         # one RF cycle is pi in the Mathieu time
    out = {"cycles": cycles}
    for key, sa, sq in (("x", a, q), ("y", -a, -q)):
        def f(t, y, sa=sa, sq=sq):
            return [y[1], -(sa - 2 * sq * math.cos(2 * t)) * y[0]]

        def hit(t, y):
            return abs(y[0]) - 1.0
        hit.terminal = True
        s = solve_ivp(f, (0.0, xi[-1]), [u0, 0.0], t_eval=xi, events=hit, rtol=1e-8, max_step=0.05)
        u = np.full(xi.size, np.nan)
        u[:s.y.shape[1]] = s.y[0]
        out[key] = u
        out[f"hit_{key}"] = float(s.t_events[0][0] / math.pi) if s.t_events[0].size else None
    return out


def effective_a(a: float, q: float, plane: str) -> tuple[float, str]:
    """The averaged restoring strength an ion feels in one plane, exactly, as an equivalent static a.

    From the one-cycle transfer matrix (Floquet): a stable ion oscillates slowly at beta (cos pi beta
    = trace / 2), as if held by a static bowl a_eff = beta^2. An ion that grows steadily (trace > 2,
    growth rate mu, cosh pi mu = trace / 2) is on a hill, a_eff = -mu^2: the DC wins, as for a heavy
    ion in the - pair's plane. trace < -2 is growth locked to the RF itself (a light ion in the + pair's
    plane): no static equivalent, kind "rf". For small q and a = 0, a_eff = q^2 / 2 (the textbook
    pseudopotential).
    """
    sa, sq = (a, q) if plane == "x" else (-a, -q)
    tr = floquet_trace(sa, sq)
    if abs(tr) < 2:
        beta = math.acos(tr / 2) / math.pi
        return beta**2, "bowl"
    if tr >= 2:
        mu = math.acosh(tr / 2) / math.pi
        return -mu**2, "hill"
    return float("nan"), "rf"


def mathieu_aq(mass_da: float, z: int, v_rf: float, u_dc: float, freq_hz: float = RF_HZ,
               r0_m: float = R0_M) -> tuple[float, float]:
    """The two numbers that decide an ion's fate, from the knobs: a = 8 z e U / (m r0^2 W^2), q = 4 z e V / (m r0^2 W^2).

    Every knob enters only through these two, and mass and charge only as m/z: double both and nothing changes.
    """
    k = z * E_CHARGE / (mass_da * DALTON * r0_m**2 * (2 * math.pi * freq_hz) ** 2)
    return 8 * k * u_dc, 4 * k * v_rf


def slow_swing_cycles(a: float, q: float, plane: str) -> float | None:
    """RF cycles for one slow swing through the centre (the cup's own period, 2 / beta); None if not held.

    The ion's averaged motion in a bowl a_eff = beta^2 has angular frequency beta in the Mathieu time, where one
    RF cycle is pi; a full swing takes 2 pi / beta, i.e. 2 / beta cycles. Deeper cup, faster swing.
    """
    a_eff, kind = effective_a(a, q, plane)
    return 2 / math.sqrt(a_eff) if kind == "bowl" and a_eff > 0 else None
