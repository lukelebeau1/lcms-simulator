"""Collision cell close-up: ions through an RF-only quadrupole filled with nitrogen.

Radial: the Q1 field with U = 0, x'' = (2 z e / (m r0^2)) V cos(W t) x and y'' the opposite sign (Mathieu with
a = 0, q = 4 z e V / (m r0^2 W^2), as in quad_stability). Axial: a small steady push along the cell (real cells
have one so cooled ions do not stall). Gas: elastic hard-sphere collisions with N2 at 300 K, scattering isotropic
in the centre-of-mass frame, at rate n sigma g (g the relative speed), so a heavy ion loses little energy per
collision and a light one a lot. Fragmentation is only a change of mass at a chosen collision, velocity kept:
when and into what it breaks are inputs, not predictions (fragmentation from first principles is out of scope).

Assumed, not the Ultivo's: the cell is a quadrupole (Agilent does not publish the "Vortex" cell's geometry; a
hexapole or octopole would have a softer low-mass cutoff), r0 4 mm, 1 MHz, 8 cm long, N2 pressure, and the
cross-section (150 A^2 at 200 Da, scaled as mass^(2/3)). Illustration only; not used by the signal model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quad_stability import DALTON, E_CHARGE, Q_END, R0_M, RF_HZ

KB, T_GAS, N2_DA = 1.380649e-23, 300.0, 28.0134
SIGMA_200 = 1.5e-18            # m^2 at 200 Da, assumed
TORR_PA = 133.322


@dataclass(frozen=True)
class Cell:
    v_rf: float = 60.0              # RF amplitude, V 0-peak
    pressure_mtorr: float = 5.0     # N2
    axial_v: float = 2.0            # steady push across the whole cell, V
    length_m: float = 0.08
    r0_m: float = R0_M
    freq_hz: float = RF_HZ
    t_max_s: float = 1.0e-3         # still inside after this: stalled
    per_cycle: int = 32             # integration steps per RF cycle


def sigma_m2(mass_da: float) -> float:
    return SIGMA_200 * (mass_da / 200.0) ** (2 / 3)


def q_value(mz: float, cell: Cell) -> float:
    return 4 * E_CHARGE * cell.v_rf / (mz * DALTON * cell.r0_m**2 * (2 * math.pi * cell.freq_hz) ** 2)


def cutoff_mz(cell: Cell) -> float:
    """Lightest m/z the cell holds (q = 0.908 there)."""
    return q_value(1.0, cell) / Q_END


def collide(v: np.ndarray, m: np.ndarray, u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Elastic hard-sphere collision of ions (velocity v, mass m in kg) with gas molecules of velocity u.

    In the centre-of-mass frame the relative speed is kept and its direction becomes random (isotropic).
    """
    M = N2_DA * DALTON
    vc = (m[:, None] * v + M * u) / (m[:, None] + M)
    g = np.linalg.norm(v - u, axis=1)
    n = rng.normal(size=v.shape)
    n /= np.linalg.norm(n, axis=1)[:, None]
    return vc + (M / (m + M) * g)[:, None] * n


def simulate(prec_mz: float, prod_mz: float, ce_v: float, breaks_at: int, cell: Cell, n_ions: int,
             start_offset: float, rng: np.random.Generator, z: int = 1, record_first: bool = True) -> dict:
    """Fly `n_ions` precursors (singly charged unless z given) through the cell; ion 0 is recorded in full.

    Each enters on axis-parallel flight at `ce_v` x z eV, `start_offset` x r0 off axis (ion 0 on the diagonal,
    the rest spread over a disc), and becomes the product at its `breaks_at`-th collision. Status per ion:
    "q3" product leaves the cell, "unbroken" precursor leaves it, "rod_prec" / "rod_prod" lost on a rod,
    "stalled" still inside at t_max.
    """
    M = N2_DA * DALTON
    omega = 2 * math.pi * cell.freq_hz
    dt = 1.0 / (cell.freq_hz * cell.per_cycle)
    n_gas = cell.pressure_mtorr * 1e-3 * TORR_PA / (KB * T_GAS)
    s_gas = math.sqrt(KB * T_GAS / M)                       # per-component gas speed spread
    m_prec, m_prod = prec_mz * z * DALTON, prod_mz * z * DALTON
    sig_prec, sig_prod = sigma_m2(prec_mz * z), sigma_m2(prod_mz * z)
    e_axial = z * E_CHARGE * cell.axial_v / cell.length_m

    pos = np.zeros((n_ions, 3))
    rr = start_offset * cell.r0_m * np.sqrt(rng.random(n_ions))
    th = rng.random(n_ions) * 2 * math.pi
    pos[:, 0], pos[:, 1] = rr * np.cos(th), rr * np.sin(th)
    pos[0, :2] = start_offset * cell.r0_m / math.sqrt(2)
    vel = np.zeros((n_ions, 3))
    vel[:, 2] = math.sqrt(2 * z * E_CHARGE * ce_v / m_prec)
    mass = np.full(n_ions, m_prec)
    sig = np.full(n_ions, sig_prec)
    ncoll = np.zeros(n_ions, dtype=int)
    broken = np.zeros(n_ions, dtype=bool)
    status = np.full(n_ions, "", dtype=object)
    t_end = np.full(n_ions, np.nan)

    def accel(p, m, t):
        k = 2 * z * E_CHARGE * cell.v_rf * math.cos(omega * t) / (m * cell.r0_m**2)
        return np.column_stack([k * p[:, 0], -k * p[:, 1], e_axial / m])

    rec = {"t": [], "x": [], "y": [], "z": [], "ke_ev": [], "prod": []}
    hits = {"t": [], "z": [], "x": [], "y": []}
    t, step = 0.0, 0
    live = np.ones(n_ions, dtype=bool)
    while live.any() and t < cell.t_max_s:
        i = np.flatnonzero(live)
        p, v, m = pos[i], vel[i], mass[i]
        v = v + 0.5 * accel(p, m, t) * dt
        p = p + v * dt
        v = v + 0.5 * accel(p, m, t + dt) * dt
        t += dt
        step += 1
        # Collisions by null-collision sampling: a bound on the rate, thinned by the true relative speed.
        g_max = np.linalg.norm(v, axis=1) + 6 * s_gas
        cand = rng.random(i.size) < n_gas * sig[i] * g_max * dt
        if cand.any():
            c = np.flatnonzero(cand)
            u = rng.normal(scale=s_gas, size=(c.size, 3))
            g = np.linalg.norm(v[c] - u, axis=1)
            ok = rng.random(c.size) < g / g_max[c]
            c, u = c[ok], u[ok]
            if c.size:
                v[c] = collide(v[c], m[c], u, rng)
                ncoll[i[c]] += 1
                brk = i[c][(ncoll[i[c]] == breaks_at) & ~broken[i[c]]]
                mass[brk], sig[brk], broken[brk] = m_prod, sig_prod, True
                if record_first and 0 in i[c]:
                    hits["t"].append(t)
                    hits["x"].append(p[i == 0][0, 0] / cell.r0_m)
                    hits["y"].append(p[i == 0][0, 1] / cell.r0_m)
                    hits["z"].append(p[i == 0][0, 2] / cell.length_m)
        pos[i], vel[i] = p, v
        if record_first and live[0] and step % 2 == 0:
            rec["t"].append(t)
            rec["x"].append(pos[0, 0] / cell.r0_m)
            rec["y"].append(pos[0, 1] / cell.r0_m)
            rec["z"].append(pos[0, 2] / cell.length_m)
            rec["ke_ev"].append(0.5 * mass[0] * float(vel[0] @ vel[0]) / E_CHARGE)
            rec["prod"].append(bool(broken[0]))
        on_rod = live & ((np.abs(pos[:, 0]) >= cell.r0_m) | (np.abs(pos[:, 1]) >= cell.r0_m))
        out = live & ~on_rod & (pos[:, 2] >= cell.length_m)
        back = live & ~on_rod & (pos[:, 2] < 0)                      # scattered back out of the entrance
        status[on_rod] = np.where(broken[on_rod], "rod_prod", "rod_prec")
        status[out] = np.where(broken[out], "q3", "unbroken")
        status[back] = "back"
        t_end[on_rod | out | back] = t
        live &= ~(on_rod | out | back)
    status[live] = "stalled"
    rec = {k: np.asarray(val) for k, val in rec.items()}
    return {"status": status, "t_end_s": t_end, "ncoll": ncoll, "first": rec,
            "hits": {k: np.asarray(val) for k, val in hits.items()},
            "q_prec": q_value(prec_mz, cell), "q_prod": q_value(prod_mz, cell), "cutoff_mz": cutoff_mz(cell)}
