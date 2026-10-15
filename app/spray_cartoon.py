"""Close-up of electrospray for the ion path view: a cartoon driven by textbook trends.

Not a model of the signal; nothing here feeds the simulator. It draws what happens and makes the
drawing respond in the right direction to the settings, using:

  * Solvent properties at 20 C (CRC Handbook): vapour pressure water 2.34 kPa, methanol 13.0 kPa;
    surface tension water 72.8, methanol 22.6 mN/m. Mixture vapour pressure is a volume-weighted
    average (ideal mixing, an approximation); mixture surface tension falls steeply with the first
    methanol, approximated as gamma = gamma_w - (gamma_w - gamma_m) sqrt(phi).
  * d-squared law of droplet evaporation: r^2 falls linearly in time, at a rate that scales with
    the solvent's vapour pressure and the heat the drying gas supplies.
  * Rayleigh limit: a droplet can hold at most q_R ~ sqrt(gamma r^3) of charge; beyond it, it
    sheds offspring carrying about 2% of the mass and 15% of the charge (Kebarle and Verkerk).

Scalings for gas temperature, gas flow, nebuliser pressure and voltage are illustrative and
monotonic in the known direction; their magnitudes are not calibrated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

P_WATER, P_MEOH = 2.34, 13.0          # kPa at 20 C
G_WATER, G_MEOH = 72.8, 22.6          # mN/m at 20 C

# Cartoon geometry (arbitrary units): needle tip at x = 0, capillary inlet at x = 100.
X_INLET, Y_HALF = 100.0, 28.0
R0_BASE, Q_RAYLEIGH_C, K_BASE = 7.0, 0.86, 0.137


@dataclass(frozen=True)
class SpraySettings:
    percent_b: float = 50.0           # methanol, % v/v
    gas_temp_c: float = 200.0
    gas_flow_l_min: float = 13.0     # Agilent 5994-1734EN source settings
    nebuliser_psi: float = 55.0
    voltage_v: float = 5000.0
    positive: bool = True


@dataclass(frozen=True)
class SolventState:
    vapour_pressure_kpa: float
    surface_tension_mn_m: float
    evaporation_rel: float            # relative to water at 200 C, 10 L/min
    droplet_size_rel: float           # initial radius relative to water at 55 psi
    charge_rel: float                 # starting charge relative to its Rayleigh limit


def solvent_state(s: SpraySettings) -> SolventState:
    phi = min(max(s.percent_b / 100.0, 0.0), 1.0)
    p = (1 - phi) * P_WATER + phi * P_MEOH
    gamma = G_WATER - (G_WATER - G_MEOH) * math.sqrt(phi)
    heat = max(s.gas_temp_c - 20.0, 5.0) / 180.0          # heat supplied, relative to 200 C gas
    evap = (p / P_WATER) * heat * math.sqrt(s.gas_flow_l_min / 10.0)
    size = (gamma / G_WATER) ** (1 / 3) * math.sqrt(55.0 / max(s.nebuliser_psi, 5.0))
    charge = 0.7 * min(s.voltage_v / 5000.0, 1.3)
    return SolventState(p, gamma, evap, size, charge)


def rayleigh_limit(r: float, gamma: float) -> float:
    """Most charges a droplet of radius r can hold (cartoon units)."""
    return Q_RAYLEIGH_C * math.sqrt((gamma / G_WATER) * r ** 3)


def simulate(s: SpraySettings, rng: np.random.Generator, species: tuple[tuple[str, float], ...] = (("crimson", 1.0),),
             n_frames: int = 150, warmup: int = 140) -> list[dict]:
    """Per-frame positions of droplets, charges, solvent, vapour, gas and released ions.

    `species` is (colour, weight) for the analytes in the eluent at this moment: each analyte
    molecule is drawn as one of them, so the ions it releases carry the same colour into Q1.
    """
    colours = [c for c, _ in species]
    w = np.array([max(x, 0.0) for _, x in species], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(colours), 1.0 / len(colours))

    def pick(n: int) -> list[str]:
        return [colours[k] for k in rng.choice(len(colours), size=n, p=w)]

    st = solvent_state(s)
    k_evap = K_BASE * st.evaporation_rel
    r0 = R0_BASE * st.droplet_size_rel
    drops, vapour, gas, ions, sions = [], [], [], [], []
    fissions = released = wet = 0
    frames = []
    for f in range(-warmup, n_frames):
        if f % 5 == 0:   # a droplet leaves the Taylor cone
            q = max(3, round(st.charge_rel * rayleigh_limit(r0, st.surface_tension_mn_m)))
            drops.append(dict(x=9.0, y=0.0, vx=0.75, vy=rng.normal(0, 0.12), r=r0, q=q,
                              an=pick(int(rng.integers(1, 3)))))
        for _ in range(2):   # drying gas flows out of the inlet toward the needle
            gas.append(dict(x=X_INLET, y=rng.uniform(-Y_HALF, Y_HALF), age=0))

        new = []
        for d in drops:
            d["x"] += d["vx"]
            d["y"] += d["vy"]
            r_old = d["r"]
            d["r"] = math.sqrt(max(d["r"] ** 2 - k_evap, 0.0))
            # Evaporated solvent leaves as vapour from the surface.
            lost = int(round(1.2 * (r_old ** 2 - d["r"] ** 2)))
            for _ in range(lost):
                a = rng.uniform(0, 2 * math.pi)
                vapour.append(dict(x=d["x"] + r_old * math.cos(a), y=d["y"] + r_old * math.sin(a),
                                   vx=0.3 * math.cos(a), vy=0.3 * math.sin(a), age=0))
            # Too much charge for the size: Rayleigh fission.
            if d["r"] > 0.6 and d["q"] >= rayleigh_limit(d["r"], st.surface_tension_mn_m):
                fissions += 1
                q_off = max(1, round(0.075 * d["q"]))
                for sgn in (-1, 1):
                    if d["q"] - q_off < 1:
                        break
                    d["q"] -= q_off
                    an = [d["an"].pop()] if d["an"] and rng.random() < 0.35 else []
                    new.append(dict(x=d["x"] + d["r"], y=d["y"], vx=1.0, vy=d["vy"] + sgn * 0.35,
                                    r=d["r"] * 0.01 ** (1 / 3), q=q_off, an=an))
                d["r"] *= 0.98 ** (1 / 3)
            # Small enough: what is left becomes free ions.
            if d["r"] < 0.8:
                for colour in d["an"][:d["q"]]:
                    ions.append(dict(x=d["x"], y=d["y"], c=colour))
                    released += 1
                for _ in range(max(d["q"] - len(d["an"]), 0)):
                    sions.append(dict(x=d["x"], y=d["y"] + rng.normal(0, 0.4)))
                d["gone"] = True
        for d in drops + new:   # still a droplet at the inlet: solvent and analyte go in un-ionised
            if not d.get("gone") and d["x"] >= X_INLET - 2:
                d["gone"] = True
                wet += 1 if f >= 0 else 0
        drops = [d for d in drops + new if not d.get("gone")]

        vg = 1.4 * math.sqrt(s.gas_flow_l_min / 10.0)
        for v in vapour:   # vapour drifts out, then the gas carries it away from the inlet
            v["x"] += v["vx"] - 0.6 * vg
            v["y"] += v["vy"]
            v["age"] += 1
        vapour = [v for v in vapour if v["age"] < 30 and v["x"] > -10]
        for g in gas:
            g["x"] -= vg
        gas = [g for g in gas if g["x"] > -10]
        for p in ions + sions:   # the field pulls ions to the inlet; the gas does not stop them
            p["x"] += 1.8
            p["y"] *= 0.93
        ions = [p for p in ions if p["x"] < X_INLET]
        sions = [p for p in sions if p["x"] < X_INLET]
        if f < 0:
            continue

        sol_x, sol_y, an_x, an_y, an_c, ch_x, ch_y, circles = [], [], [], [], [], [], [], []
        for d in drops:
            r, cx, cy = d["r"], d["x"], d["y"]
            circles.append((cx, cy, r))
            g = rng   # jiggle of molecules inside the droplet
            n_sol = int(round(1.2 * r * r))
            rr = r * np.sqrt(g.random(n_sol)) * 0.85
            th = g.uniform(0, 2 * math.pi, n_sol)
            sol_x += list(cx + rr * np.cos(th))
            sol_y += list(cy + rr * np.sin(th))
            for colour in d["an"]:
                a, rho = g.uniform(0, 2 * math.pi), r * 0.5 * g.random()
                an_x.append(cx + rho * math.cos(a))
                an_y.append(cy + rho * math.sin(a))
                an_c.append(colour)
            th_q = np.linspace(0, 2 * math.pi, d["q"], endpoint=False) + g.uniform(0, 1)
            ch_x += list(cx + (r + 0.6) * np.cos(th_q))
            ch_y += list(cy + (r + 0.6) * np.sin(th_q))
        frames.append(dict(
            solvent=(sol_x, sol_y), analyte=(an_x, an_y), analyte_c=an_c, charges=(ch_x, ch_y), circles=circles,
            vapour=([v["x"] for v in vapour], [v["y"] for v in vapour]),
            gas=([g["x"] for g in gas], [g["y"] for g in gas]),
            ions=([p["x"] for p in ions], [p["y"] for p in ions]), ions_c=[p["c"] for p in ions],
            sions=([p["x"] for p in sions], [p["y"] for p in sions]),
            n_drops=len(drops), fissions=fissions, released=released, wet=wet,
        ))
    return frames
