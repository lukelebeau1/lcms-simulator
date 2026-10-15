"""The ions entering Q1 at one moment: every compound eluting, at its precursor m/z, plus background.

Shared by the spray close-up, the Q1 close-up and the molecules-to-response view, so they all
show the same population in the same colours. Built on the model's numbers where they exist:

  * ng of each compound on the column: the bench-step ledger (with the user's edits);
  * when and how sharply it elutes: retention and peak width from the LC model;
  * suppression: the matrix profile at that time;
  * ions per molecule reaching Q1: the simulator's response factor / gain, divided by the typical
    downstream fractions below (the same split as the molecules-to-response view).

Assumed, and labelled as such: those downstream fractions, Q1's pass window (unit resolution,
0.7 m/z FWHM, peak transmission 0.3), the share carried by a compound's second precursor (an
adduct), and the background: solvent/additive cluster ions falling off with m/z, plus matrix ions
around m/z 300-450 where the matrix profile shows suppression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lcms_sim.io.loaders import InputSet
from lcms_sim.io.schemas import Polarity
from lcms_sim.lc.gradient import LCConditions
from lcms_sim.lc.peaks import gaussian, peak_sigma
from lcms_sim.lc.retention import retention_time
from lcms_sim.quant.ledger import AVOGADRO
from lcms_sim.quant.matrix import MatrixProfile
from lcms_sim.trace import TYPICAL_DOWNSTREAM as _TYPICAL_DOWNSTREAM

# Typical orders of magnitude for a triple quad with conventional ESI; not measured here.
TYPICAL_DOWNSTREAM = list(_TYPICAL_DOWNSTREAM)   # the core's typical ion-optics shares (lcms_sim.trace)
Q1_PEAK, Q1_FWHM = 0.3, 0.7
SECOND_PRECURSOR_SHARE = 0.5            # a compound's other precursor (an adduct), relative to the quantifier's
BACKGROUND_TOTAL = 2.0e7                # solvent/additive cluster ions per s entering Q1, m/z 100-1000
MATRIX_TOTAL = 6.0e7                    # matrix ions per s at full suppression
PALETTE = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#ff7f0e"]
TARGET_COLOUR, BACKGROUND_COLOUR, MATRIX_COLOUR = "crimson", "gold", "#9a9a9a"


@dataclass(frozen=True)
class Species:
    name: str
    kind: str                # "target", "coeluting" or "background"
    mz: float
    polarity: Polarity
    rate: float              # ions per second entering Q1 at this moment
    colour: str


def ions_per_molecule(response_factor: float, gain: float, molecular_weight: float) -> float:
    """Ions entering Q1 per molecule injected: the model's response / gain, before the downstream stages."""
    lumped = response_factor / gain / (AVOGADRO * 1e-9 / molecular_weight)
    return lumped / math.prod(f for _, f, _ in TYPICAL_DOWNSTREAM)


def population(i: InputSet, cond: LCConditions, column_ng: dict[str, float], t: float, target: str,
               profile: MatrixProfile, in_matrix: bool = True) -> list[Species]:
    """Every compound with ions arriving at time t (within 4 sigma of its apex), one entry per precursor m/z."""
    mf = float(profile.factor(t)) if in_matrix else 1.0
    out, colours = [], iter(PALETTE * 4)
    for a in i.analytes:
        ng = column_ng.get(a.name, 0.0)
        if ng <= 0 or not a.molecular_weight:
            continue
        tR = retention_time(a.log_kw, a.S, cond)
        sig = peak_sigma(a.log_kw, a.S, tR, cond, i.system.plate_number, i.system.extra_column_sigma_s)
        if abs(t - tR) > 4 * sig:
            continue
        mol_per_s = ng * AVOGADRO * 1e-9 / a.molecular_weight * float(gaussian(t, tR, sig)) / 60.0
        rate = mol_per_s * ions_per_molecule(a.response_factor, i.acquisition.gain_area_per_ion, a.molecular_weight) * mf
        trs = [x for x in i.transitions if x.analyte == a.name]
        quant = next((x for x in trs if x.role.value == "quantifier"), trs[0])
        colour = TARGET_COLOUR if a.name == target else next(colours)
        kind = "target" if a.name == target else "coeluting"
        seen = set()
        for x in [quant] + trs:
            key = (round(x.precursor_mz, 1), x.polarity)
            if key in seen:
                continue
            seen.add(key)
            share = 1.0 if x is quant else SECOND_PRECURSOR_SHARE
            out.append(Species(a.name, kind, x.precursor_mz, x.polarity, rate * share, colour))
    return out


_GRID = np.linspace(100.0, 1000.0, 9001)
_SOLVENT_NORM = float(np.trapezoid(np.exp(-(_GRID - 100.0) / 250.0), _GRID))
_MATRIX_NORM = float(np.trapezoid(np.exp(-0.5 * ((_GRID - 380.0) / 60.0) ** 2), _GRID))


def background_density(mz, matrix_suppression: float) -> np.ndarray:
    """Background ions per s per unit m/z entering Q1: solvent/additive clusters plus matrix (assumed).

    Normalised over m/z 100-1000, so BACKGROUND_TOTAL and MATRIX_TOTAL are the totals over that range.
    """
    mz = np.asarray(mz, dtype=float)
    solvent = BACKGROUND_TOTAL * np.exp(-(mz - 100.0) / 250.0) / _SOLVENT_NORM
    matrix = MATRIX_TOTAL * matrix_suppression * np.exp(-0.5 * ((mz - 380.0) / 60.0) ** 2) / _MATRIX_NORM
    return np.where(mz >= 100.0, solvent + matrix, 0.0)


def q1_transmission(mz, setpoint: float):
    """Fraction of ions of a given m/z that Q1 passes when set to `setpoint` (Gaussian window, assumed)."""
    s = Q1_FWHM / 2.3548
    return Q1_PEAK * np.exp(-0.5 * ((np.asarray(mz, dtype=float) - setpoint) / s) ** 2)


def background_in_window(setpoint: float, matrix_suppression: float) -> float:
    """Background ions per s that Q1 passes at this setpoint."""
    mz = np.linspace(setpoint - 3, setpoint + 3, 241)
    return float(np.trapezoid(background_density(mz, matrix_suppression) * q1_transmission(mz, setpoint), mz))
