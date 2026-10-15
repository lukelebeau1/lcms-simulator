"""Crosstalk between transitions: another compound's ions getting through both mass filters.

A transition j counts every ion that Q1 passes at its precursor setting AND that, after the
collision cell, Q3 passes at its product setting. A co-eluting compound k contributes when one
of its own precursor > product pairs falls inside both windows. With unit-resolution windows
modelled as Gaussians of the given FWHM (relative transmission 1 on setting):

    factor(j <- k) = sum over k's transitions t:  r_t x T1(P_j - P_t) x T3(p_j - p_t)

where r_t is the area of k's transition t relative to k's quantifier (its relative response).
Transition j then picks up factor x (k's quantifier peak), at k's retention time and shape.

k's fragmentation is taken from k's own transitions in the method: fragments the method does not
list are unknown and ignored, and r_t is at k's collision energy, not j's (assumed). The window
shape is an approximation of a real quadrupole's flat-topped pass band.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from lcms_sim.io.schemas import Transition

FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


@dataclass(frozen=True)
class Crosstalk:
    source: str      # the compound whose ions leak in
    factor: float    # area in this transition per unit of the source's quantifier area


def window_transmission(delta_mz: float, fwhm_mz: float) -> float:
    """Relative transmission of a quadrupole set `delta_mz` away from the ion, Gaussian window."""
    s = fwhm_mz * FWHM_TO_SIGMA
    return math.exp(-0.5 * (delta_mz / s) ** 2)


def crosstalk_map(transitions: Sequence[Transition], q1_fwhm_mz: float = 0.7, q3_fwhm_mz: float = 0.7,
                  min_factor: float = 1e-4) -> dict[int, list[Crosstalk]]:
    """For each transition index, the other compounds that leak into it and by how much."""
    by_analyte: dict[str, list[Transition]] = {}
    for t in transitions:
        by_analyte.setdefault(t.analyte, []).append(t)
    out: dict[int, list[Crosstalk]] = {}
    for j, tj in enumerate(transitions):
        for k, own in by_analyte.items():
            if k == tj.analyte:
                continue
            f = sum((t.relative_response or 1.0)
                    * window_transmission(tj.precursor_mz - t.precursor_mz, q1_fwhm_mz)
                    * window_transmission(tj.product_mz - t.product_mz, q3_fwhm_mz)
                    for t in own if t.polarity is tj.polarity)
            if f >= min_factor:
                out.setdefault(j, []).append(Crosstalk(k, f))
    return out
