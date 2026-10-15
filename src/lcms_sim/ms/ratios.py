"""Ion ratios: qualifier area over quantifier area, and their counting-limited precision.

    RSD_ratio ~ sqrt(1 / N_quant + 1 / N_qual)

valid when background is negligible and only counting noise acts. Proportional noise shared
by both transitions (per injection) cancels in the ratio; per-point noise does not.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from lcms_sim.io.schemas import TransitionRole
from lcms_sim.ms.noise import TransitionSignal


def counting_limit_rsd(n_quant: float, n_qual: float) -> float:
    return math.sqrt(1.0 / n_quant + 1.0 / n_qual)


@dataclass(frozen=True)
class IonRatio:
    analyte: str
    transition_index: int
    observed: float
    expected: float

    @property
    def deviation(self) -> float:
        """Relative deviation from the expected ratio."""
        return self.observed / self.expected - 1.0

    def within(self, tolerance: float = 0.30) -> bool:
        """Default tolerance +-30% relative, as in SANTE/11312/2021 for LC-MS/MS identification."""
        return abs(self.deviation) <= tolerance


def ion_ratios(signals: Sequence[TransitionSignal], expected: dict[int, float]) -> list[IonRatio]:
    """One ratio per qualifier, against its analyte's quantifier. `expected` maps transition index to ratio."""
    quant = {s.analyte: s for s in signals if s.role is TransitionRole.quantifier}
    out = []
    for s in signals:
        if s.role is TransitionRole.qualifier and s.analyte in quant and quant[s.analyte].area > 0:
            out.append(IonRatio(s.analyte, s.transition_index, s.area / quant[s.analyte].area,
                                expected.get(s.transition_index, float("nan"))))
    return out
