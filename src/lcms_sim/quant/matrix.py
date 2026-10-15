"""Matrix profile from post-column infusion.

    MF(t) = matrix-blank trace / solvent-blank trace, smoothed and interpolated at retention time.

MF = 1 means no effect; 0.4 means 60% suppression. A per-analyte scale stretches the
deviation from 1, because co-eluting compounds are not suppressed equally:
    MF_a = 1 - scale_a (1 - MF(t))
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from lcms_sim.io.schemas import InfusionPoint


@dataclass(frozen=True)
class MatrixProfile:
    times: np.ndarray
    mf: np.ndarray

    def factor(self, t, scale: float = 1.0):
        base = np.interp(t, self.times, self.mf)
        return np.maximum(1.0 - scale * (1.0 - base), 0.0)


def unity_profile() -> MatrixProfile:
    return MatrixProfile(np.array([0.0, 1e6]), np.array([1.0, 1.0]))


def profile_from_traces(matrix: Sequence[InfusionPoint], solvent: Sequence[InfusionPoint],
                        smooth_min: float = 0.05) -> MatrixProfile:
    tm = np.array([p.time_min for p in matrix])
    ts = np.array([p.time_min for p in solvent])
    sm = np.array([p.signal for p in matrix])
    ss = np.interp(tm, ts, np.array([p.signal for p in solvent]))
    ratio = np.where(ss > 0, sm / np.where(ss > 0, ss, 1.0), 1.0)
    step = np.median(np.diff(tm)) if tm.size > 1 else 1.0
    width = max(1, int(round(2 * smooth_min / step)) | 1)
    return MatrixProfile(tm, median_filter(ratio, size=width, mode="nearest"))
