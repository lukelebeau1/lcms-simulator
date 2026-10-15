"""Retention by analogy: place compounds without a measured RT using the method's measured pesticide RTs.

Provenance: assumed (an estimate), used only to place matrix ions in time; analytes keep their measured RTs.
The measured RTs of the neutral pesticides are fitted against hydrophobicity (PubChem XLogP3) and size:

    tR = t_lo + (t_hi - t_lo) / (1 + exp(-(x - x0) / w)),   x = XLogP3 + b log10(mass / 300)

A logistic in x because a gradient compresses retention at both ends: very polar compounds come out near the
void, very hydrophobic ones bunch up where the gradient reaches 100% organic. The size term lets large
molecules (abamectin) elute later than log P alone says. The leave-one-out error of the fit is the stated
uncertainty of every placement. Nearest pesticides in (x) are reported as the analogues.

Assumptions: XLogP3 is a computed log P with its own errors (it runs high for chlorantraniliprole, boscalid,
fipronil); ionisation at the method pH is ignored (fit on neutral compounds only); same column, gradient, flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


def _x(xlogp, mass, b: float):
    return np.asarray(xlogp, dtype=float) + b * np.log10(np.asarray(mass, dtype=float) / 300.0)


def _curve(X, t_lo, t_hi, x0, w, b):
    return t_lo + (t_hi - t_lo) / (1.0 + np.exp(-(_x(X[0], X[1], b) - x0) / w))


@dataclass(frozen=True)
class AnalogyModel:
    t_lo: float
    t_hi: float
    x0: float
    w: float
    b: float
    loo_rmse_min: float
    loo_median_abs_min: float
    names: tuple[str, ...]
    x_train: tuple[float, ...]
    rt_train: tuple[float, ...]

    def predict(self, xlogp, mass) -> np.ndarray:
        return _curve(np.vstack([np.atleast_1d(xlogp), np.atleast_1d(mass)]), self.t_lo, self.t_hi, self.x0,
                      self.w, self.b)

    def analogues(self, xlogp: float, mass: float, k: int = 3) -> list[tuple[str, float]]:
        """The k pesticides nearest in the fitted hydrophobicity scale, with their measured RTs."""
        x = float(_x(xlogp, mass, self.b))
        order = np.argsort(np.abs(np.asarray(self.x_train) - x))[:k]
        return [(self.names[i], self.rt_train[i]) for i in order]


def _fit(xlogp, mass, rt, t_end):
    lo = [0.0, float(np.max(rt)), -10.0, 0.05, -10.0]
    hi = [float(np.min(rt)), t_end, 20.0, 10.0, 10.0]
    p0 = [0.9 * float(np.min(rt)), min(t_end, float(np.max(rt)) + 0.5), float(np.median(xlogp)), 1.0, 2.0]
    p, _ = curve_fit(_curve, np.vstack([xlogp, mass]), rt, p0=p0, bounds=(lo, hi), maxfev=20000)
    return p


def fit_analogy(names, xlogp, mass, rt, t_end: float) -> AnalogyModel:
    """Fit the curve to measured RTs and report its leave-one-out error.

    `t_end` (min) caps the upper plateau: the time the end of the 100% organic hold reaches the column outlet
    (programmed end + dwell time + void time). The pesticides alone do not reach the plateau, so without the
    cap it runs away.
    """
    names = list(names)
    xlogp, mass, rt = (np.asarray(v, dtype=float) for v in (xlogp, mass, rt))
    p = _fit(xlogp, mass, rt, t_end)
    res = []
    for i in range(rt.size):
        m = np.arange(rt.size) != i
        pi = _fit(xlogp[m], mass[m], rt[m], t_end)
        res.append(rt[i] - _curve(np.vstack([xlogp[i:i + 1], mass[i:i + 1]]), *pi)[0])
    res = np.asarray(res)
    return AnalogyModel(*map(float, p), loo_rmse_min=float(np.sqrt(np.mean(res**2))),
                        loo_median_abs_min=float(np.median(np.abs(res))), names=tuple(names),
                        x_train=tuple(map(float, _x(xlogp, mass, p[4]))), rt_train=tuple(map(float, rt)))
