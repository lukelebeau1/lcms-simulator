"""Calibration fits: linear or quadratic, weighted, with three ways to treat the origin.

Response y is area, or area ratio to the internal standard; x is the matching concentration
(or concentration ratio). Weighted least squares minimises sum w_i (y_i - f(x_i))^2.

Origin:
  ignore   fit the levels as given
  include  add (0, 0) as an extra point (weight taken from the lowest non-zero level)
  force    fit with no intercept
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lcms_sim.io.schemas import CalibrationModel, OriginHandling, Weighting


@dataclass(frozen=True)
class CalibrationFit:
    model: CalibrationModel
    weighting: Weighting
    origin: OriginHandling
    coef: tuple[float, ...]          # (intercept, slope) or (intercept, slope, curvature)
    r2: float
    x: tuple[float, ...]
    y: tuple[float, ...]
    accuracy_pct: tuple[float, ...]  # back-calculated / nominal x 100, per level (nan at x = 0)

    @property
    def intercept(self) -> float:
        return self.coef[0]

    @property
    def slope(self) -> float:
        return self.coef[1]

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        out = self.coef[0] + self.coef[1] * x
        if len(self.coef) == 3:
            out = out + self.coef[2] * x * x
        return out

    def inverse(self, y: float) -> float:
        """Concentration for a response. For a quadratic, the root on the rising branch nearest the data."""
        a0, a1 = self.coef[0], self.coef[1]
        if len(self.coef) == 2:
            return (y - a0) / a1 if a1 != 0 else math.nan
        a2 = self.coef[2]
        if a2 == 0:
            return (y - a0) / a1
        disc = a1 * a1 - 4 * a2 * (a0 - y)
        if disc < 0:
            return math.nan
        r = math.sqrt(disc)
        roots = [(-a1 + r) / (2 * a2), (-a1 - r) / (2 * a2)]
        # Choose the root where the curve is increasing (derivative > 0), as on the calibrated range.
        rising = [x for x in roots if a1 + 2 * a2 * x > 0]
        pool = rising or roots
        return min(pool, key=lambda x: abs(x - np.median(self.x)))


def weights(x: np.ndarray, weighting: Weighting) -> np.ndarray:
    if weighting is Weighting.none:
        return np.ones_like(x)
    pos = x[x > 0]
    floor = pos.min() if pos.size else 1.0
    xx = np.where(x > 0, x, floor)
    return 1.0 / xx if weighting is Weighting.inv_x else 1.0 / (xx * xx)


def fit_calibration(
    x: Sequence[float],
    y: Sequence[float],
    model: CalibrationModel = CalibrationModel.linear,
    weighting: Weighting = Weighting.none,
    origin: OriginHandling = OriginHandling.ignore,
) -> CalibrationFit:
    x0 = np.asarray(x, dtype=float)
    y0 = np.asarray(y, dtype=float)
    xs, ys = x0, y0
    if origin is OriginHandling.include:
        xs, ys = np.append(x0, 0.0), np.append(y0, 0.0)
    w = weights(xs, weighting)

    cols = [xs] if origin is OriginHandling.force else [np.ones_like(xs), xs]
    if model is CalibrationModel.quadratic:
        cols.append(xs * xs)
    X = np.column_stack(cols)
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], ys * sw, rcond=None)
    coef = tuple(float(b) for b in ([0.0, *beta] if origin is OriginHandling.force else beta))

    pred = X @ beta
    ybar = np.sum(w * ys) / np.sum(w)
    ss_res = np.sum(w * (ys - pred) ** 2)
    ss_tot = np.sum(w * (ys - ybar) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan

    fit = CalibrationFit(model, weighting, origin, coef, r2, tuple(x0), tuple(y0), ())
    acc = tuple(100.0 * fit.inverse(yi) / xi if xi > 0 else math.nan for xi, yi in zip(x0, y0))
    return CalibrationFit(model, weighting, origin, coef, r2, tuple(x0), tuple(y0), acc)
