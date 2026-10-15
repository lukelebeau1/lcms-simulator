"""Peak width and shape.

    sigma_t = G t0 (1 + ke) / sqrt(N)

ke is k at the moment of elution, evaluated at phi_inlet(tR - t0). G is the band compression
factor, default 1. Extra-column broadening adds in quadrature.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.special import erfc, erfcx

from lcms_sim.lc.gradient import LCConditions
from lcms_sim.lc.retention import LN10, k_lss


def k_at_elution(log_kw: float, S: float, tR: float, cond: LCConditions) -> float:
    return float(k_lss(cond.phi_inlet(tR - cond.t0), log_kw, S))


def snyder_G(k0: float, b: float) -> float:
    """Band compression factor, G = sqrt(1 + p + p^2/3) / (1 + p), p = 2.303 k0 b / (k0 + 1).

    Quoted in the build plan from Snyder and Dolan's gradient elution text and NOT yet checked
    against the source. Off by default.
    """
    warnings.warn("snyder_G is unverified against its source", stacklevel=2)
    p = LN10 * k0 * b / (k0 + 1.0)
    return math.sqrt(1.0 + p + p * p / 3.0) / (1.0 + p)


def peak_sigma(log_kw: float, S: float, tR: float, cond: LCConditions, plate_number: float,
               extra_column_sigma_s: float = 0.0, G: float = 1.0) -> float:
    """Gaussian standard deviation of the peak in time (min)."""
    ke = k_at_elution(log_kw, S, tR, cond)
    col = G * cond.t0 * (1.0 + ke) / math.sqrt(plate_number)
    ext = extra_column_sigma_s / 60.0
    return math.hypot(col, ext)


def gaussian(t, center: float, sigma: float, area: float = 1.0):
    t = np.asarray(t, dtype=float)
    return area / (sigma * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def emg(t, center: float, sigma: float, tau: float, area: float = 1.0):
    """Exponentially modified Gaussian: Gaussian at `center` convolved with exp(-t/tau)/tau.

    The apex sits after `center` and the mean is center + tau. tau = 0 gives the Gaussian.
    Written with erfcx so it stays finite for small tau.
    """
    if tau <= 0:
        return gaussian(t, center, sigma, area)
    t = np.asarray(t, dtype=float)
    z = (t - center) / sigma
    s = sigma / tau
    x = (s - z) / math.sqrt(2)
    # Two equal forms of exp(s^2/2 - s z) erfc(x), each used where it cannot overflow:
    # x >= 0 (leading edge): exp(-z^2/2) erfcx(x);  x < 0 (tail): exp(s^2/2 - s z) erfc(x).
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        lead = np.exp(-0.5 * z * z) * erfcx(np.maximum(x, 0.0))
        tail = np.exp(0.5 * s * s - s * z) * erfc(np.minimum(x, 0.0))
    return area / (2 * tau) * np.where(x >= 0, lead, tail)
