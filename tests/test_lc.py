"""Phase 1: LC retention, fitting, peak width and shape, against the validation table."""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from lcms_sim.io.schemas import GradientProgram
from lcms_sim.lc.gradient import LCConditions, phi_at
from lcms_sim.lc.peaks import emg, gaussian, peak_sigma
from lcms_sim.lc.retention import (
    fit_log_kw,
    fit_lss,
    k_lss,
    k_ph,
    retention_time,
    retention_time_closed_form,
    retention_time_numeric,
)


def cond(points, t0=0.7, tD=1.0):
    prog = GradientProgram.model_validate({"points": [{"time_min": t, "percent_b": b} for t, b in points]})
    return LCConditions.from_program(prog, t0=t0, tD=tD)


# --- gradient --------------------------------------------------------------


def test_phi_interpolation_and_steps():
    times, phis = (0, 1, 1, 2), (0.1, 0.5, 0.9, 0.9)
    assert phi_at(times, phis, 0.5) == pytest.approx(0.3)
    assert phi_at(times, phis, 1.0) == pytest.approx(0.9)  # later value at the step
    assert phi_at(times, phis, -1) == pytest.approx(0.1)
    assert phi_at(times, phis, 5) == pytest.approx(0.9)


def test_inlet_is_delayed_by_dwell():
    c = cond([(0, 10), (10, 90)], tD=2.0)
    assert c.phi_inlet(1.5) == pytest.approx(0.10)
    assert c.phi_inlet(7.0) == pytest.approx(0.10 + 0.8 * 5 / 10)


# --- retention -------------------------------------------------------------


@pytest.mark.parametrize("lkw,S", [(1.0, 3.0), (3.0, 4.0), (5.0, 6.0)])
def test_flat_gradient(lkw, S):
    c = cond([(0, 40), (30, 40)])
    k = 10 ** (lkw - S * 0.4)
    assert retention_time(lkw, S, c) == pytest.approx(c.t0 * (1 + k), rel=1e-4)


@pytest.mark.parametrize("lkw,S", [(1.5, 3.0), (2.5, 3.5), (3.5, 4.5), (5.0, 5.5), (7.0, 7.0)])
def test_linear_gradient_matches_closed_form(lkw, S):
    t0, tD, tG = 0.7, 0.6, 10.0
    c = cond([(0, 5), (tG, 95), (100, 95)], t0=t0, tD=tD)
    expected = retention_time_closed_form(lkw, S, 0.05, 0.90, tG, t0, tD)
    if expected - t0 - tD < tG:  # eluted during the gradient, where the oracle applies
        assert retention_time(lkw, S, c) == pytest.approx(expected, rel=1e-3)


def test_elution_during_dwell_is_isocratic():
    lkw, S = 0.5, 3.0  # k0 small so it elutes before the gradient arrives
    c = cond([(0, 5), (10, 95)], t0=0.7, tD=3.0)
    k0 = 10 ** (lkw - S * 0.05)
    assert c.t0 * k0 < c.tD
    assert retention_time(lkw, S, c) == pytest.approx(c.t0 * (1 + k0), rel=1e-4)


@pytest.mark.parametrize("lkw,S", [(2.0, 3.5), (4.0, 5.0), (6.0, 6.5)])
def test_exact_solver_matches_brute_force_on_multisegment_program(lkw, S):
    c = cond([(0, 30), (1, 30), (2, 75), (8, 96), (9, 100), (9.5, 100), (9.51, 30)], t0=0.71, tD=1.2)
    exact = retention_time(lkw, S, c)
    numeric = retention_time_numeric(lambda p: k_lss(p, lkw, S), c, t_max=40)
    assert exact == pytest.approx(numeric, rel=1e-4)


def test_exact_solver_matches_adaptive_quadrature():
    lkw, S = 4.0, 5.0
    c = cond([(0, 30), (1, 30), (2, 75), (8, 96)], t0=0.71, tD=1.2)
    tR = retention_time(lkw, S, c)
    integral, _ = quad(lambda t: 1 / (c.t0 * k_lss(c.phi_inlet(t), lkw, S)), 0, tR - c.t0,
                       points=[1.2, 2.2, 3.2, 9.2], limit=200)
    assert integral == pytest.approx(1.0, rel=1e-8)


def test_strongly_retained_elutes_late_on_final_hold():
    c = cond([(0, 5), (1, 5)])
    k = 10 ** (12.0 - 0.5 * 0.05)
    assert retention_time(12.0, 0.5, c) == pytest.approx(c.t0 * (1 + k), rel=1e-6)


# --- fitting ---------------------------------------------------------------


@pytest.mark.parametrize("lkw,S", [(2.0, 3.0), (3.5, 4.5), (5.5, 6.0)])
def test_fit_recovers_parameters(lkw, S):
    runs = []
    for tG in (10.0, 20.0, 30.0):
        c = cond([(0, 5), (tG, 95), (tG + 10, 95)], t0=0.55, tD=0.9)
        runs.append((c, retention_time(lkw, S, c)))
    fit = fit_lss(runs)
    assert fit.log_kw == pytest.approx(lkw, rel=0.01)
    assert fit.S == pytest.approx(S, rel=0.01)
    assert not fit.flagged


def test_fit_log_kw_from_one_retention_time():
    c = cond([(0, 30), (1, 30), (2, 75), (8, 96), (9, 100)], t0=0.71, tD=1.2)
    tR = retention_time(4.2, 5.0, c)
    assert fit_log_kw(tR, 5.0, c) == pytest.approx(4.2, abs=1e-8)


def test_retention_is_continuous_across_a_hold_with_float_noise_slope():
    """A hold from 30% to 30% can come out with a slope of 1e-16; the solver must not quantise tR there
    (log(1 + u) did, in steps of ~0.2 min, for peaks eluting as the ramp reached the inlet)."""
    c = cond([(0, 30), (1, 30), (2, 75), (8, 96), (9, 100)], t0=0.7069, tD=1.2)
    lkws = np.linspace(1.40, 1.55, 31)
    trs = np.array([retention_time(x, 3.7831, c) for x in lkws])
    assert np.all(np.diff(trs) > 0)
    assert np.max(np.diff(trs)) < 0.05
    brute = retention_time_numeric(lambda phi: 10.0 ** (1.47 - 3.7831 * phi), c)
    assert retention_time(1.47, 3.7831, c) == pytest.approx(brute, abs=2e-3)


# --- pH --------------------------------------------------------------------


def test_ph_weighting_limits():
    kw = dict(log_kw_neutral=4.0, S_neutral=4.0, log_kw_ionised=2.0, S_ionised=3.5)
    neutral, ionised = k_lss(0.5, 4.0, 4.0), k_lss(0.5, 2.0, 3.5)
    assert k_ph(0.5, ph=1.0, pka=5.0, acid=True, **kw) == pytest.approx(neutral, rel=1e-3)
    assert k_ph(0.5, ph=11.0, pka=5.0, acid=True, **kw) == pytest.approx(ionised, rel=1e-3)
    assert k_ph(0.5, ph=5.0, pka=5.0, acid=True, **kw) == pytest.approx((neutral + ionised) / 2)
    assert k_ph(0.5, ph=9.0, pka=5.0, acid=False, **kw) == pytest.approx(neutral, rel=1e-3)


# --- peaks -----------------------------------------------------------------


def test_isocratic_peak_width():
    c = cond([(0, 40), (30, 40)])
    lkw, S, N = 3.0, 4.0, 10000
    tR = retention_time(lkw, S, c)
    k = 10 ** (lkw - S * 0.4)
    assert peak_sigma(lkw, S, tR, c, N) == pytest.approx(c.t0 * (1 + k) / math.sqrt(N), rel=1e-6)


def test_extra_column_adds_in_quadrature():
    c = cond([(0, 40), (30, 40)])
    tR = retention_time(3.0, 4.0, c)
    col = peak_sigma(3.0, 4.0, tR, c, 10000)
    both = peak_sigma(3.0, 4.0, tR, c, 10000, extra_column_sigma_s=1.2)
    assert both == pytest.approx(math.hypot(col, 0.02))


def test_peak_shapes_integrate_to_area():
    t = np.linspace(0, 20, 200001)
    assert np.trapezoid(gaussian(t, 5, 0.05, 3.0), t) == pytest.approx(3.0, rel=1e-6)
    y = emg(t, 5, 0.05, 0.1, 3.0)
    assert np.trapezoid(y, t) == pytest.approx(3.0, rel=1e-5)
    assert np.trapezoid(t * y, t) / 3.0 == pytest.approx(5.1, rel=1e-5)  # mean = center + tau
    assert np.allclose(emg(t, 5, 0.05, 0.0), gaussian(t, 5, 0.05))


# --- inside the column -------------------------------------------------------

LAB_GRADIENT = [(0, 10), (0.5, 10), (4, 75), (9.5, 87), (10, 100), (12, 100), (12.01, 10)]


@pytest.mark.parametrize("lkw,S", [(2.0, 4.0), (4.0, 5.0), (6.0, 6.5), (0.5, 3.0)])
def test_band_reaches_outlet_at_retention_time(lkw, S):
    from lcms_sim.lc.column import band_position

    c = cond(LAB_GRADIENT, t0=0.88, tD=1.0)
    tR = retention_time(lkw, S, c)
    assert band_position(lkw, S, c, tR - 1e-4) < 1.0
    assert band_position(lkw, S, c, tR - 1e-4) == pytest.approx(1.0, abs=1e-3)
    assert band_position(lkw, S, c, tR + 1e-6) == 1.0


def test_band_position_isocratic_is_linear_in_time():
    from lcms_sim.lc.column import band_position

    c = cond([(0, 40), (30, 40)], t0=1.0, tD=0.5)
    lkw, S = 2.0, 4.0
    k = float(k_lss(0.4, lkw, S))
    for t in (0.3, 1.0, 2.5):
        assert band_position(lkw, S, c, t) == pytest.approx(t / (1.0 + k), rel=1e-9)


def test_band_only_moves_forward_and_starts_at_inlet():
    from lcms_sim.lc.column import band_position

    c = cond(LAB_GRADIENT, t0=0.88, tD=1.0)
    xs = [band_position(4.0, 5.0, c, t) for t in np.linspace(0, 12, 121)]
    assert xs[0] == 0.0
    assert all(b >= a - 1e-12 for a, b in zip(xs, xs[1:]))


def test_composition_in_column_lags_the_inlet_by_travel_time():
    from lcms_sim.lc.column import phi_in_column

    c = cond(LAB_GRADIENT, t0=0.88, tD=1.0)
    t = 4.0
    assert phi_in_column(c, 0.0, t) == pytest.approx(c.phi_inlet(t))
    assert phi_in_column(c, 1.0, t) == pytest.approx(c.phi_inlet(t - 0.88))


def test_band_sigma_matches_peak_width_at_outlet():
    """At the outlet, sigma_x x t0 (1 + k) equals the time-domain sigma under isocratic conditions."""
    from lcms_sim.lc.column import band_sigma_fraction

    N, t0, k = 15000.0, 0.88, 3.0
    sigma_t = t0 * (1 + k) / math.sqrt(N)
    assert band_sigma_fraction(1.0, N) * t0 * (1 + k) == pytest.approx(sigma_t)


def test_band_trajectory_matches_band_position():
    from lcms_sim.lc.column import band_position, band_trajectory

    c = cond(LAB_GRADIENT, t0=0.88, tD=1.0)
    times = np.linspace(0, 15, 61)
    for lkw, S in [(0.5, 3.0), (2.0, 4.0), (4.0, 5.0), (6.0, 6.5)]:
        traj = band_trajectory(lkw, S, c, times)
        ref = [band_position(lkw, S, c, t) for t in times]
        assert traj == pytest.approx(ref, abs=1e-4)
