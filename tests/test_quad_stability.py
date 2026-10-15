"""The stability diagram drawn in the Q1 close-up: series boundaries against a direct Floquet calculation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from quad_stability import (APEX_A, APEX_Q, APEX_RATIO, floquet_trace, ratio_for_width,  # noqa: E402
                            rf_amplitude_v, stable, upper_boundary, window)


def test_apex_where_the_textbook_puts_it():
    assert APEX_Q == pytest.approx(0.706, abs=2e-3)
    assert APEX_A == pytest.approx(0.237, abs=2e-3)


@pytest.mark.parametrize("q", [0.3, 0.5, 0.65, 0.706, 0.75, 0.85])
def test_series_boundary_matches_direct_calculation(q):
    top = float(upper_boundary(q))
    assert stable(0.97 * top, q)                 # just under the drawn boundary: stable
    assert not stable(top + 0.01, q)             # just over it: unstable


def test_floquet_trace_free_motion():
    assert floquet_trace(1.0, 0.0) == pytest.approx(2 * __import__("math").cos(__import__("math").pi), rel=1e-6)


def test_window_narrows_toward_the_apex():
    w_far, w_near = window(0.8 * APEX_RATIO, 300.0), window(0.99 * APEX_RATIO, 300.0)
    assert (w_far[1] - w_far[0]) > (w_near[1] - w_near[0]) > 0
    assert window(APEX_RATIO * 1.01, 300.0) is None


def test_ratio_for_width_round_trip():
    r = ratio_for_width(0.7, 289.1)
    lo, hi, _ = window(r, 289.1)
    assert hi - lo == pytest.approx(0.7, rel=1e-6)
    assert lo < 289.1 < hi


def test_rf_amplitude_scales_with_mz():
    assert rf_amplitude_v(700.0, 0.7) == pytest.approx(10 * rf_amplitude_v(70.0, 0.7))


def test_paths_light_ion_hits_the_plus_pair_heavy_ion_the_minus_pair():
    from quad_stability import paths
    r = ratio_for_width(0.7, 289.1)
    _, _, q_mid = window(r, 289.1)
    on = paths(r * q_mid, q_mid, 80)
    assert on["hit_x"] is None and on["hit_y"] is None
    light_q = q_mid * 289.1 / 288.1
    heavy_q = q_mid * 289.1 / 290.1
    light, heavy = paths(r * light_q, light_q, 80), paths(r * heavy_q, heavy_q, 80)
    assert light["hit_x"] is not None and light["hit_y"] is None    # thrown out by the +U pair's plane
    assert heavy["hit_y"] is not None and heavy["hit_x"] is None    # drifts out in the -U pair's plane


def test_effective_a_matches_textbook_and_the_three_ions():
    from quad_stability import effective_a
    for q in (0.1, 0.2):
        assert effective_a(0.0, q, "x")[0] == pytest.approx(q**2 / 2, rel=0.03)   # pseudopotential limit
    assert effective_a(0.2, 0.0, "x") == (pytest.approx(0.2), "bowl")             # DC only: bowl along x
    assert effective_a(0.2, 0.0, "y")[1] == "hill"                                # ...and a hill along y
    r = ratio_for_width(0.7, 289.1)
    _, _, q_mid = window(r, 289.1)
    on = [effective_a(r * q_mid, q_mid, p)[1] for p in ("x", "y")]
    q_l, q_h = q_mid * 289.1 / 288.1, q_mid * 289.1 / 290.1
    assert on == ["bowl", "bowl"]
    assert effective_a(r * q_h, q_h, "y")[1] == "hill"        # heavier: the DC hill wins in the - plane
    assert effective_a(r * q_l, q_l, "x")[1] == "rf"          # lighter: driven out by the RF in the + plane


def test_mathieu_aq_knobs():
    from quad_stability import mathieu_aq
    a, q = mathieu_aq(289.1, 1, rf_amplitude_v(289.1, 0.5), 0.0)
    assert q == pytest.approx(0.5, rel=1e-9) and a == 0.0
    assert mathieu_aq(600, 2, 300, 40) == pytest.approx(mathieu_aq(300, 1, 300, 40))   # only m/z matters
    a1, q1 = mathieu_aq(300, 1, 300, 40, freq_hz=1e6)
    a2, q2 = mathieu_aq(300, 1, 300, 40, freq_hz=2e6)
    assert q2 == pytest.approx(q1 / 4) and a2 == pytest.approx(a1 / 4)                  # 1 / f^2
    assert mathieu_aq(300, 1, 300, 0, r0_m=8e-3)[1] == pytest.approx(mathieu_aq(300, 1, 300, 0)[1] / 4)


def test_slow_swing_matches_pseudopotential_and_vanishes_when_lost():
    import math
    from quad_stability import slow_swing_cycles
    q = 0.1                                       # small q, a = 0: beta ~ q / sqrt 2, period ~ 2 sqrt 2 / q
    assert slow_swing_cycles(0.0, q, "x") == pytest.approx(2 * math.sqrt(2) / q, rel=0.02)
    assert slow_swing_cycles(0.0, 0.95, "x") is None             # past q = 0.908: shaken out
    assert slow_swing_cycles(0.05, 0.0, "y") is None             # DC only: a hill in y
    assert slow_swing_cycles(0.0, 0.2, "x") < slow_swing_cycles(0.0, 0.1, "x")   # deeper cup, faster swing
