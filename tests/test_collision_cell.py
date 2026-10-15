"""Collision cell close-up: collision kinematics, the RF-only field, and fragments past the low-mass cutoff."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from collision_cell import N2_DA, Cell, collide, cutoff_mz, q_value, simulate  # noqa: E402
from quad_stability import DALTON, Q_END  # noqa: E402

M = N2_DA * DALTON


def test_collision_conserves_energy_and_momentum():
    rng = np.random.default_rng(1)
    m = np.full(500, 184 * DALTON)
    v = rng.normal(scale=3000, size=(500, 3))
    u = rng.normal(scale=300, size=(500, 3))
    v2 = collide(v, m, u, rng)
    # The gas molecule's velocity after the collision follows from momentum conservation.
    u2 = u + (m[:, None] * (v - v2)) / M
    ke = lambda vv, mm: 0.5 * mm * np.sum(vv**2, axis=1)  # noqa: E731
    assert np.allclose(ke(v, m) + ke(u, M), ke(v2, m) + ke(u2, M), rtol=1e-9)


@pytest.mark.parametrize("mass_da", [49.0, 184.0, 600.0])
def test_mean_energy_loss_is_the_hard_sphere_value(mass_da):
    """Gas at rest, isotropic scattering: the ion keeps on average 1 - 2 m M / (m + M)^2 of its energy."""
    rng = np.random.default_rng(2)
    m = np.full(200_000, mass_da * DALTON)
    v = np.zeros((m.size, 3))
    v[:, 2] = 1000.0
    v2 = collide(v, m, np.zeros_like(v), rng)
    lost = 1 - np.mean(np.sum(v2**2, axis=1)) / 1000.0**2
    mm = mass_da * DALTON
    assert lost == pytest.approx(2 * mm * M / (mm + M) ** 2, rel=0.01)


def test_rf_only_field_matches_the_stability_edge():
    """No gas, slow ions (about 250 RF cycles in the cell): held below q = 0.908, lost above."""
    cell = Cell(pressure_mtorr=0.0, axial_v=0.0)
    rng = np.random.default_rng(3)
    for q_target, lost in ((0.85, False), (0.95, True)):
        mz = q_value(1.0, cell) / q_target
        r = simulate(mz, mz, 0.1, 10**9, cell, 1, 0.05, rng)
        assert (r["status"][0] == "rod_prec") is lost, (q_target, r["status"][0])
    assert cutoff_mz(cell) == pytest.approx(q_value(1.0, cell) / Q_END)


def test_gas_cools_the_ion():
    r = simulate(289.1, 289.1, 20.0, 10**9, Cell(pressure_mtorr=10.0, axial_v=2.0), 1, 0.05,
                 np.random.default_rng(4))
    ke = r["first"]["ke_ev"]
    assert ke.max() < 20.5                                  # enters at 20 eV; the RF adds little
    assert np.median(ke[-50:]) < 0.2                       # down from 20 eV by > 100x
    assert r["ncoll"][0] > 20


def test_fragment_past_the_cutoff_is_lost_one_below_it_passes():
    cell = Cell(v_rf=100.0)                                 # cutoff about m/z 67
    rng = np.random.default_rng(5)
    lost = simulate(184.0, 49.0, 20.0, 1, cell, 40, 0.05, rng)
    kept = simulate(184.0, 143.0, 20.0, 1, cell, 40, 0.05, rng)
    assert q_value(49.0, cell) > Q_END > q_value(143.0, cell)
    assert np.mean(lost["status"] == "rod_prod") > 0.9
    assert np.mean(kept["status"] == "q3") > 0.8


def test_reproducible_with_the_same_seed():
    a = simulate(184.0, 95.0, 20.0, 3, Cell(), 20, 0.1, np.random.default_rng(7))
    b = simulate(184.0, 95.0, 20.0, 3, Cell(), 20, 0.1, np.random.default_rng(7))
    assert list(a["status"]) == list(b["status"]) and np.array_equal(a["ncoll"], b["ncoll"])
    assert math.isfinite(a["first"]["t"][-1])
