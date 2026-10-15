"""The C18 surface scene: the drawn behaviour follows the retention numbers it is given."""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import c18_scene as C  # noqa: E402
import molecules as M  # noqa: E402


def parse(frame: str):
    lines = frame.split("\n")[2:]
    return [(p[0], np.array(list(map(float, p[1:])))) for p in (ln.split() for ln in lines)]


@pytest.mark.parametrize("kval", [0.25, 1.0, 4.0])
def test_held_share_is_k_over_1_plus_k(kval):
    n = 6000
    s = C.build(M.load("Carbaryl"), np.full(n, 0.6), np.full(n, kval), np.random.default_rng(1))
    assert np.mean(s.held) == pytest.approx(kval / (1 + kval), abs=0.04)


def test_strongly_retained_stays_put():
    s = C.build(M.load("Bifenthrin"), np.full(300, 0.1), np.full(300, 1e5), np.random.default_rng(2))
    assert all(s.held)


def test_methanol_count_follows_the_mole_fraction():
    assert C.methanol_mole_fraction(0.0) == 0.0 and C.methanol_mole_fraction(1.0) == 1.0
    assert C.methanol_mole_fraction(0.5) == pytest.approx((0.5 / 40.7) / (0.5 / 40.7 + 0.5 / 18.05))
    n_solv = 400
    s = C.build(M.load("Carbaryl"), np.array([0.2, 0.9]), np.array([1.0, 1.0]), np.random.default_rng(3),
                n_solvent=n_solv)
    for f, phi in enumerate((0.2, 0.9)):
        atoms = parse(s.frames[f])[s.n_analyte[f]:]
        n_meoh = sum(1 for e, _ in atoms if e == "C")
        assert n_meoh / n_solv == pytest.approx(C.methanol_mole_fraction(phi), abs=0.07)


def test_held_molecule_sits_polar_end_up_oily_end_in_the_chains():
    m = M.load("Myclobutanil")
    s = C.build(m, np.full(50, 0.3), np.full(50, 1e5), np.random.default_rng(4))
    c = M.contributions(m)
    pos = np.array([p for _, p in parse(s.frames[10])[:s.n_analyte[10]]])
    oily = (pos * np.clip(c, 0, None)[:, None]).sum(axis=0) / np.clip(c, 0, None).sum()
    polar = (pos * np.clip(-c, 0, None)[:, None]).sum(axis=0) / np.clip(-c, 0, None).sum()
    assert polar[2] > oily[2]                          # polar end up, toward the solvent
    assert pos.mean(axis=0)[2] < C.TOP_Z               # partitioned into the chain layer, not perched on top


def test_moderate_k_means_frequent_short_holds():
    s = C.build(M.load("Carbaryl"), np.full(600, 0.6), np.full(600, 1.0), np.random.default_rng(6))
    releases = sum(1 for a, b in zip(s.held, s.held[1:]) if a and not b)
    assert releases > 40                               # caught and released every ~10 frames, not once


def test_copies_show_the_time_average_as_an_average_over_molecules():
    n, kval, copies = 400, 1.5, 8
    s = C.build(M.load("Carbaryl"), np.full(n, 0.6), np.full(n, kval), np.random.default_rng(7), n_copies=copies)
    assert s.n_analyte[0] == copies * M.load("Carbaryl").GetNumAtoms()
    shares = [int(re.search(r"in this window, (\d+) held", t).group(1)) / copies for t in s.info]
    assert np.mean(shares) == pytest.approx(kval / (1 + kval), abs=0.05)


def test_eluted_frames_have_no_analyte_and_everything_stays_in_the_box():
    n = 80
    s = C.build(M.load("Myclobutanil"), np.linspace(0.1, 0.9, n), np.geomspace(1e3, 0.1, n),
                np.random.default_rng(5), eluted=np.arange(n) > 60)
    assert s.n_analyte[70] == 0 and s.n_analyte[10] == M.load("Myclobutanil").GetNumAtoms()
    for f in (0, 30, 79):
        z = np.array([p[2] for _, p in parse(s.frames[f])])
        assert z.min() > 0 and z.max() < C.CEIL_Z + 12
    assert "3Dmol" in C.html(s)


def test_second_compound_leaves_the_window_as_the_bands_separate():
    n = 120
    together = np.r_[np.ones(40), np.linspace(1, 0, 40), np.zeros(40)]
    a = C.Species(M.load("Myclobutanil"), "Myclobutanil", np.full(n, 0.5), np.full(n, 3.0), np.ones(n),
                  np.ones(n), [""] * n, np.zeros(n, dtype=bool))
    b = C.Species(M.load("Acephate"), "Acephate", np.full(n, 0.5), np.full(n, 0.2), together, np.ones(n),
                  [""] * n, np.zeros(n, dtype=bool))
    s = C.build_multi([a, b], np.random.default_rng(8), n_copies=4)
    n_b = [p.count(1) for p in s.present]
    assert n_b[10] == 4 and n_b[-1] == 0              # all in the window together, then gone
    assert s.present[-1].count(0) == 4                # the followed compound stays
    assert len(s.tags[10]) == 8                       # a name tag over every molecule


def test_centres_mark_each_drawn_copy_for_the_follow_camera():
    n = 30
    a = C.Species(M.load("Myclobutanil"), "Myclobutanil", np.full(n, 0.5), np.full(n, 3.0), np.ones(n),
                  np.ones(n), [""] * n, np.zeros(n, dtype=bool))
    b = C.Species(M.load("Acephate"), "Acephate", np.full(n, 0.5), np.full(n, 0.2), np.ones(n), np.ones(n),
                  [""] * n, np.zeros(n, dtype=bool))
    s = C.build_multi([a, b], np.random.default_rng(9), n_copies=3)
    assert s.names == ["Myclobutanil", "Acephate"]
    sizes = [M.load(nm).GetNumAtoms() for nm in s.names]
    for f in (0, 15, 29):
        atoms = parse(s.frames[f])
        assert len(s.centers[f]) == len(s.present[f]) == 6
        off = 0
        for sp, c in zip(s.present[f], s.centers[f]):
            pos = np.array([p for _, p in atoms[off:off + sizes[sp]]])
            off += sizes[sp]
            assert np.allclose(pos.mean(axis=0), c, atol=0.02)          # the centre is the copy's centroid
            assert np.max(np.linalg.norm(pos - c, axis=1)) <= s.extents[sp] + 0.05


def test_player_camera_rotates_and_zooms_but_never_pans():
    n = 5
    a = C.Species(M.load("Carbaryl"), "Carbaryl", np.full(n, 0.5), np.full(n, 1.0), np.ones(n), np.ones(n),
                  [""] * n, np.zeros(n, dtype=bool))
    page = C.html(C.build_multi([a], np.random.default_rng(10)))
    assert "nomouse: true" in page                        # 3Dmol's own handlers (which include a pan) are off
    assert 'View: close on Carbaryl' in page and "Rotate: turntable" in page
    assert "setView(" in page and "translate(" not in page
