"""Retention by analogy to the measured pesticide RTs (the example's: Agilent 5994-1734EN on its gradient)."""

import numpy as np
import pandas as pd
import pytest

from lcms_sim.io.loaders import load_input_set
from lcms_sim.lc.analogy import fit_analogy
from lcms_sim.lc.gradient import LCConditions
from lcms_sim.paths import EXAMPLES, REFERENCE

_I = load_input_set(EXAMPLES)
_C = LCConditions.from_method(_I.system, _I.lc_method)
_HOLD_END = max(t for t, p in zip(_C.times, _C.phis) if p >= 1.0)
T_END = _HOLD_END + _C.tD + _C.t0     # end of the 100% B hold, as it reaches the column outlet


def test_recovers_a_known_curve():
    rng = np.random.default_rng(0)
    xl = rng.uniform(-2, 8, 60)
    mass = rng.uniform(150, 900, 60)
    x = xl + 1.5 * np.log10(mass / 300)
    rt = 1.0 + 11.0 / (1 + np.exp(-(x - 3.0) / 1.2))
    m = fit_analogy([f"c{i}" for i in range(60)], xl, mass, rt, t_end=12.0)
    assert m.loo_rmse_min < 1e-3
    assert m.predict(3.0, 300.0)[0] == pytest.approx(6.5, abs=1e-3)


@pytest.fixture(scope="module")
def fit():
    """Fitted to the neutral targets' measured retention times as the example's transition table lists them."""
    x = pd.read_csv(REFERENCE / "pesticide_xlogp.csv")
    rt = pd.DataFrame([(t.analyte, t.retention_time_min) for t in _I.transitions if t.role.value == "quantifier"],
                      columns=["compound", "rt"])
    ab = pd.DataFrame([(a.name, a.acid_base.value) for a in _I.analytes], columns=["compound", "acid_base"])
    d = x.merge(rt, on="compound").merge(ab, on="compound")
    d = d[d.acid_base == "neutral"]
    return fit_analogy(d.compound, d.xlogp, d.monoisotopic_mass, d.rt, t_end=T_END)


def test_fit_is_monotone_and_within_the_run(fit):
    xs = np.linspace(-6, 20, 200)
    rt = fit.predict(xs, np.full(xs.size, 300.0))
    assert np.all(np.diff(rt) >= 0)
    assert 0.5 < rt[0] and rt[-1] <= T_END + 1e-9     # void to the end of the 100% organic hold


def test_fit_error_is_what_the_docs_state(fit):
    assert fit.loo_rmse_min < 0.65          # 0.58 min on the 13 min run
    assert fit.loo_median_abs_min < 0.4     # 0.34 min


def test_analogues_are_pesticides_near_in_hydrophobicity(fit):
    names = [n for n, _ in fit.analogues(5.3, 348.9, k=3)]   # chlorpyrifos's own position
    assert "Chlorpyrifos" in names
