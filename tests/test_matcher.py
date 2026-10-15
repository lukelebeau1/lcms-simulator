"""Phase 6: the symptom matcher. The bundled sweep table is current, an injected fault ranks near the top,
the reference scenario ranks common causes first, and the confusion table documents what cannot be told apart."""

import numpy as np
import pytest

from lcms_sim.faults.library import LIBRARY, Scenario
from lcms_sim.faults.matcher import (BASELINE, BASELINE_SD, SWEEP_SEEDS, TIE_MARGIN, TOLERANCE_FLOOR, _mean_sd,
                                     clean_observation, confusion, default_checks, extract_run, load_sweep_table,
                                     match, separations, sweep_magnitudes, tolerances)
from lcms_sim.faults.signatures import FEATURE_MEANING, FEATURES
from lcms_sim.io.loaders import load_input_set
from lcms_sim.paths import EXAMPLES


@pytest.fixture(scope="module")
def inputs():
    return load_input_set(EXAMPLES)


@pytest.fixture(scope="module")
def table():
    return load_sweep_table()


@pytest.fixture(scope="module")
def checks():
    return default_checks()


def test_sweep_table_is_current(inputs, table):
    rows = {(r.fault, round(float(r.magnitude), 9)) for r in table.itertuples() if r.fault not in (BASELINE, BASELINE_SD)}
    want = {(k, round(float(m), 9)) for k, f in LIBRARY.items() for m in sweep_magnitudes(f)}
    assert rows == want, "the fault library or its sweep grid changed: rerun scripts/make_fault_sweeps.py"
    assert {BASELINE, BASELINE_SD} <= set(table.fault)
    assert set(table.columns) >= {"fault", "magnitude", *FEATURES}
    # One cell against a fresh simulation with the sweep seeds: the table was built from this pipeline.
    f = LIBRARY["volume_error"]
    m = float(sweep_magnitudes(f)[-1])
    mean, _ = _mean_sd([extract_run(f.apply(Scenario.baseline(inputs), m), s) for s in SWEEP_SEEDS])
    row = table[(table.fault == "volume_error") & np.isclose(table.magnitude, m)].iloc[0]
    for k in FEATURES:
        if np.isfinite(mean[k]):
            assert row[k] == pytest.approx(mean[k], rel=1e-6, abs=1e-9), k


def test_tolerances_cover_the_replicate_noise(table):
    tol = tolerances(table)
    sd = table[table.fault == BASELINE_SD].iloc[0]
    for k in FEATURES:
        assert tol[k] >= TOLERANCE_FLOOR[k]
        if np.isfinite(sd[k]):
            assert tol[k] >= 2 * sd[k]


def test_feature_meanings_cover_every_feature():
    assert set(FEATURE_MEANING) == set(FEATURES)
    assert all(v for v in FEATURE_MEANING.values())


def test_the_baseline_matches_no_fault(table, checks):
    base = table[table.fault == BASELINE].iloc[0]
    obs = {k: float(base[k]) for k in FEATURES}
    res = match(obs, table, checks=checks)
    assert res.leader().key == BASELINE and res.leader().consistent


@pytest.mark.parametrize("key", list(LIBRARY))
def test_an_injected_fault_ranks_in_the_top_three_or_is_documented_as_a_tie(inputs, table, checks, key):
    f = LIBRARY[key]
    obs = extract_run(f.apply(Scenario.baseline(inputs), f.default), seed=11)   # a seed outside the sweep's
    res = match(obs, table, checks=checks)
    d_self = next(m.distance for m in res.ranked if m.key == key)
    assert res.rank_of(key) <= 3 or d_self - res.leader().distance <= TIE_MARGIN, res.frame().head(5)


def test_reference_scenario_pt_at_30_percent_uniform(table, checks):
    """The plan's acceptance case: proficiency test at about 30% of assigned, uniform across analytes,
    dilution ruled out. Common causes first; retention-dependent suppression low."""
    obs = {"sample_bias": 0.30, "sample_bias_spread": 0.0, "bias_rt_corr": 0.0}
    res = match(obs, table, exclude=["volume_error"], checks=checks)
    top2 = {m.key for m in res.ranked[:2]}
    assert top2 == {"stock_strong", "istd_overspiked"}, res.frame().head(6)
    # Suppression is ruled out: its bias cannot reach 30% at any size in the library's range, and the matcher
    # says so by distance (tolerance units), not by its position among the other ruled-out faults.
    ms = next(m for m in res.ranked if m.key == "matrix_suppression")
    assert "sample_bias" in ms.unmatched and ms.distance - res.leader().distance > 5
    assert res.rank_of("matrix_suppression") > 3 and res.rank_of("low_recovery") > 2
    assert "volume_error" not in {m.key for m in res.ranked}
    lead = res.leader()
    assert lead.consistent and 2.5 < lead.magnitude < 4.0 if lead.key == "stock_strong" else True
    # The matcher names the bench check that separates the leading pair.
    names = {s.check.name for s in res.separations}
    assert "IS areas, samples against calibrators" in names or "QC from an independent stock" in names


def test_a_single_feature_narrows_to_the_faults_that_move_it(table):
    res = match({"rt_shift": 0.5}, table)                     # everything eluting half a minute late
    assert res.leader().key in ("dwell_volume", "organic_error")
    assert {"dwell_volume", "organic_error"} <= {m.key for m in res.ranked[:4]}
    assert match({"rt_shift": 0.5, "nothing": 1.0, "sample_bias": float("nan")}, table).leader().key == res.leader().key


def test_clean_observation_drops_unknown_and_missing():
    assert clean_observation({"sample_bias": 0.5, "x": 1, "rt_shift": float("nan"), "qc_recovery": None}) == {"sample_bias": 0.5}


def test_checks_name_library_faults_and_cover_every_fault(checks):
    listed = {k for c in checks for k in c.outcomes}
    assert listed <= set(LIBRARY)
    assert listed == set(LIBRARY), set(LIBRARY) - listed
    for c in checks:
        assert c.name and c.do
    assert separations("stock_strong", "istd_overspiked", checks)
    assert all(s.outcome_a != s.outcome_b for s in separations("stock_strong", "istd_overspiked", checks))
    assert not separations("stock_strong", "stock_strong", checks)


def test_confusion_table_documents_what_cannot_be_told_apart(table, checks):
    conf = confusion(table, checks)
    assert set(conf.fault) == set(LIBRARY)
    assert (conf["rank"] >= 1).all()
    assert (conf["rank"] == 1).mean() >= 0.5                  # most faults rank first on their own signature
    for r in conf.itertuples():
        if r._4:                                              # "confused with"
            assert r._5                                       # "separated by" names a check or says none is listed
