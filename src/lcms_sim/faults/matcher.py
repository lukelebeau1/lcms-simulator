"""Symptom matcher: which library faults are consistent with an observed feature vector.

Each fault's magnitude is swept (Fault.sweep) and the feature vector simulated with the noisy detector, averaged
over a few seeds; the table is computed once and bundled (scripts/make_fault_sweeps.py -> data/examples/
fault_sweeps.csv, with the baseline's mean and its spread over seeds, the replicate noise). An observation is any
subset of the features. Each fault is scored by its best distance to the observation over the features supplied,
with every difference divided by that feature's tolerance (the larger of a floor that matters at the bench and
twice the replicate noise), so a minute of retention and a ratio of bias count alike:

    distance(m) = sqrt( mean_f [ (feature_f(m) - observed_f) / tolerance_f ]^2 ),  features(m) interpolated
    between sweep points, best fit = argmin over m.

The result is a ranked list: fault, best-fit magnitude, which features sit within tolerance and which do not,
and, for the leading candidates, the bench check whose outcome differs between them (data/examples/checks.yaml).
"No fault in the library" is a candidate too, at the baseline. The matcher reports consistency, not
probability: faults within the tie margin of each other are reported as indistinguishable on the features
supplied, and a fault outside the library cannot be found at all.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lcms_sim.faults.library import LIBRARY, Fault, Scenario
from lcms_sim.faults.signatures import FEATURES, extract
from lcms_sim.io.loaders import InputSet, load_checks
from lcms_sim.io.schemas import SeparatingCheck
from lcms_sim.paths import EXAMPLES
from lcms_sim.rng import make_rng

BASELINE = "baseline"                 # the no-fault candidate's key in the table and the ranking
BASELINE_SD = "baseline_sd"           # the table row holding the replicate noise
SWEEP_POINTS = 9
SWEEP_SEEDS = (0, 1, 2)
SWEEP_FILE = EXAMPLES / "fault_sweeps.csv"
CHECKS_FILE = EXAMPLES / "checks.yaml"
TIE_MARGIN = 0.25                     # in tolerance units: closer than this and two faults are not told apart
FINE = 25                             # interpolation points per sweep interval

# What a difference has to exceed to mean something at the bench; the replicate noise raises it where larger.
TOLERANCE_FLOOR: dict[str, float] = {
    "sample_bias": 0.05, "sample_bias_spread": 0.02, "bias_rt_corr": 0.30, "qc_recovery": 0.05,
    "low_high_ratio": 0.10, "istd_area_ratio": 0.10, "istd_missing": 0.05, "istd_order_corr": 0.30,
    "rt_shift": 0.05, "rt_shift_spread": 0.03, "ion_ratio_fail_sample": 0.10, "ion_ratio_fail_cal": 0.05,
    "blank_positive": 0.10, "cal_low_error": 2.0, "cal_high_error": 2.0, "unreportable": 0.05,
}


# ---------------------------------------------------------------------------
# The sweep table
# ---------------------------------------------------------------------------


def extract_run(scenario: Scenario, seed: int) -> dict[str, float]:
    return extract(scenario.simulate(make_rng(seed)), scenario.inputs)


def _mean_sd(rows: Sequence[Mapping[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    mean, sd = {}, {}
    for k in FEATURES:
        v = np.array([r[k] for r in rows], float)
        v = v[np.isfinite(v)]
        mean[k] = float(v.mean()) if v.size else math.nan
        sd[k] = float(v.std()) if v.size > 1 else math.nan
    return mean, sd


def sweep_magnitudes(fault: Fault, n: int = SWEEP_POINTS) -> np.ndarray:
    """The fault's sweep grid without repeats (an on/off fault has one point)."""
    return np.unique(fault.sweep(n))


def sweep_table(inputs: InputSet, n: int = SWEEP_POINTS, seeds: Sequence[int] = SWEEP_SEEDS,
                faults: Mapping[str, Fault] | None = None,
                progress: Callable[[str, float], None] | None = None) -> pd.DataFrame:
    """Mean feature vector per fault and magnitude over the seeds, with the baseline's mean and spread."""
    base = Scenario.baseline(inputs)
    mean, sd = _mean_sd([extract_run(base, s) for s in seeds])
    rows = [{"fault": BASELINE, "magnitude": math.nan, **mean}, {"fault": BASELINE_SD, "magnitude": math.nan, **sd}]
    for key, f in (faults or LIBRARY).items():
        for m in sweep_magnitudes(f, n):
            mean, _ = _mean_sd([extract_run(f.apply(base, float(m)), s) for s in seeds])
            rows.append({"fault": key, "magnitude": float(m), **mean})
            if progress:
                progress(key, float(m))
    return pd.DataFrame(rows, columns=["fault", "magnitude", *FEATURES])


def load_sweep_table(path: str | Path = SWEEP_FILE) -> pd.DataFrame:
    table = pd.read_csv(path)
    missing = [c for c in ("fault", "magnitude", *FEATURES) if c not in table.columns]
    if missing:
        raise ValueError(f"{Path(path).name} lacks columns {missing}; rebuild it with scripts/make_fault_sweeps.py")
    return table


def tolerances(table: pd.DataFrame, floor: Mapping[str, float] = TOLERANCE_FLOOR) -> dict[str, float]:
    """Per feature, the larger of the bench floor and twice the baseline's replicate noise."""
    sd_rows = table[table.fault == BASELINE_SD]
    sd = sd_rows.iloc[0] if len(sd_rows) else {}
    out = {}
    for k in FEATURES:
        noise = float(sd[k]) if k in sd and np.isfinite(sd[k]) else 0.0
        out[k] = max(float(floor[k]), 2.0 * noise)
    return out


def default_checks(path: str | Path = CHECKS_FILE) -> list[SeparatingCheck]:
    return load_checks(path).checks


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    key: str
    name: str
    stage: str
    scope: str
    magnitude: float                 # best fit; nan for the baseline
    magnitude_label: str
    distance: float                  # in tolerance units; nan when no supplied feature could be compared
    matched: tuple[str, ...]         # supplied features within tolerance at the best fit
    unmatched: tuple[str, ...]       # supplied features outside tolerance
    simulated: dict[str, float]      # the feature vector at the best fit

    @property
    def consistent(self) -> bool:
        return bool(self.matched) and not self.unmatched


@dataclass(frozen=True)
class Separation:
    a: str
    b: str
    check: SeparatingCheck
    outcome_a: str
    outcome_b: str


@dataclass(frozen=True)
class MatchResult:
    observation: dict[str, float]
    tolerance: dict[str, float]
    ranked: list[Match]
    ties: list[tuple[str, str]]       # leading pairs closer than the tie margin: not told apart
    separations: list[Separation]     # for the leading pairs, the checks whose outcome differs

    def leader(self) -> Match:
        return self.ranked[0]

    def rank_of(self, key: str) -> int:
        return next(i for i, m in enumerate(self.ranked) if m.key == key) + 1

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"rank": i + 1, "fault": m.name, "key": m.key, "stage": m.stage, "scope": m.scope,
                              "magnitude": m.magnitude, "magnitude label": m.magnitude_label,
                              "distance": m.distance, "matched": ", ".join(m.matched),
                              "unmatched": ", ".join(m.unmatched)} for i, m in enumerate(self.ranked)])


def clean_observation(observation: Mapping[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in observation.items() if k in FEATURES and v is not None and np.isfinite(v)}


def _distance(sim: Mapping[str, float], obs: Mapping[str, float], tol: Mapping[str, float]) -> tuple[float, list, list]:
    z, matched, unmatched = [], [], []
    for k, o in obs.items():
        s = sim.get(k, math.nan)
        if not np.isfinite(s):
            continue
        d = (s - o) / tol[k]
        z.append(d * d)
        (matched if abs(d) <= 1.0 else unmatched).append(k)
    return (math.sqrt(float(np.mean(z))) if z else math.nan), matched, unmatched


def _grid(sub: pd.DataFrame, fine: int = FINE) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Magnitudes and interpolated feature vectors between the fault's sweep points."""
    sub = sub.sort_values("magnitude")
    ms = sub.magnitude.to_numpy(float)
    if len(ms) == 1:
        return ms, [dict(sub.iloc[0][list(FEATURES)])]
    grid = np.unique(np.concatenate([np.linspace(ms[i], ms[i + 1], fine, endpoint=False) for i in range(len(ms) - 1)]
                                    + [ms[-1:]]))
    vals = {}
    for k in FEATURES:
        y = sub[k].to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() == 0:
            vals[k] = np.full(grid.size, math.nan)
        elif ok.sum() == 1:
            vals[k] = np.where(np.isclose(grid, ms[ok][0]), y[ok][0], math.nan)
        else:
            vals[k] = np.interp(grid, ms[ok], y[ok], left=math.nan, right=math.nan)
            # keep nan where a bracketing sweep point is nan
            bad = ~ok
            for i in np.flatnonzero(bad):
                lo = ms[i - 1] if i > 0 else -math.inf
                hi = ms[i + 1] if i < len(ms) - 1 else math.inf
                vals[k][(grid > lo) & (grid < hi)] = math.nan
    return grid, [{k: float(vals[k][j]) for k in FEATURES} for j in range(grid.size)]


def match(observation: Mapping[str, float], table: pd.DataFrame, exclude: Sequence[str] = (),
          checks: Sequence[SeparatingCheck] | None = None, tie_margin: float = TIE_MARGIN,
          top: int = 3, tolerance: Mapping[str, float] | None = None) -> MatchResult:
    """Rank every library fault (and the baseline) by its best distance to the observation."""
    obs = clean_observation(observation)
    tol = dict(tolerance) if tolerance is not None else tolerances(table)
    out: list[Match] = []
    base = table[table.fault == BASELINE]
    if len(base):
        sim = {k: float(base.iloc[0][k]) for k in FEATURES}
        d, m, u = _distance(sim, obs, tol)
        out.append(Match(BASELINE, "No fault in the library", "", "", math.nan, "", d, tuple(m), tuple(u), sim))
    for key, f in LIBRARY.items():
        if key in exclude:
            continue
        sub = table[table.fault == key]
        if not len(sub):
            continue
        grid, vecs = _grid(sub)
        best = None
        for mag, sim in zip(grid, vecs):
            d, m, u = _distance(sim, obs, tol)
            if best is None or (np.isfinite(d) and (not np.isfinite(best[0]) or d < best[0])):
                best = (d, mag, m, u, sim)
        d, mag, m, u, sim = best
        out.append(Match(key, f.name, f.stage, f.scope, float(mag), f.magnitude_label, d, tuple(m), tuple(u), sim))
    out.sort(key=lambda x: (not np.isfinite(x.distance), x.distance if np.isfinite(x.distance) else 0.0))
    lead = [x for x in out[:top] if np.isfinite(x.distance)]
    ties = [(a.key, b.key) for i, a in enumerate(lead) for b in lead[i + 1:] if abs(a.distance - b.distance) <= tie_margin]
    seps: list[Separation] = []
    if checks:
        pairs = ties or [(a.key, b.key) for i, a in enumerate(lead) for b in lead[i + 1:]]
        for a, b in pairs:
            seps += separations(a, b, checks)
    return MatchResult(obs, tol, out, ties, seps)


def separations(a: str, b: str, checks: Sequence[SeparatingCheck]) -> list[Separation]:
    """The checks whose outcome differs between faults a and b (the baseline leaves every check unchanged)."""
    return [Separation(a, b, c, c.outcome(a), c.outcome(b)) for c in checks if c.separates(a, b)]


# ---------------------------------------------------------------------------
# The confusion table
# ---------------------------------------------------------------------------


def confusion(table: pd.DataFrame, checks: Sequence[SeparatingCheck] | None = None,
              tie_margin: float = TIE_MARGIN) -> pd.DataFrame:
    """Each fault, at its sweep point nearest its default magnitude, matched on every feature: its own rank, the
    faults within the tie margin of it, and the check that separates each. Documents where the results alone
    cannot tell faults apart and a bench experiment is required."""
    rows = []
    for key, f in LIBRARY.items():
        sub = table[table.fault == key]
        if not len(sub):
            continue
        row = sub.iloc[int((sub.magnitude - f.default).abs().argmin())]
        obs = {k: float(row[k]) for k in FEATURES if np.isfinite(row[k])}
        res = match(obs, table, checks=checks, tie_margin=tie_margin, top=len(LIBRARY) + 1)
        d_self = next(m.distance for m in res.ranked if m.key == key)
        confused = [m.key for m in res.ranked if m.key != key and np.isfinite(m.distance) and m.distance <= d_self + tie_margin]
        sep = []
        for other in confused:
            s = separations(key, other, checks) if checks else []
            sep.append(f"{other}: {s[0].check.name}" if s else f"{other}: no check listed")
        rows.append({"fault": key, "magnitude": float(row.magnitude), "rank": res.rank_of(key),
                     "confused with": ", ".join(confused), "separated by": "; ".join(sep)})
    return pd.DataFrame(rows)
