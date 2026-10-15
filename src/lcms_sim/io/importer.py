"""Import a lab's own files by column mapping: any quant batch export, any MRM transition table.

An export's content is fixed, whatever the vendor: which injection, of what kind, at what level, and per
compound the retention time, area, concentration and accuracy. Only the layout varies. `read_table` tells a
wide layout (MassHunter: two header rows, "<compound> Results" over a block of RT / Final Conc. / Accuracy
columns) from a long one (one row per injection and compound: Sciex, Thermo and Waters exports, or any
hand-made table) and turns both into one long frame. `guess_mapping` names the columns from a synonym list;
the user corrects it. `to_long` applies the mapping and fixes the column names and units.

From the long table: `summarise` (one row per compound), `observe` (the symptom vector the matcher takes, for
every feature the export can support, with the reason for each it cannot), and `transitions_from_table` for
an MRM table (validated against the Transition schema, every problem listed with its row). `make_lab_inputs`
builds a lab input folder from the bundled example as templates plus the imported transitions, with a README
naming which files are the lab's and which are still the example's. Nothing here touches the bundled data.

A MassHunter export may also carry qualifier blocks and each target's ISTD block; the wide reader attaches a
qualifier to the block before it and keeps each internal standard once per injection, as its own compound.
A name injected twice at different times becomes two injections.

Limits, stated where they bite: the first qualifier's ratio is kept on its row ("Qualifier Ratio"), but the
ion ratio features are not computed from it yet; comparing blanks with the lowest calibrator assumes one
concentration unit for every row; sample bias needs expected values (an accuracy column, or an expected
concentration) for the samples.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import ValidationError
from scipy.stats import spearmanr

from lcms_sim.io.loaders import FILES, InputSet, InputValidationError, Problem, load_input_set
from lcms_sim.io.schemas import Polarity, Transition, TransitionRole
from lcms_sim.paths import EXAMPLES, LAB

Layout = Literal["wide", "long"]
Kind = Literal["blank", "calibrator", "qc", "sample"]
KINDS: tuple[Kind, ...] = ("blank", "calibrator", "qc", "sample")

# --- the quant export's roles and the headers vendors use for them ---------------------------------------------

ROLES: dict[str, tuple[str, ...]] = {
    "injection": ("name", "sample", "sample name", "sample id", "injection", "vial", "data file", "filename",
                  "file name", "sample file"),
    "kind": ("type", "sample type", "kind", "sample kind"),
    "level": ("level", "cal level", "calibration level", "std level", "level name", "standard level"),
    "acquired": ("acq. date-time", "acq date-time", "acquired", "acquisition date", "acquisition time", "date",
                 "injection time", "acquired time"),
    "analyte": ("compound", "compound name", "analyte", "analyte name", "component", "component name", "target"),
    "rt_min": ("rt", "retention time", "ret. time", "ret time", "retention time (min)", "rt (min)", "ret. time (min)"),
    "area": ("area", "response", "peak area", "quant area", "resp.", "resp", "quantifier area"),
    "conc": ("final conc.", "final conc", "calculated concentration", "calc. conc.", "calc conc", "conc.", "conc",
             "concentration", "result", "calculated conc", "final concentration"),
    "accuracy_pct": ("accuracy", "accuracy (%)", "accuracy%", "% accuracy", "recovery", "recovery (%)", "% recovery",
                     "accuracy %"),
    "expected": ("exp. conc.", "exp conc", "expected concentration", "expected", "nominal", "actual concentration",
                 "nominal conc", "expected conc"),
}
REQUIRED = ("injection", "analyte")
NUMERIC = ("rt_min", "area", "conc", "accuracy_pct", "expected")
LONG_COLUMNS = ("injection", "order", "kind_raw", "kind", "level", "acquired", "analyte", "rt_min", "area", "conc",
                "accuracy_pct", "expected")

# What vendors call each kind of injection, and what a name says when the type column is missing or odd.
KIND_LABELS: dict[str, Kind] = {
    "blank": "blank", "double blank": "blank", "matrix blank": "blank", "solvent blank": "blank",
    "reagent blank": "blank", "blk": "blank",
    "cal": "calibrator", "calibration": "calibrator", "calibrator": "calibrator", "standard": "calibrator",
    "std": "calibrator", "cal std": "calibrator", "calibration standard": "calibrator",
    "qc": "qc", "quality control": "qc", "ccv": "qc", "icv": "qc", "lcs": "qc", "spike": "qc", "matrix spike": "qc",
    "sample": "sample", "unknown": "sample", "analyte": "sample", "unk": "sample",
}
NAME_HINTS: tuple[tuple[str, Kind], ...] = (
    ("blank", "blank"), ("rb", "blank"), ("mb", "blank"), ("cal", "calibrator"), ("std", "calibrator"),
    ("ccv", "qc"), ("icv", "qc"), ("lcs", "qc"), ("loq", "qc"), ("qc", "qc"), ("spike", "qc"), ("fs", "qc"),
)

ALIASES = (("imazilil", "imazalil"), ("imidicloprid", "imidacloprid"), ("ivermectrin", "ivermectin"),
           ("permethrins", "permethrin"))


def _key(name: str) -> str:
    s = re.sub(r"\s+", " ", str(name).strip().lower()).rstrip(":")
    return s[:-1] if s.endswith(".") else s


_SYNONYMS = {role: {_key(x) for x in names} for role, names in ROLES.items()}


def normalise_name(name: str) -> str:
    """Compound names as keys: case, spaces and punctuation dropped, the known misspellings mapped."""
    s = re.sub(r"[^a-z0-9]", "", str(name).lower())
    for a, b in ALIASES:
        s = s.replace(a, b)
    return s


# --- reading ---------------------------------------------------------------------------------------------------


@dataclass
class RawTable:
    layout: Layout
    frame: pd.DataFrame          # one row per injection and compound; the wide layout's compound is in "analyte"
    source: str = ""
    note: str = ""

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self.frame.columns]


def _decode(source) -> str:
    if isinstance(source, (str, Path)) and not (isinstance(source, str) and "\n" in source):
        return Path(source).read_text(encoding="utf-8-sig")
    if isinstance(source, bytes):
        return source.decode("utf-8-sig", errors="replace")
    if hasattr(source, "read"):
        data = source.read()
        return data.decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else str(data)
    return str(source)


def _delimiter(text: str) -> str:
    try:
        return csv.Sniffer().sniff(text[:4096], delimiters=",\t;").delimiter
    except csv.Error:
        return ","


def is_wide(rows: Sequence[Sequence[str]]) -> bool:
    """MassHunter's shape: compound names on the first row over repeated sub-columns on the second."""
    if len(rows) < 3:
        return False
    head, cols = rows[0], rows[1]
    if any(str(c).strip().endswith("Results") for c in head):
        return True
    blanks = sum(1 for c in head if not str(c).strip())
    repeated = len(cols) - len(set(cols))
    return blanks >= len(head) / 2 and repeated >= 2


_QUALIFIER_BLOCK = re.compile(r"^qualifier\s*\((.+?)\)\s*results$", re.IGNORECASE)
_ISTD_BLOCK = re.compile(r"^(.*?)\s*\(istd\)\s*results$", re.IGNORECASE)


def wide_to_long(rows: Sequence[Sequence[str]]) -> pd.DataFrame:
    """Two header rows -> one row per injection and compound, with the compound in column "analyte".

    A MassHunter export can carry, after each target's block, "Qualifier (a -> b) Results" blocks (a Ratio
    column) and the target's "<IS> (ISTD) Results" block, followed by the IS's own qualifier blocks. A qualifier
    belongs to the block before it; the first one's transition and ratio go on its owner's row ("Qualifier",
    "Qualifier Ratio"). An internal standard repeats under every target it serves, so it is kept once per
    injection, as a compound of its own."""
    head, cols, data = [str(c) for c in rows[0]], [str(c) for c in rows[1]], rows[2:]
    starts = [i for i, c in enumerate(head) if c.strip() and i > 0 and _key(c) != "sample"]
    if not starts:
        raise ValueError("no compound blocks found on the first header row")
    lead = cols[:starts[0]]
    blocks: list[tuple[int, int, str, str]] = []                  # start, end, kind, name
    for n, i in enumerate(starts):
        j = starts[n + 1] if n + 1 < len(starts) else len(cols)
        label = head[i].strip()
        if m := _QUALIFIER_BLOCK.match(label):
            blocks.append((i, j, "qualifier", re.sub(r"\s*->\s*", " > ", m.group(1).strip())))
        elif m := _ISTD_BLOCK.match(label):
            blocks.append((i, j, "istd", m.group(1).strip()))
        else:
            blocks.append((i, j, "target", label.removesuffix("Results").strip()))
    subs: list[str] = []
    for i, j, kind, _ in blocks:
        if kind != "qualifier":
            subs.extend(s for s in cols[i:j] if s not in subs)
    extra = ["Qualifier", "Qualifier Ratio"] if any(b[2] == "qualifier" for b in blocks) else []
    out = []
    for r in data:
        r = list(r) + [""] * (len(cols) - len(r))
        base = {c: r[k] for k, c in enumerate(lead)}
        seen_istd: set[str] = set()
        owner: dict | None = None                                  # the row a qualifier block belongs to
        for i, j, kind, name in blocks:
            if kind == "qualifier":
                if owner is not None and not owner.get("Qualifier"):
                    ratio = next((r[k] for k in range(i, j) if _key(cols[k]) == "ratio"), r[i])
                    owner["Qualifier"], owner["Qualifier Ratio"] = name, ratio
                continue
            if kind == "istd":
                if name in seen_istd:
                    owner = None                                   # already kept, with its own qualifier
                    continue
                seen_istd.add(name)
            row = dict(base, analyte=name)
            for k in range(i, j):
                row[cols[k]] = r[k]
            out.append(row)
            owner = row
    return pd.DataFrame(out, columns=[*lead, "analyte", *subs, *extra]).fillna("")


def read_table(source, name: str = "") -> RawTable:
    """A CSV or tab-separated export, wide or long, as a long frame of strings."""
    text = _decode(source)
    delim = _delimiter(text)
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delim) if any(str(c).strip() for c in r)]
    if is_wide(rows):
        frame = wide_to_long(rows)
        return RawTable("wide", frame, name, f"{len(rows) - 2} injections x {frame.analyte.nunique()} compounds, "
                                              "compound blocks on the first header row")
    frame = pd.DataFrame(rows[1:], columns=[str(c) for c in rows[0]]).fillna("") if rows else pd.DataFrame()
    return RawTable("long", frame, name, f"{len(frame)} rows, one per injection and compound")


# --- mapping ---------------------------------------------------------------------------------------------------


def guess_mapping(columns: Sequence[str], roles: Mapping[str, Iterable[str]] | None = None) -> dict[str, str]:
    """role -> column, by header synonyms; a column serves one role."""
    syn = _SYNONYMS if roles is None else {r: {_key(x) for x in names} for r, names in roles.items()}
    out: dict[str, str] = {}
    used: set[str] = set()
    for role, names in syn.items():
        for c in columns:
            if c not in used and _key(c) in names:
                out[role] = c
                used.add(c)
                break
    return out


def guess_kind(raw: str, injection: str = "") -> Kind:
    k = _key(raw)
    if k in KIND_LABELS:
        return KIND_LABELS[k]
    for label, kind in KIND_LABELS.items():
        if k and label in k:
            return kind
    name = _key(injection)
    for hint, kind in NAME_HINTS:
        if name.startswith(hint) or f" {hint}" in name or f"_{hint}" in name or f"-{hint}" in name:
            return kind
    return "sample"


def to_long(frame: pd.DataFrame, mapping: Mapping[str, str], kind_map: Mapping[str, Kind] | None = None) -> pd.DataFrame:
    """The mapped export as one long table with fixed column names (LONG_COLUMNS)."""
    missing = [r for r in REQUIRED if r not in mapping or mapping[r] not in frame.columns]
    if missing:
        raise ValueError(f"the mapping needs a column for: {', '.join(missing)}")
    out = pd.DataFrame()
    out["injection"] = frame[mapping["injection"]].astype(str).str.strip()
    out["kind_raw"] = frame[mapping["kind"]].astype(str).str.strip() if mapping.get("kind") in frame.columns else ""
    out["level"] = frame[mapping["level"]].astype(str).str.strip() if mapping.get("level") in frame.columns else ""
    out["acquired"] = frame[mapping["acquired"]].astype(str).str.strip() if mapping.get("acquired") in frame.columns else ""
    out["analyte"] = frame[mapping["analyte"]].astype(str).str.strip()
    for col in NUMERIC:
        src = mapping.get(col)
        out[col] = pd.to_numeric(frame[src].astype(str).str.replace(",", ""), errors="coerce") if src in frame.columns else np.nan
    out = out[(out.injection != "") & (out.analyte != "")].reset_index(drop=True)
    if (out.acquired != "").any():
        # The same name injected again later (a reagent blank repeated in the run) is another injection:
        # the acquisition time tells them apart; the second gets " (2)" and so on.
        seen: dict[str, int] = {}
        rename: dict[tuple[str, str], str] = {}
        for inj, acq in dict.fromkeys(zip(out.injection, out.acquired)):
            seen[inj] = seen.get(inj, 0) + 1
            rename[(inj, acq)] = inj if seen[inj] == 1 else f"{inj} ({seen[inj]})"
        out["injection"] = [rename[(i, a)] for i, a in zip(out.injection, out.acquired)]
    kind_map = {_key(k): v for k, v in (kind_map or {}).items()}
    out["kind"] = [kind_map.get(_key(k)) or guess_kind(k, inj) for k, inj in zip(out.kind_raw, out.injection)]
    order = {inj: n for n, inj in enumerate(dict.fromkeys(out.injection))}       # file order of first appearance
    out["order"] = out.injection.map(order)
    cal = out.kind == "calibrator"
    if cal.any() and (out.loc[cal, "level"] == "").all() and out.loc[cal, "expected"].notna().any():
        # No level column (Sciex, Waters): the calibrators' expected concentrations rank the levels per compound.
        ranks = out[cal].groupby("analyte").expected.rank(method="dense")
        out.loc[cal, "level"] = ranks.map(lambda r: str(int(r)) if r == r else "")
    return out[list(LONG_COLUMNS)]


def kind_table(long: pd.DataFrame) -> pd.DataFrame:
    """One row per raw type label: the kind it was read as and how many injections carry it (for the editor)."""
    rows = []
    for raw, g in long.groupby("kind_raw", sort=False):
        rows.append({"label in the export": raw or "(blank)", "kind": g.kind.mode().iloc[0],
                     "injections": g.injection.nunique(), "example": g.injection.iloc[0]})
    return pd.DataFrame(rows)


def match_names(export_names: Iterable[str], method_names: Iterable[str]) -> dict[str, str | None]:
    """export compound -> method analyte with the same normalised name, None when the method lacks it."""
    by_norm = {normalise_name(m): m for m in method_names}
    return {e: by_norm.get(normalise_name(e)) for e in dict.fromkeys(export_names)}


def with_method_names(long: pd.DataFrame, names: Mapping[str, str | None]) -> pd.DataFrame:
    """The long table with analytes renamed to the method's names where matched; unmatched rows keep theirs."""
    out = long.copy()
    out["analyte"] = [names.get(a) or a for a in out.analyte]
    return out


# --- summaries -------------------------------------------------------------------------------------------------


def level_order(long: pd.DataFrame) -> list[str]:
    """Calibrator levels from low to high: numeric when they parse, else by their mean concentration."""
    cal = long[long.kind == "calibrator"]
    levels = [lv for lv in dict.fromkeys(cal.level) if lv != ""]
    try:
        return sorted(levels, key=float)
    except ValueError:
        means = {lv: cal[cal.level == lv].conc.mean() for lv in levels}
        return sorted(levels, key=lambda lv: (math.isnan(means[lv]), means[lv]))


def _ratio(g: pd.DataFrame) -> pd.Series:
    """reported / expected per row: conc / expected where both exist, else accuracy / 100."""
    r = g.conc / g.expected.where(g.expected > 0)
    return r.where(np.isfinite(r), g.accuracy_pct / 100.0)


def summarise(long: pd.DataFrame) -> pd.DataFrame:
    """One row per compound: retention, calibration accuracy, QC accuracy, blanks, the lowest calibrator."""
    levels = level_order(long)
    rows = []
    for name, g in long.groupby("analyte", sort=False):
        hit = g[g.kind.isin(["calibrator", "qc"])]
        if hit.accuracy_pct.notna().any():
            hit = hit[hit.accuracy_pct > 50]
        hit = hit[hit.rt_min.notna()]
        cal, qc, blank, smp = (g[g.kind == k] for k in ("calibrator", "qc", "blank", "sample"))
        acc_qc = _ratio(qc) * 100
        row = {"analyte": name, "rt_median_min": hit.rt_min.median(), "rt_sd_min": hit.rt_min.std(), "rt_n": len(hit),
               "cal_accuracy_min_pct": (_ratio(cal) * 100).min(), "cal_accuracy_max_pct": (_ratio(cal) * 100).max(),
               "cal_points_outside_80_120": int(((_ratio(cal) * 100 < 80) | (_ratio(cal) * 100 > 120)).sum()),
               "qc_accuracy_mean_pct": acc_qc.mean(),
               "qc_accuracy_rsd_pct": 100 * acc_qc.std() / acc_qc.mean() if acc_qc.notna().sum() > 1 and acc_qc.mean() else np.nan,
               "qc_n": int(acc_qc.notna().sum()), "sample_n": smp.injection.nunique(),
               "blank_conc_median": blank.conc.median(), "blank_conc_max": blank.conc.max(),
               "cal_low_conc": cal[cal.level == levels[0]].conc.median() if levels else np.nan,
               "cal_area_mean": cal.area.mean()}
        rows.append(row)
    return pd.DataFrame(rows)


# --- the symptom vector ----------------------------------------------------------------------------------------


@dataclass
class Observation:
    features: dict[str, float] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)       # feature -> how it was read, or why it could not be
    names: dict[str, str | None] = field(default_factory=dict)  # export compound -> method analyte

    def frame(self) -> pd.DataFrame:
        from lcms_sim.faults.signatures import FEATURES
        return pd.DataFrame([{"feature": k, "value": self.features.get(k, np.nan), "note": self.notes.get(k, "")}
                             for k in FEATURES])


def _median(x) -> float:
    a = np.array([v for v in x if v is not None and np.isfinite(v)], float)
    return float(np.median(a)) if a.size else math.nan


def _robust_sd(x) -> float:
    a = np.array([v for v in x if v is not None and np.isfinite(v)], float)
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if a.size > 2 else math.nan


def _spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return math.nan
    return float(spearmanr(x[ok], y[ok]).statistic)


def observe(long: pd.DataFrame, method: InputSet | None = None) -> Observation:
    """The feature vector of a batch, for every feature the export supports; a note says how, or why not."""
    obs = Observation()
    f, notes = obs.features, obs.notes
    targets_m = [a.name for a in method.analytes if not a.is_internal_standard] if method else []
    istds_m = [a.name for a in method.analytes if a.is_internal_standard] if method else []
    obs.names = match_names(long.analyte.unique(), targets_m + istds_m) if method else {a: None for a in long.analyte.unique()}
    is_names = {e for e, m in obs.names.items() if m in istds_m}
    targets = [a for a in long.analyte.unique() if a not in is_names]
    has_area, has_ratio = long.area.notna().any(), (long.accuracy_pct.notna().any() or long.expected.notna().any())

    # bias in samples, across compounds
    smp = long[(long.kind == "sample") & long.analyte.isin(targets)]
    per = {a: _median(_ratio(g)) for a, g in smp.groupby("analyte")} if has_ratio else {}
    per = {a: v for a, v in per.items() if np.isfinite(v) and v > 0}
    if per:
        f["sample_bias"] = _median(per.values())
        logs = {a: math.log10(v) for a, v in per.items()}
        notes["sample_bias"] = f"median reported / expected over {len(per)} compounds in {smp.injection.nunique()} samples"
        f["sample_bias_spread"] = _robust_sd(logs.values())
        notes["sample_bias_spread"] = "robust spread of log10(reported / expected) across those compounds"
        rt = {a: g.rt_min.median() for a, g in long[long.kind.isin(["calibrator", "qc", "sample"])].groupby("analyte")}
        f["bias_rt_corr"] = _spearman([rt.get(a, math.nan) for a in logs], list(logs.values()))
        notes["bias_rt_corr"] = "Spearman of log bias with the compound's median retention time"
    else:
        why = ("no sample-kind injections" if smp.empty else
               "samples carry no expected concentration or accuracy in this export (a PT sample needs its assigned values)")
        for k in ("sample_bias", "sample_bias_spread", "bias_rt_corr"):
            notes[k] = why
    for k in ("sample_bias_spread", "bias_rt_corr"):
        if k in f and not np.isfinite(f[k]):
            del f[k]
            notes[k] = ("too few compounds with a sample bias (needs 3, or 4 for the correlation)" if len(per) < 4
                        else "undefined: the bias (or the retention time) is the same for every compound")

    # QCs and calibrators
    qc = long[(long.kind == "qc") & long.analyte.isin(targets)]
    r_qc = _ratio(qc) if has_ratio else pd.Series(dtype=float)
    if r_qc.notna().any():
        f["qc_recovery"] = _median(r_qc)
        notes["qc_recovery"] = f"median reported / expected over {int(r_qc.notna().sum())} QC results"
    else:
        notes["qc_recovery"] = "no QC injections with an expected value"
    loq = qc[qc.injection.str.contains("loq", case=False) | qc.level.str.contains("loq", case=False)]
    ccv = qc[qc.injection.str.contains("ccv", case=False) | qc.level.str.contains("ccv", case=False)]
    if has_ratio and not loq.empty and not ccv.empty:
        f["low_high_ratio"] = _median(_ratio(loq)) / _median(_ratio(ccv))
        notes["low_high_ratio"] = "median LOQ recovery over median CCV recovery, found by name"
    else:
        notes["low_high_ratio"] = "needs QC injections named or levelled LOQ and CCV"
    levels = level_order(long)
    cal = long[(long.kind == "calibrator") & long.analyte.isin(targets)]
    if has_ratio and len(levels) >= 2 and cal.accuracy_pct.notna().any() | cal.expected.notna().any():
        lo, hi = cal[cal.level == levels[0]], cal[cal.level == levels[-1]]
        f["cal_low_error"] = _median((_ratio(lo) * 100 - 100).abs())
        f["cal_high_error"] = _median((_ratio(hi) * 100 - 100).abs())
        notes["cal_low_error"] = f"median |accuracy - 100| at level {levels[0]}"
        notes["cal_high_error"] = f"median |accuracy - 100| at level {levels[-1]}"
    else:
        notes["cal_low_error"] = notes["cal_high_error"] = "needs calibrators with an accuracy or expected value at two or more levels"

    # retention against the method
    if method:
        method_rt = {t.analyte: t.retention_time_min for t in method.transitions if t.role is TransitionRole.quantifier}
        hits = long[long.kind.isin(["calibrator", "qc"]) & long.rt_min.notna()]
        if hits.accuracy_pct.notna().any():
            hits = hits[hits.accuracy_pct > 50]
        shifts = []
        for e, g in hits.groupby("analyte"):
            m = obs.names.get(e)
            if m in method_rt:
                shifts.append(g.rt_min.median() - method_rt[m])
        if shifts:
            f["rt_shift"] = _median(shifts)
            notes["rt_shift"] = f"median (batch - method) retention time over {len(shifts)} matched compounds"
            if len(shifts) > 2:
                f["rt_shift_spread"] = _robust_sd(shifts)
                notes["rt_shift_spread"] = "robust spread of that shift across compounds"
            else:
                notes["rt_shift_spread"] = "fewer than 3 matched compounds"
        else:
            notes["rt_shift"] = notes["rt_shift_spread"] = "no export compound matches a method analyte by name"
    else:
        notes["rt_shift"] = notes["rt_shift_spread"] = "no method loaded to compare with"

    # internal standards
    if method and has_area and is_names:
        cal_i = long[long.kind == "calibrator"]
        non = long[long.kind.isin(["sample", "qc"])]
        ratios, missing = [], 0
        for e in is_names:
            c = cal_i[cal_i.analyte == e].area.mean()
            n = non[non.analyte == e].area.mean()
            if not np.isfinite(c) or c <= 0:
                missing += 1
                continue
            if np.isfinite(n):
                ratios.append(n / c)
        if ratios:
            f["istd_area_ratio"] = _median(ratios)
            notes["istd_area_ratio"] = f"median over {len(ratios)} IS of mean area in samples and QCs over mean in calibrators"
        else:
            notes["istd_area_ratio"] = "no IS has areas in both calibrators and samples"
        f["istd_missing"] = missing / len(is_names)
        notes["istd_missing"] = f"{missing} of {len(is_names)} IS with no area in the calibrators"
        with_is = long[long.analyte.isin(is_names) & (long.area > 0)]
        if with_is.injection.nunique() > 3:
            mean = with_is.groupby("analyte").area.mean()
            norm = with_is.assign(norm=with_is.area / with_is.analyte.map(mean)).groupby("order").norm.median()
            f["istd_order_corr"] = _spearman(norm.index.to_numpy(), norm.to_numpy())
            notes["istd_order_corr"] = "Spearman of the IS area (normalised per IS) with the injection's order in the file"
        else:
            notes["istd_order_corr"] = "fewer than 4 injections with IS areas"
    else:
        why = ("no method loaded" if not method else "the export has no area column" if not has_area
               else "no export compound matches a method internal standard by name")
        for k in ("istd_area_ratio", "istd_missing", "istd_order_corr"):
            notes[k] = why
    if "istd_order_corr" in f and not np.isfinite(f["istd_order_corr"]):
        del f["istd_order_corr"]
        notes["istd_order_corr"] = "IS areas do not vary enough to rank"

    # blanks, against the lowest calibrator in the export's own units
    blanks = long[(long.kind == "blank") & (long.order > 0) & long.analyte.isin(targets)]
    if levels and not blanks.empty and cal.conc.notna().any():
        low = cal[cal.level == levels[0]].groupby("analyte").conc.median()
        pos = {a for a, g in blanks.groupby("analyte") if a in low and np.isfinite(low[a]) and (g.conc > low[a]).any()}
        f["blank_positive"] = len(pos) / len(targets) if targets else math.nan
        notes["blank_positive"] = (f"{len(pos)} of {len(targets)} compounds above the level-{levels[0]} calibrator's "
                                   "concentration in a blank after the first injection (one unit for every row assumed)")
    else:
        notes["blank_positive"] = "needs blanks after the first injection and calibrator concentrations"

    smp_all = long[(long.kind == "sample") & long.analyte.isin(targets)]
    if not smp_all.empty:
        f["unreportable"] = float(smp_all.conc.isna().mean())
        notes["unreportable"] = f"share of {len(smp_all)} sample results with no concentration"
    else:
        notes["unreportable"] = "no sample-kind injections"
    for k in ("ion_ratio_fail_sample", "ion_ratio_fail_cal"):
        notes[k] = ("not computed from an export yet (qualifier ratios, when the export has them, are kept on "
                    "each row); the matcher ignores this feature")
    for k in ("sample_bias", "qc_recovery", "low_high_ratio", "cal_low_error", "cal_high_error", "rt_shift",
              "istd_area_ratio", "blank_positive", "unreportable"):
        if k in f and not np.isfinite(f[k]):
            del f[k]
            notes[k] = "could not be computed from the rows present"
    return obs


# --- transition tables -----------------------------------------------------------------------------------------

TRANSITION_ROLES: dict[str, tuple[str, ...]] = {
    "analyte": ("compound name", "compound", "analyte", "name", "component name", "component", "analyte name"),
    "precursor_mz": ("precursor ion", "precursor", "precursor m/z", "q1", "q1 mass", "q1 m/z", "parent", "parent ion",
                     "ms1", "precursor (m/z)"),
    "product_mz": ("product ion", "product", "product m/z", "q3", "q3 mass", "q3 m/z", "fragment", "daughter", "ms2",
                   "product (m/z)"),
    "retention_time_min": ("ret time (min)", "ret time", "ret. time", "rt", "retention time", "expected rt", "rt (min)",
                           "retention time (min)"),
    "window_min": ("delta ret time", "delta rt", "rt window", "window", "rt tolerance", "ret time window",
                   "delta ret time (min)", "window (min)"),
    "collision_energy_v": ("collision energy", "ce", "ce (v)", "collision energy (v)", "ce (ev)", "collision energy (ev)"),
    "fragmentor_v": ("fragmentor", "fragmentor (v)", "dp", "declustering potential", "cone", "cone voltage"),
    "cell_accel_v": ("cell accelerator voltage", "cell accel", "cell accelerator", "cxp", "collision cell exit potential"),
    "polarity": ("polarity", "ion polarity", "ion mode", "mode"),
    "role": ("role", "quant/qual", "ion type", "transition type", "quantifier", "type"),
    "is_istd": ("istd?", "istd", "is", "internal standard", "is istd", "istd flag"),
    "dwell_ms": ("dwell", "dwell time", "dwell (ms)", "dwell time (ms)"),
}
TRANSITION_REQUIRED = ("analyte", "precursor_mz", "product_mz", "retention_time_min")


@dataclass
class TransitionImport:
    transitions: list[Transition]
    frame: pd.DataFrame                     # the rows as read, in the schema's column names
    problems: list[Problem]
    notes: list[str]

    def ok(self) -> bool:
        return not self.problems


def _polarity(v: str) -> Polarity:
    k = _key(v)
    return Polarity.negative if k.startswith("neg") or k == "-" else Polarity.positive


def _role(v: str) -> TransitionRole | None:
    k = _key(v)
    if k.startswith("quant") or k in ("target", "primary", "q"):
        return TransitionRole.quantifier
    if k.startswith("qual") or k in ("confirm", "confirming", "secondary"):
        return TransitionRole.qualifier
    return None


def transitions_from_table(frame: pd.DataFrame, mapping: Mapping[str, str], default_window_min: float = 1.0,
                           names: Mapping[str, str | None] | None = None, file: str = "transitions") -> TransitionImport:
    """Rows of an MRM table -> Transition models. Role from a role column, else the first product of each
    compound is the quantifier. Qualifiers get no ion ratio (unknown until measured). Problems carry the row."""
    problems: list[Problem] = []
    notes: list[str] = []
    missing = [r for r in TRANSITION_REQUIRED if mapping.get(r) not in frame.columns]
    if missing:
        raise ValueError(f"the mapping needs a column for: {', '.join(missing)}")

    def col(role):
        c = mapping.get(role)
        return frame[c].astype(str).str.strip() if c in frame.columns else None

    def num(role):
        s = col(role)
        return pd.to_numeric(s.str.replace(",", ""), errors="coerce") if s is not None else None

    analyte = col("analyte")
    if names:
        analyte = analyte.map(lambda a: names.get(a) or a)
    rows = pd.DataFrame({"analyte": analyte, "precursor_mz": num("precursor_mz"), "product_mz": num("product_mz"),
                         "retention_time_min": num("retention_time_min")})
    win = num("window_min")
    if win is None or win.isna().all():
        rows["window_min"] = default_window_min
        notes.append(f"no window column: every window set to {default_window_min:g} min")
    else:
        rows["window_min"] = win.fillna(default_window_min)
    for role in ("collision_energy_v", "fragmentor_v", "cell_accel_v"):
        v = num(role)
        rows[role] = v if v is not None else np.nan
    pol = col("polarity")
    rows["polarity"] = [_polarity(p).value for p in pol] if pol is not None else Polarity.positive.value
    if pol is None:
        notes.append("no polarity column: positive assumed")
    r = col("role")
    roles = [_role(v) for v in r] if r is not None else [None] * len(rows)
    first_seen: set[str] = set()
    out_roles = []
    for a, ro in zip(rows.analyte, roles):
        if ro is None:
            ro = TransitionRole.qualifier if a in first_seen else TransitionRole.quantifier
        first_seen.add(a)
        out_roles.append(ro.value)
    rows["role"] = out_roles
    if r is None:
        notes.append("no role column: the first product of each compound is its quantifier, the rest qualifiers")
    rows["relative_response"] = [1.0 if ro == "quantifier" else np.nan for ro in rows.role]
    if (rows.role == "qualifier").any():
        notes.append("qualifier ion ratios are unknown until measured from a standard (relative_response left empty)")
    rows = rows[rows.analyte != ""].reset_index(drop=True)
    trs: list[Transition] = []
    for k, row in rows.iterrows():
        data = {c: (None if (isinstance(v, float) and math.isnan(v)) else v) for c, v in row.items()}
        try:
            trs.append(Transition.model_validate(data))
        except ValidationError as e:
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"]) or None
                problems.append(Problem(file, int(k) + 2, loc, err["msg"]))
    quant_per = rows[rows.role == "quantifier"].groupby("analyte").size()
    for a, n in quant_per.items():
        if n > 1:
            problems.append(Problem(file, None, "role", f"{a}: {n} quantifiers; one per compound"))
    return TransitionImport(trs, rows, problems, notes)


def transitions_frame(trs: Sequence[Transition]) -> pd.DataFrame:
    """Transitions as the simulator's transitions.csv columns."""
    cols = list(Transition.model_fields)
    return pd.DataFrame([{c: (getattr(t, c).value if hasattr(getattr(t, c), "value") else getattr(t, c)) for c in cols}
                         for t in trs], columns=cols)


# --- saving and the lab input set ------------------------------------------------------------------------------


def save_import(folder: Path, long: pd.DataFrame | None = None, summary: pd.DataFrame | None = None,
                observation: Observation | None = None, transitions: pd.DataFrame | None = None) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    if long is not None:
        long.to_csv(folder / "long.csv", index=False)
        written.append(folder / "long.csv")
    if summary is not None:
        summary.to_csv(folder / "summary.csv", index=False)
        written.append(folder / "summary.csv")
    if observation is not None:
        (folder / "observation.json").write_text(json.dumps({"features": observation.features, "notes": observation.notes,
                                                             "names": observation.names}, indent=2), encoding="utf-8")
        written.append(folder / "observation.json")
    if transitions is not None:
        transitions.to_csv(folder / "transitions.csv", index=False)
        written.append(folder / "transitions.csv")
    return written


EXTRA_FILES = ("prep_recipe.yaml", "containers.yaml", "checks.yaml")


def field_notes(model) -> list[str]:
    """One line per field of a schema: name, type, description, for the templates' README."""
    lines = []
    for name, info in model.model_fields.items():
        kind = getattr(info.annotation, "__name__", str(info.annotation)).replace("typing.", "")
        desc = info.description or ""
        req = "" if info.is_required() else f" (default {info.default!r})" if info.default is not None else " (optional)"
        lines.append(f"- `{name}`: {kind}{req}. {desc}".rstrip())
    return lines


def make_lab_inputs(dest: Path, examples: Path = EXAMPLES, transitions: pd.DataFrame | None = None,
                    overwrite_transitions: bool = True) -> list[str]:
    """A lab input folder: every file the loader reads, copied from the example as a template unless already
    there, the imported transitions written over transitions.csv, and a README naming which is which."""
    dest, examples = Path(dest), Path(examples)
    dest.mkdir(parents=True, exist_ok=True)
    status = []
    for key, (name, loader) in FILES.items():
        target = dest / name
        if name == "transitions.csv" and transitions is not None and (overwrite_transitions or not target.exists()):
            transitions.to_csv(target, index=False)
            status.append(f"{name}: from the imported transition table")
        elif target.exists():
            status.append(f"{name}: already here (kept)")
        else:
            shutil.copy(examples / name, target)
            status.append(f"{name}: copied from the bundled example as a template; replace with the lab's values")
    for name in EXTRA_FILES:
        if not (dest / name).exists() and (examples / name).exists():
            shutil.copy(examples / name, dest / name)
            status.append(f"{name}: copied from the bundled example")
    from lcms_sim.io import schemas as S
    models = {"system.yaml": S.SystemConstants, "lc_method.yaml": S.LCMethod, "acquisition.yaml": S.AcquisitionMethod,
              "analytes.csv": S.Analyte, "transitions.csv": S.Transition, "prep.yaml": S.PrepProcedure,
              "calibration.yaml": S.CalibrationSettings, "samples.yaml": S.SampleDefinition,
              "infusion_matrix.csv": S.InfusionPoint}
    readme = ["# Lab input set", "", "Built by the importer. Each file the simulator reads, and where it came from:", ""]
    readme += [f"- {s}" for s in status]
    readme += ["", "Edit the templates to the lab's values, then validate on the Import page; every problem is "
               "listed with its file, row and column. Units are in the field names (docs/io.md).", "",
               "## Fields", ""]
    for name, model in models.items():
        readme += [f"### {name} ({model.__name__})", ""] + field_notes(model) + [""]
    (dest / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return status


def validate_inputs(folder: Path) -> list[Problem]:
    try:
        load_input_set(folder)
    except InputValidationError as e:
        return list(e.problems)
    return []


def lab_folder(tag: str) -> Path:
    """data/lab/<tag>, with the tag reduced to letters, digits, _ . - and no leading or trailing dot, so a name such
    as '..' cannot leave data/lab."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag.strip()).strip(".") or "batch"
    return LAB / safe
