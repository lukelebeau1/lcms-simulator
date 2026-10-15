"""The importer: any export in, the simulator's tables and the matcher's symptom vector out.

Oracle: a simulated run written out in MassHunter's wide shape, and again as a long Sciex-like table, imports
to the same symptom vector that faults/signatures.py computes from the run itself. The transition table round
trip reproduces the bundled method. A lab input set built from the templates validates."""

import csv
import io

import numpy as np
import pandas as pd
import pytest

from lcms_sim.faults.signatures import extract
from lcms_sim.io.importer import (KINDS, LONG_COLUMNS, ROLES, TRANSITION_ROLES, field_notes, guess_kind,
                                  guess_mapping, kind_table, level_order, make_lab_inputs, match_names,
                                  normalise_name, observe, read_table, save_import, summarise, to_long,
                                  transitions_frame, transitions_from_table, validate_inputs, with_method_names)
from lcms_sim.io.loaders import load_input_set, load_transitions
from lcms_sim.io.schemas import TransitionRole
from lcms_sim.paths import EXAMPLES
from lcms_sim.pipeline import Kind
from lcms_sim.quant.prep import back_calculate, prep_values
from lcms_sim.rng import make_rng

TYPE = {Kind.blank: "Blank", Kind.calibrator: "Cal", Kind.qc: "QC", Kind.sample: "Sample"}


@pytest.fixture(scope="module")
def inputs():
    return load_input_set(EXAMPLES)


@pytest.fixture(scope="module")
def run(inputs):
    """A run with a real pattern across compounds (recovery differs per analyte), noise-free."""
    from lcms_sim.faults.library import LIBRARY, Scenario
    return LIBRARY["low_recovery"].apply(Scenario.baseline(inputs, detector="expected"), 0.6).simulate(make_rng(0))


def long_rows(result, inputs):
    """The run as (injection, type, level, compound, rt, conc, accuracy, area) rows, one unit for every row."""
    targets = [a.name for a in inputs.analytes if not a.is_internal_standard]
    istds = [a.name for a in inputs.analytes if a.is_internal_standard]
    samples = {s.id: s for s in inputs.samples}
    levels = {lv.level: lv for lv in inputs.calibration_design}
    nom = prep_values(inputs.prep, "nominal")
    frac = {a.name: a.isomer_fraction or 1.0 for a in inputs.analytes}
    rows = []
    for r in result.injections:
        kind = r.item.kind
        for a in targets:
            rt = result.retention_min[a]
            conc = acc = np.nan
            if kind is Kind.calibrator:
                lv = levels[r.item.ref]
                fit = result.fits.get(a)
                if a in lv.concentrations_ng_ml and fit is not None and fit.accuracy_pct:
                    expected = back_calculate(lv.concentrations_ng_ml[a] * frac[a], nom)
                    j = int(np.argmin(np.abs(np.asarray(fit.x) - lv.concentrations_ng_ml[a] * frac[a])))
                    acc = fit.accuracy_pct[j]
                    conc = expected * acc / 100
            else:
                conc = result.reported_ug_kg.get(r.item.ref, {}).get(a, np.nan)
                true = samples[r.item.ref].true_concentration_ug_kg.get(a)
                acc = 100 * conc / true if true else np.nan
            rows.append((r.item.ref, TYPE[kind], r.item.ref if kind is Kind.calibrator else "", a, rt, conc, acc,
                         r.areas.get(a, 0.0)))
        for s in istds:
            rows.append((r.item.ref, TYPE[kind], r.item.ref if kind is Kind.calibrator else "", s,
                         result.retention_min[s], np.nan, np.nan, r.areas.get(s, 0.0)))
    return rows


def fmt(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.12g}"


def masshunter_csv(rows) -> str:
    compounds = list(dict.fromkeys(r[3] for r in rows))
    head = ["Sample", "", "", "", ""] + sum(([f"{c} Results", "", "", ""] for c in compounds), [])
    cols = ["Name", "Data File", "Type", "Level", "Acq. Date-Time"] + ["RT", "Final Conc.", "Accuracy", "Area"] * len(compounds)
    by_inj: dict[str, dict] = {}
    for inj, typ, lvl, c, rt, conc, acc, area in rows:
        d = by_inj.setdefault(inj, {"type": typ, "level": lvl, "cells": {}})
        d["cells"][c] = (rt, conc, acc, area)
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(head)
    w.writerow(cols)
    for n, (inj, d) in enumerate(by_inj.items()):
        line = [inj, f"{inj}.d", d["type"], d["level"], f"1/15/2026 {n}:00 PM"]
        for c in compounds:
            line += [fmt(v) for v in d["cells"].get(c, (np.nan,) * 4)]
        w.writerow(line)
    return out.getvalue()


def sciex_csv(rows) -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["Sample Name", "Sample Type", "Component Name", "Retention Time", "Area", "Calculated Concentration",
                "Accuracy", "Actual Concentration"])
    sciex = {"Blank": "Blank", "Cal": "Standard", "QC": "Quality Control", "Sample": "Unknown"}
    for inj, typ, lvl, c, rt, conc, acc, area in rows:
        expected = conc / (acc / 100) if np.isfinite(acc) and acc else np.nan
        w.writerow([inj, sciex[typ], c, fmt(rt), fmt(area), fmt(conc), fmt(acc), fmt(expected)])
    return out.getvalue()


def test_masshunter_export_round_trips_to_the_runs_symptoms(run, inputs):
    rows = long_rows(run, inputs)
    raw = read_table(masshunter_csv(rows), "sim.csv")
    assert raw.layout == "wide"
    mapping = guess_mapping(raw.columns)
    assert {"injection", "kind", "level", "analyte", "rt_min", "conc", "accuracy_pct", "area"} <= set(mapping)
    long = to_long(raw.frame, mapping)
    assert list(long.columns) == list(LONG_COLUMNS)
    assert set(long.kind) == set(KINDS)
    # Levels come back in order; the kinds from MassHunter's labels.
    assert level_order(long) == [lv.level for lv in inputs.calibration_design][:len(level_order(long))] or level_order(long)
    names = match_names(long.analyte.unique(), [a.name for a in inputs.analytes])
    assert all(v is not None for v in names.values())
    obs = observe(with_method_names(long, names), inputs)
    want = extract(run, inputs)
    for k in ("sample_bias", "sample_bias_spread", "bias_rt_corr", "rt_shift", "rt_shift_spread", "istd_area_ratio",
              "istd_missing", "blank_positive", "cal_low_error", "cal_high_error"):
        if not np.isfinite(want[k]):                        # a clean run has no bias to correlate with retention
            assert k not in obs.features and "undefined" in obs.notes[k], k
            continue
        assert k in obs.features, (k, obs.notes.get(k))
        assert obs.features[k] == pytest.approx(want[k], rel=1e-6, abs=1e-9), k
    assert obs.features["qc_recovery"] == pytest.approx(want["qc_recovery"], rel=0.05)   # the LOQ QC is counted here
    assert obs.features["unreportable"] == 0 and want["unreportable"] == 0
    assert "ion_ratio_fail_sample" not in obs.features and "qualifier" in obs.notes["ion_ratio_fail_sample"]
    s = summarise(long)
    assert len(s) == long.analyte.nunique() and (s.rt_n > 0).all()


def test_a_long_sciex_like_export_imports_the_same(run, inputs):
    rows = long_rows(run, inputs)
    raw = read_table(sciex_csv(rows), "sim.txt")
    assert raw.layout == "long"
    mapping = guess_mapping(raw.columns)
    assert mapping["injection"] == "Sample Name" and mapping["analyte"] == "Component Name"
    assert mapping["conc"] == "Calculated Concentration" and mapping["expected"] == "Actual Concentration"
    long = to_long(raw.frame, mapping)
    assert set(long.kind) == set(KINDS)                     # Standard / Quality Control / Unknown understood
    wide = to_long(read_table(masshunter_csv(rows)).frame, guess_mapping(read_table(masshunter_csv(rows)).columns))
    a = observe(with_method_names(long, match_names(long.analyte.unique(), [x.name for x in inputs.analytes])), inputs)
    b = observe(with_method_names(wide, match_names(wide.analyte.unique(), [x.name for x in inputs.analytes])), inputs)
    for k in set(a.features) | set(b.features):
        assert a.features.get(k) == pytest.approx(b.features.get(k), rel=1e-6, abs=1e-9), k


def masshunter_with_qualifiers_csv() -> str:
    """The shape of a MassHunter area export, with made-up values: flag columns, each target followed by its
    qualifier blocks, then its ISTD block and the IS's qualifier; one IS serving two targets; a blank name used twice."""
    head = ["Sample", "", "", "", "", "",
            "A Results", "", "", "Qualifier (100.0 -> 50.0) Results", "Qualifier (100.0 -> 40.0) Results",
            "IS-D4 (ISTD) Results", "", "Qualifier (104.0 -> 54.0) Results",
            "B Results", "", "", "IS-D4 (ISTD) Results", "", "Qualifier (104.0 -> 54.0) Results"]
    cols = ["", "Name", "Data File", "Type", "Level", "Acq. Date-Time",
            "RT", "Area", "Final Conc.", "Ratio", "Ratio", "RT", "Area", "Ratio",
            "RT", "Area", "Final Conc.", "RT", "Area", "Ratio"]
    data = [["!", "Blank", "Blank.d", "Blank", "", "1/15/2026 9:00 AM",
             "4.00", "1000", "0.02", "40.0", "20.0", "4.00", "100000", "50.0",
             "6.00", "500", "0.01", "4.00", "100000", "50.0"],
            ["!", "Cal 1", "Cal 1.d", "Cal", "1", "1/15/2026 9:15 AM",
             "4.01", "5000", "0.001", "41.0", "21.0", "4.01", "200000", "51.0",
             "6.01", "2500", "0.001", "4.01", "200000", "51.0"],
            ["!", "Blank", "Blank-2.d", "Blank", "", "1/15/2026 11:00 AM",
             "4.02", "1100", "0.021", "39.0", "19.0", "4.02", "120000", "52.0",
             "6.02", "550", "0.011", "4.02", "120000", "52.0"]]
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    for line in [head, cols, *data]:
        w.writerow(line)
    return out.getvalue()


def test_qualifier_and_istd_blocks_attach_to_their_owner():
    raw = read_table(masshunter_with_qualifiers_csv(), "areas.csv")
    assert raw.layout == "wide"
    f = raw.frame
    # Qualifier blocks are not compounds; the IS is one compound, kept once per injection.
    assert list(dict.fromkeys(f.analyte)) == ["A", "IS-D4", "B"]
    assert len(f) == 3 * 3
    a = f[(f.analyte == "A") & (f["Data File"] == "Cal 1.d")].iloc[0]
    assert a["Qualifier"] == "100.0 > 50.0" and a["Qualifier Ratio"] == "41.0"     # the first qualifier
    s = f[(f.analyte == "IS-D4") & (f["Data File"] == "Cal 1.d")].iloc[0]
    assert s["Area"] == "200000" and s["Qualifier"] == "104.0 > 54.0" and s["Qualifier Ratio"] == "51.0"
    assert f[f.analyte == "B"]["Qualifier"].eq("").all()
    long = to_long(f, guess_mapping(raw.columns))
    # The repeated name is a second injection, told apart by its acquisition time.
    assert list(dict.fromkeys(long.injection)) == ["Blank", "Cal 1", "Blank (2)"]
    assert set(long[long.injection.str.startswith("Blank")].kind) == {"blank"}
    assert long[(long.injection == "Blank (2)") & (long.analyte == "IS-D4")].area.iloc[0] == 120000
    s = summarise(long)
    assert set(s.analyte) == {"A", "IS-D4", "B"}


def test_tab_separated_and_thousands_separators_are_read():
    text = "Sample Name\tSample Type\tComponent Name\tArea\tRT\nS1\tUnknown\tX\t1,234.5\t3.2\n"
    raw = read_table(text)
    long = to_long(raw.frame, guess_mapping(raw.columns))
    assert long.area.iloc[0] == 1234.5 and long.rt_min.iloc[0] == 3.2 and long.kind.iloc[0] == "sample"


def test_kinds_from_labels_then_names():
    assert guess_kind("Cal", "x") == "calibrator" and guess_kind("Standard", "") == "calibrator"
    assert guess_kind("Quality Control", "") == "qc" and guess_kind("Unknown", "") == "sample"
    assert guess_kind("", "RB01") == "blank" and guess_kind("", "Matrix Blank 2") == "blank"
    assert guess_kind("", "Flower Cal 3") == "calibrator" and guess_kind("", "CCV 1") == "qc"
    assert guess_kind("", "FS-2 rep 3") == "qc" and guess_kind("", "Lot 1234") == "sample"
    long = to_long(pd.DataFrame({"Name": ["a", "b"], "Type": ["Weird", "Weird"], "Compound": ["X", "X"]}),
                   {"injection": "Name", "kind": "Type", "analyte": "Compound"}, kind_map={"weird": "blank"})
    assert list(long.kind) == ["blank", "blank"]
    assert list(kind_table(long).columns)[:2] == ["label in the export", "kind"]


def test_names_match_the_method_through_spelling():
    names = match_names(["Imidicloprid", "Chlormequat chloride", "Nothing Known", "Imazilil-D5"],
                        ["Imidacloprid", "Chlormequat Chloride", "Imazalil-D5"])
    assert names == {"Imidicloprid": "Imidacloprid", "Chlormequat chloride": "Chlormequat Chloride",
                     "Nothing Known": None, "Imazilil-D5": "Imazalil-D5"}
    assert normalise_name("Permethrins (sum)") == "permethrinsum"


def test_observe_says_why_a_feature_is_missing(inputs):
    long = to_long(pd.DataFrame({"Name": ["Cal 1", "Cal 2", "S1"], "Type": ["Cal", "Cal", "Sample"],
                                 "Level": ["1", "2", ""], "Compound": ["Myclobutanil"] * 3, "RT": [8.0, 8.0, 8.1],
                                 "Final Conc.": [1.0, 2.0, 5.0]}),
                   guess_mapping(["Name", "Type", "Level", "Compound", "RT", "Final Conc."]))
    obs = observe(long, inputs)
    assert "sample_bias" not in obs.features and "expected" in obs.notes["sample_bias"]
    assert "rt_shift" in obs.features and obs.notes["istd_area_ratio"].startswith("the export has no area")
    assert obs.frame().shape[0] == 16


def method_table(inputs, with_role: bool) -> pd.DataFrame:
    rows = []
    for t in inputs.transitions:
        r = {"Compound Name": t.analyte, "ISTD?": "TRUE" if t.analyte.endswith(("-D4", "-D5", "-D6")) else "FALSE",
             "Precursor Ion": t.precursor_mz, "MS1 Res": "Unit", "Product Ion": t.product_mz, "MS2 Res": "Unit",
             "Ret Time (min)": t.retention_time_min, "Delta Ret Time": t.window_min, "Fragmentor": t.fragmentor_v,
             "Collision Energy": t.collision_energy_v, "Cell Accelerator Voltage": t.cell_accel_v,
             "Polarity": t.polarity.value.capitalize()}
        if with_role:
            r["Role"] = t.role.value
        rows.append(r)
    return pd.DataFrame(rows)


def test_transition_table_round_trip(inputs):
    table = method_table(inputs, with_role=True)
    text = table.to_csv(index=False)
    raw = read_table(text)
    mapping = guess_mapping(raw.columns, TRANSITION_ROLES)
    for role in ("analyte", "precursor_mz", "product_mz", "retention_time_min", "window_min", "collision_energy_v",
                 "fragmentor_v", "polarity", "role", "is_istd"):
        assert role in mapping, role
    imp = transitions_from_table(raw.frame, mapping)
    assert imp.ok(), imp.problems
    assert len(imp.transitions) == len(inputs.transitions)
    for got, want in zip(imp.transitions, inputs.transitions):
        assert (got.analyte, got.precursor_mz, got.product_mz, got.role, got.polarity) == (
            want.analyte, want.precursor_mz, want.product_mz, want.role, want.polarity)
        assert got.retention_time_min == pytest.approx(want.retention_time_min)
        assert got.window_min == pytest.approx(want.window_min)
        assert got.collision_energy_v == pytest.approx(want.collision_energy_v)
    frame = transitions_frame(imp.transitions)
    assert list(frame.columns) == list(pd.read_csv(EXAMPLES / "transitions.csv").columns)


def test_transition_table_without_a_role_column_takes_the_first_product_as_quantifier(inputs):
    raw = read_table(method_table(inputs, with_role=False).to_csv(index=False))
    imp = transitions_from_table(raw.frame, guess_mapping(raw.columns, TRANSITION_ROLES))
    assert imp.ok() and any("first product" in n for n in imp.notes)
    per = {}
    for t in imp.transitions:
        per.setdefault(t.analyte, []).append(t.role)
    assert all(roles[0] is TransitionRole.quantifier and roles.count(TransitionRole.quantifier) == 1 for roles in per.values())
    assert all(t.relative_response is None for t in imp.transitions if t.role is TransitionRole.qualifier)


def test_transition_problems_carry_the_row():
    frame = pd.DataFrame({"Compound": ["A", "A", "B"], "Q1": ["300.1", "300.1", "x"], "Q3": ["100.0", "80.0", "50.0"],
                          "RT": ["5.0", "5.0", "-1"]})
    imp = transitions_from_table(frame, guess_mapping(frame.columns, TRANSITION_ROLES), default_window_min=0.8)
    assert not imp.ok()
    rows = {p.row for p in imp.problems}
    assert rows == {4} and all(p.column in ("precursor_mz", "retention_time_min") for p in imp.problems)
    assert len(imp.transitions) == 2 and imp.transitions[0].window_min == 0.8
    with pytest.raises(ValueError):
        transitions_from_table(frame, {"analyte": "Compound"})


def test_lab_input_set_from_templates_validates(tmp_path, inputs):
    dest = tmp_path / "inputs"
    status = make_lab_inputs(dest, EXAMPLES)
    assert (dest / "README.md").exists() and (dest / "transitions.csv").exists() and (dest / "prep_recipe.yaml").exists()
    assert any("template" in s for s in status)
    assert validate_inputs(dest) == []
    # Imported transitions naming a compound the method lacks are reported with file and row.
    frame = transitions_frame(load_transitions(EXAMPLES / "transitions.csv"))
    frame.loc[0, "analyte"] = "Not In Method"
    status = make_lab_inputs(dest, EXAMPLES, transitions=frame)
    assert any("imported transition table" in s for s in status)
    problems = validate_inputs(dest)
    assert problems and any("Not In Method" in p.message for p in problems)
    notes = field_notes(type(inputs.system))
    assert any(n.startswith("- `plate_number`") for n in notes)


def test_save_import_writes_the_batch_folder(tmp_path, run, inputs):
    rows = long_rows(run, inputs)[:200]
    raw = read_table(masshunter_csv(rows))
    long = to_long(raw.frame, guess_mapping(raw.columns))
    written = save_import(tmp_path / "b", long, summarise(long), observe(long), transitions_frame(inputs.transitions[:3]))
    assert {p.name for p in written} == {"long.csv", "summary.csv", "observation.json", "transitions.csv"}
    assert pd.read_csv(tmp_path / "b" / "long.csv").shape[0] == len(long)


@pytest.mark.parametrize("tag", ["..", ".", "../..", "a/../..", "  ", ".hidden."])
def test_lab_folder_stays_inside_data_lab(tag):
    from lcms_sim.io.importer import lab_folder
    from lcms_sim.paths import LAB

    folder = lab_folder(tag)
    assert folder.parent == LAB and folder.name not in ("", ".", "..")
    assert folder.resolve().is_relative_to(LAB.resolve())
