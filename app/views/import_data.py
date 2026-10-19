"""Your data: bring in any quant batch export or MRM transition table by naming its columns, read the batch's
symptoms for the matcher, and build a lab input set from the templates."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import input_folder, inputs, local_mode, page, probe_links  # noqa: E402

from lcms_sim.io.importer import (KINDS, ROLES, TRANSITION_ROLES, guess_mapping, kind_table, lab_folder,  # noqa: E402
                                  make_lab_inputs, match_names, observe, read_table, save_import, summarise,
                                  to_long, transitions_frame, transitions_from_table, validate_inputs,
                                  with_method_names)
from lcms_sim.paths import EXAMPLES, LAB  # noqa: E402

page("Import your data",
     "Any vendor's quant batch export or MRM transition table, by naming its columns. The content is the same "
     "everywhere, only the layout differs. A batch export gives the simulator the batch's symptoms for the "
     "matcher and a per-compound summary; a transition table gives it the method. A lab input set starts from "
     "the bundled files as templates. Nothing here changes the bundled example.", show_rail=False)

ROLE_LABEL = {"injection": "Injection or sample name", "kind": "Type of injection", "level": "Calibration level",
              "acquired": "Acquired at", "analyte": "Compound", "rt_min": "Retention time (min)", "area": "Area",
              "conc": "Concentration", "accuracy_pct": "Accuracy (%)", "expected": "Expected concentration"}
TR_LABEL = {"analyte": "Compound", "precursor_mz": "Precursor m/z", "product_mz": "Product m/z",
            "retention_time_min": "Retention time (min)", "window_min": "Window width (min)",
            "collision_energy_v": "Collision energy (V)", "fragmentor_v": "Fragmentor (V)",
            "cell_accel_v": "Cell accelerator (V)", "polarity": "Polarity", "role": "Quantifier or qualifier",
            "is_istd": "Is an internal standard", "dwell_ms": "Dwell (ms)"}
NONE = "(none)"

i = inputs()
method_names = [a.name for a in i.analytes]

LOCAL = local_mode()

with st.sidebar:
    st.subheader("Import")
    what = st.radio("What", ["Quant batch export", "Transition table", "Lab input set"])
    folder = None
    if LOCAL:
        tag = st.text_input("Batch name", "my_batch", help="Saved under data/lab/<name>/, which git ignores")
        folder = lab_folder(tag)
        st.caption(f"Saves to {folder.relative_to(LAB.parent.parent) if folder.is_relative_to(LAB.parent.parent) else folder}")
    else:
        st.caption("Your file stays in this browser session: nothing is saved on the server, and results come back "
                   "as downloads.")
    data = name = None
    if what != "Lab input set":
        source = st.radio("File", ["Upload", "A file already in data/lab"]) if LOCAL else "Upload"
        if source == "Upload":
            up = st.file_uploader("CSV, TSV or TXT export", type=["csv", "tsv", "txt"])
            if up is not None:
                data, name = up.getvalue(), up.name
        else:
            found = sorted(p for ext in ("*.csv", "*.tsv", "*.txt") for p in LAB.glob(ext)) if LAB.exists() else []
            pick = st.selectbox("File", [NONE] + [p.name for p in found])
            if pick != NONE:
                data, name = (LAB / pick).read_bytes(), pick


@st.cache_data(show_spinner=False)
def parse(data: bytes, name: str):
    return read_table(data, name)


def mapping_editor(columns: list[str], roles: dict, labels: dict, key: str, per_row: int = 5) -> dict[str, str]:
    guess = guess_mapping(columns, roles)
    out: dict[str, str] = {}
    cols = st.columns(per_row, gap="small")
    for n, role in enumerate(roles):
        with cols[n % per_row]:
            opts = [NONE] + columns
            default = guess.get(role, NONE)
            v = st.selectbox(labels[role], opts, index=opts.index(default) if default in opts else 0, key=f"{key}_{role}")
            if v != NONE:
                out[role] = v
    return out


def rel(p: Path) -> str:
    root = LAB.parent.parent
    return str(p.relative_to(root)) if p.is_relative_to(root) else str(p)


# ---------------------------------------------------------------------------------------------------------------
if what == "Quant batch export":
    if data is None:
        st.info(f"Upload a batch export in the sidebar{', or pick one already in data/lab' if LOCAL else ''}. "
                "MassHunter's wide export (two header rows) and long exports (one row per injection and compound: "
                "Sciex, Thermo, Waters, or a hand-made table) are both read.")
        st.stop()
    raw = parse(data, name)
    st.subheader("1. What was read")
    st.markdown(f"**{name}**: {raw.layout} layout, {raw.note}.")
    st.dataframe(raw.frame.head(8), hide_index=True, width="stretch")

    st.subheader("2. Name the columns")
    st.caption("Guessed from the headers; correct any. Injection and compound are required. Concentration and "
               "accuracy give the bias features, area gives the internal standard features.")
    mapping = mapping_editor(raw.columns, ROLES, ROLE_LABEL, "q")
    try:
        long0 = to_long(raw.frame, mapping)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.subheader("3. Kinds of injection")
    st.caption("Each label the export uses for the type of injection, read as blank, calibrator, QC or sample. "
               "Change a kind here; with no type column the names decide (RB, MB, Cal, CCV, LCS, FS, ...).")
    kt = kind_table(long0)
    edited = st.data_editor(kt, hide_index=True, width="stretch", key="kinds",
                            column_config={"kind": st.column_config.SelectboxColumn("kind", options=list(KINDS), required=True)},
                            disabled=["label in the export", "injections", "example"])
    kind_map = {("" if r["label in the export"] == "(blank)" else r["label in the export"]): r["kind"]
                for _, r in edited.iterrows()}
    long = to_long(raw.frame, mapping, kind_map)

    st.subheader("4. Compound names against the method")
    names = match_names(long.analyte.unique(), method_names)
    unmatched = [e for e, m in names.items() if m is None]
    st.markdown(f"{len(names) - len(unmatched)} of {len(names)} export compounds match a method analyte by name"
                + (f"; not in the method: {', '.join(unmatched)}" if unmatched else "."))
    with st.expander("The matches"):
        st.dataframe(pd.DataFrame([{"export": e, "method": m or "(not in the method)"} for e, m in names.items()]),
                     hide_index=True, width="stretch")
    long_m = with_method_names(long, names)

    st.subheader("5. One row per compound")
    summary = summarise(long_m)
    st.dataframe(summary, hide_index=True, width="stretch", height=320,
                 column_config={c: st.column_config.NumberColumn(format="%.4g") for c in summary.columns if c != "analyte"})

    st.subheader("6. The batch's symptoms")
    obs = observe(long_m, i)
    st.caption("Every feature the export supports, against the loaded method; the note says how it was read or why it "
               "could not be. The matcher takes the subset.")
    st.dataframe(obs.frame(), hide_index=True, width="stretch", height=600,
                 column_config={"value": st.column_config.NumberColumn(format="%.4g")})
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Send these symptoms to the matcher"):
            st.session_state["observation"] = dict(obs.features)
            st.session_state["observation_from"] = name
            st.success("Sent. Open the symptom matcher and load the preset 'From the fault lab'.")
    with c2:
        if LOCAL:
            if st.button(f"Save under {rel(folder)}"):
                written = save_import(folder, long_m, summary, obs)
                st.success("Written: " + ", ".join(p.name for p in written))
        else:
            stem = Path(name).stem
            st.download_button("Download long.csv", long_m.to_csv(index=False), f"{stem}_long.csv", "text/csv")
            st.download_button("Download summary.csv", summary.to_csv(index=False), f"{stem}_summary.csv", "text/csv")
            st.download_button("Download observation.json",
                               json.dumps({"features": obs.features, "notes": obs.notes, "names": obs.names}, indent=2),
                               f"{stem}_observation.json", "application/json")
    probe_links([("views/symptom_matcher.py", "Symptom matcher")])

# ---------------------------------------------------------------------------------------------------------------
elif what == "Transition table":
    if data is None:
        st.info("Upload the MRM method's transition table in the sidebar (the acquisition method exported as CSV, or "
                "any table with compound, precursor, product and retention time)"
                f"{', or pick one already in data/lab' if LOCAL else ''}.")
        st.stop()
    raw = parse(data, name)
    if raw.layout != "long":
        st.error("A transition table is one row per transition; this file reads as a wide quant export.")
        st.stop()
    st.subheader("1. What was read")
    st.markdown(f"**{name}**: {raw.note}.")
    st.dataframe(raw.frame.head(8), hide_index=True, width="stretch")
    st.subheader("2. Name the columns")
    mapping = mapping_editor(raw.columns, TRANSITION_ROLES, TR_LABEL, "t", per_row=4)
    window = st.number_input("Window width when the table has none (min)", 0.1, 10.0, 1.0, 0.1)
    names = match_names(raw.frame[mapping["analyte"]].unique() if mapping.get("analyte") in raw.columns else [], method_names)
    try:
        imp = transitions_from_table(raw.frame, mapping, default_window_min=window, names=names, file=name)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    st.subheader("3. The transitions, validated")
    for n in imp.notes:
        st.caption(n)
    if imp.problems:
        st.error(f"{len(imp.problems)} problem(s); rows are counted from the file's header as row 1:")
        st.dataframe(pd.DataFrame([{"row": p.row, "column": p.column, "problem": p.message} for p in imp.problems]),
                     hide_index=True, width="stretch")
    else:
        st.success(f"{len(imp.transitions)} transitions for {imp.frame.analyte.nunique()} compounds, all valid.")
    unmatched = sorted({a for a in imp.frame.analyte if a not in method_names})
    if unmatched:
        st.warning(f"Not in the loaded method's analyte table ({len(unmatched)}): {', '.join(unmatched[:12])}"
                   + (" ..." if len(unmatched) > 12 else "") + ". A lab input set needs them in analytes.csv too.")
    frame = transitions_frame(imp.transitions)
    st.dataframe(frame, hide_index=True, width="stretch", height=400)
    if not LOCAL:
        st.download_button("Download transitions.csv", frame.to_csv(index=False), "transitions.csv", "text/csv")
    elif st.button(f"Save transitions.csv under {rel(folder)}"):
        written = save_import(folder, transitions=frame)
        st.success("Written: " + ", ".join(rel(p) for p in written))

# ---------------------------------------------------------------------------------------------------------------
else:
    if not LOCAL:
        st.info("A lab input set is a folder of files on disk, so it is built and used only when the app runs on your "
                "own machine: clone the repository and start it with `python scripts/run_app.py`. The folder then "
                "holds every file the simulator reads, with a README listing each field; data/examples shows the "
                "format.")
        st.stop()
    dest = LAB / "inputs"
    st.subheader("1. The lab input set")
    st.markdown(f"Folder: `{rel(dest)}`. Every file the simulator reads, copied from the bundled example as a "
                "template unless already there, and the batch's saved transitions written over transitions.csv. "
                "The README in the folder lists every file and field.")
    imported = folder / "transitions.csv"
    st.caption(f"Imported transitions for this batch: {'found' if imported.exists() else 'none saved yet'} ({rel(imported)}).")
    if st.button("Create or refresh the folder"):
        trs = pd.read_csv(imported) if imported.exists() else None
        status = make_lab_inputs(dest, EXAMPLES, transitions=trs)
        st.success("\n".join(f"- {s}" for s in status))
    if dest.exists():
        with st.expander("README.md"):
            st.markdown((dest / "README.md").read_text(encoding="utf-8"))
        st.subheader("2. Validate")
        problems = validate_inputs(dest)
        if problems:
            st.error(f"{len(problems)} problem(s):")
            st.dataframe(pd.DataFrame([{"file": p.file, "row": p.row, "column": p.column, "problem": p.message}
                                       for p in problems]), hide_index=True, width="stretch")
        else:
            st.success("The folder loads and every cross-reference between files holds.")
    else:
        st.info("No lab input set in data/lab/inputs yet.")

    st.subheader("3. Use an input set in the app")
    sets = sorted(d for d in LAB.iterdir() if (d / "analytes.csv").exists()) if LAB.exists() else []
    current = input_folder()
    st.caption(f"The app is reading {rel(current)}. Any folder in data/lab with the standard files is listed.")
    if sets:
        pick = st.selectbox("Input set", sets, format_func=rel,
                            index=sets.index(current) if current in sets else 0)
        problems = validate_inputs(pick)
        if problems:
            st.error(f"{rel(pick)}: {len(problems)} problem(s), first: {problems[0]}")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Use this input set", disabled=not sets or bool(problems)):
            st.session_state["input_folder"] = str(pick)
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()
    with c2:
        if st.button("Back to the bundled example", disabled=current == EXAMPLES):
            st.session_state.pop("input_folder", None)
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()
