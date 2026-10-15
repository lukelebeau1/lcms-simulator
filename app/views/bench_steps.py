"""Pre-analytical: every bench step from reference standard to injection, as amounts in each vessel."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import Station, follow_index, inputs, method, method_sources, page, recipe, sources_line, targets  # noqa: E402

from lcms_sim.quant.ledger import RecipeError, lineage, molecules, run_recipe, with_actual  # noqa: E402

page("Bench steps",
     "The method's preparation written out step by step, from the reference standards to the injection. Each vessel holds a volume and an amount (ng) of every "
     "compound; pipetting moves a share of the amount, so concentrations follow. Edit the Actual column to "
     "model what happened at the bench: the change carries to every vessel made from it.", station=Station.bench)

i = inputs()
r = recipe()
names = targets(i) + [a.name for a in i.analytes if a.is_internal_standard]
mw = {a.name: a.molecular_weight for a in i.analytes}
vessels_in_order = list(dict.fromkeys(s.vessel for s in r.steps))

if "bench_actual" not in st.session_state:
    st.session_state.bench_actual = {}
edits: dict[int, float] = st.session_state.bench_actual

with st.sidebar:
    st.subheader("On this page")
    compound = st.selectbox("Compound", names, index=follow_index(names))
    follow = st.selectbox("Vessel", vessels_in_order, index=follow_index(vessels_in_order, "injection"))
    changed_slot = st.empty()   # filled in after the table has applied this run's edits
    only_carrying = st.checkbox("Only steps that hold this compound", value=True,
                                help="Hides stocks and transfers of other compounds that do not touch it")
    if st.button("Reset all actual values"):
        st.session_state.bench_actual = {}
        for key in [k for k in st.session_state if str(k).startswith("editor_")]:
            del st.session_state[key]
        st.rerun()

keep = lineage(r, follow)
if only_carrying:
    _, log0 = run_recipe(r, "nominal")
    keep = [k for k in keep if log0[k].vessel.ng.get(compound, 0.0) > 0]


def amount_of(k: int) -> tuple[str, float | None, float | None]:
    """Unit, nominal and current actual value of step k (a volume, or a mass for a weighing)."""
    s = r.steps[k]
    na, unit = (s.volume_ul, "uL") if s.volume_ul is not None else (s.mass_g, "g") if s.mass_g is not None else (None, "")
    return (unit, na.nominal, edits.get(k, na.actual)) if na is not None else ("", None, None)


# Steps behind the followed vessel, with the editable actual values.
st.subheader(f"Steps that make {follow}")
rows = []
for k in keep:
    s = r.steps[k]
    unit, nom, act = amount_of(k)
    rows.append({"#": k + 1, "Actual": act, "Nominal": nom, "Unit": unit, "Step": s.op, "Into": s.vessel,
                 "From": s.source or "", "Note": s.note})
table = pd.DataFrame(rows)
edited = st.data_editor(
    table, hide_index=True, width="stretch", key=f"editor_{follow}",
    disabled=[c for c in table.columns if c != "Actual"],
    column_config={"Actual": st.column_config.NumberColumn("Actual", min_value=0.0, format="%.4g",
                                                           help="What was actually pipetted or weighed")},
)
for row in edited.itertuples():
    k = row[1] - 1
    nom, act = row.Nominal, row.Actual
    if nom is None or pd.isna(nom) or act is None or pd.isna(act):
        continue
    if abs(act - nom) > 1e-12:
        edits[k] = float(act)
    else:
        edits.pop(k, None)

changed_slot.caption(f"{len(edits)} step(s) changed from nominal: "
                     + (", ".join(f"#{k + 1}" for k in sorted(edits)) if edits else "none"))

try:
    vn, log_n = run_recipe(r, "nominal")
    va, log_a = run_recipe(with_actual(r, edits), "actual")
except RecipeError as e:
    st.error(f"{e}. Lower that volume or the ones drawn from the same vessel earlier.")
    st.stop()

# What the compound does at each of those steps.
st.subheader(f"{compound} through those steps")
trace = []
for k in keep:
    n, a = log_n[k], log_a[k]
    cn, ca = n.vessel.conc_ng_ml(compound), a.vessel.conc_ng_ml(compound)
    trace.append({"#": k + 1, "After this step": f"{n.step.op} -> {n.vessel.name}",
                  "Volume (uL)": a.vessel.volume_ul,
                  f"{compound} (ng)": a.vessel.ng.get(compound, 0.0),
                  # Molecules only where they are counted onto the column: the injection.
                  "Molecules": (molecules(a.vessel.ng.get(compound, 0.0), mw[compound])
                                if n.step.op == "inject" and mw.get(compound) else None),
                  "Concentration (ng/mL)": ca,
                  "Nominal (ng/mL)": cn,
                  "Actual / nominal": ca / cn if cn else None})
st.dataframe(pd.DataFrame(trace).style.format(
    {"Volume (uL)": "{:,.6g}", f"{compound} (ng)": "{:,.6g}", "Molecules": "{:.3e}",
     "Concentration (ng/mL)": "{:,.6g}",
     "Nominal (ng/mL)": "{:,.6g}", "Actual / nominal": "{:.4f}"}, na_rep="-"), hide_index=True, width="stretch")

# The followed vessel.
v_n, v_a = vn[follow], va[follow]
st.subheader(follow)
c = st.columns(4)
if follow.startswith("Column:"):
    ng_a = v_a.ng.get(compound, 0.0)
    c[0].metric("On the column", f"{1000 * ng_a:.4g} pg")
    c[1].metric("Molecules", f"{molecules(ng_a, mw[compound]):.3g}" if mw.get(compound) else "-",
                help=f"ng x 1e-9 / {mw.get(compound)} g/mol x Avogadro")
    c[2].metric("Injected volume", f"{v_a.volume_ul:g} uL")
    c[3].metric("Actual / nominal", f"{ng_a / v_n.ng[compound]:.4f}" if v_n.ng.get(compound) else "-")
else:
    c[0].metric("Volume", f"{v_a.volume_ul:.4g} uL")
    c[1].metric(f"{compound}", f"{v_a.ng.get(compound, 0.0):.4g} ng")
    c[2].metric("Concentration", f"{v_a.conc_ng_ml(compound):.4g} ng/mL")
    cn = v_n.conc_ng_ml(compound)
    c[3].metric("Actual / nominal", f"{v_a.conc_ng_ml(compound) / cn:.4f}" if cn else "-")
st.caption("The signal and reported results in the other views still use the closed-form prep values; "
           "feeding these bench-step amounts into them is the next step.")
sources_line((*method_sources(), "mettler_pipetting"),
             label="Sources (steps and volumes from the method; why an air-displacement pipette under-delivers "
                   "volatile or viscous liquids, the Follow page's bench factors; molecules = ng / molecular weight x "
                   "Avogadro's number)")

remark = method().remarks.get("bench_steps")
if remark:
    with st.expander("Try it: a note for this method"):
        st.markdown(remark)
