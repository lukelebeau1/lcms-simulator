"""Analytical, MS: from molecules on the column to ions counted, and from ions counted to the reported area."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import (chart, Station, follow_index, inputs, matrix_profile, method_provenance, page, provenance_table,  # noqa: E402
                    recipe, sources_line, targets)
import style  # noqa: E402
from ion_stream import TYPICAL_DOWNSTREAM  # noqa: E402

from lcms_sim.io.schemas import TransitionRole  # noqa: E402
from lcms_sim.references import STAGE_SOURCES, short_cite  # noqa: E402
from lcms_sim.trace import STAGE_KEYS  # noqa: E402
from lcms_sim.lc.gradient import LCConditions  # noqa: E402
from lcms_sim.lc.retention import retention_time  # noqa: E402
from lcms_sim.ms.acquisition import build_schedule  # noqa: E402
from lcms_sim.ms.noise import expected_counted_ions  # noqa: E402
from lcms_sim.quant.ledger import RecipeError, molecules, run_recipe, with_actual  # noqa: E402

page("MS: molecules to response",
     "Billions of molecules go onto the column; a few thousand ions are counted. Each stage passes a fraction. "
     "Then the counted ions are scaled up into the area the software reports.", station=Station.detector)

i = inputs()
r = recipe()
names = targets(i)
cond = LCConditions.from_method(i.system, i.lc_method)
columns = [s.vessel for s in r.steps if s.op == "inject"]

with st.sidebar:
    st.subheader("Injection")
    compound = st.selectbox("Compound", names, index=follow_index(names))
    column = st.selectbox("Vial injected", columns, index=follow_index(columns, "injection"),
                          format_func=lambda c: c.removeprefix("Column: "))
    trs = [t for t in i.transitions if t.analyte == compound]
    k_tr = st.selectbox("Transition", range(len(trs)),
                        format_func=lambda k: f"{trs[k].precursor_mz:g} > {trs[k].product_mz:g} ({trs[k].role.value})")

a = next(x for x in i.analytes if x.name == compound)
tr = trs[k_tr]
edits = st.session_state.get("bench_actual", {})
try:
    vessels, _ = run_recipe(with_actual(r, edits), "actual")
except RecipeError as e:
    st.error(f"The bench steps cannot be run as edited: {e}")
    st.stop()
ng = vessels[column].ng.get(compound, 0.0)
if ng <= 0:
    st.info(f"{column.removeprefix('Column: ')} holds no {compound}. Pick a calibrator, QC or a sample that contains it.")
    st.stop()
n0 = molecules(ng, a.molecular_weight)

# The simulator's own numbers for this compound and transition.
tR = retention_time(a.log_kw, a.S, cond)
sched = build_schedule(i.transitions, i.acquisition, i.lc_method.run_time_min)
j = i.transitions.index(tr)
kc = int(np.clip(np.searchsorted(sched.cycle_starts, tR, side="right") - 1, 0, len(sched.cycle_starts) - 1))
dwell = float(sched.dwell_ms[kc]) if sched.active[kc, j] else 0.0
cycle = i.acquisition.cycle_time_ms
gain = i.acquisition.gain_area_per_ion
mf = float(matrix_profile().factor(tR)) if vessels[column].matrix_g > 0 else 1.0
rel = tr.relative_response or 1.0
area = a.response_factor * ng * mf * rel                     # the simulator's expected area
lumped = a.response_factor / gain / (n0 / ng)                 # ions reaching the detector per molecule, listening always

TYPICAL = TYPICAL_DOWNSTREAM   # shared with the ion path close-ups (app/ion_stream.py)
spray_default = lumped / math.prod(f for _, f, _ in TYPICAL)

st.subheader("Where the molecules go")
st.caption(f"{column.removeprefix('Column: ')}: {1000 * ng:.4g} pg of {compound} injected, from Bench steps"
           + (f" with {len(edits)} edited step(s)." if edits else ".")
           + " Edit the Fraction column of the typical stages to ask what-if; the others come from the model.")
stages = pd.DataFrame([
    {"Stage": "Ionised in the spray and drawn into the inlet", "Fraction": spray_default,
     "Typical": "1e-5-1e-2 at 0.6 mL/min; most of the spray never enters",
     "Source": "assumed: what is left of the simulator's response factor after the typical stages "
               f"(range: {short_cite(STAGE_SOURCES['spray'])})"},
    {"Stage": "Not suppressed by co-eluting matrix", "Fraction": mf, "Typical": "0.2-1",
     "Source": f"model: matrix profile at this retention time ({short_cite(('matuszewski_2003', 'bonfiglio_1999'))})"
     if mf != 1.0 else "model: no matrix in this vial"},
    *[{"Stage": s, "Fraction": f, "Typical": t, "Source": f"assumed: typical value ({short_cite(STAGE_SOURCES[key])})"}
      for (s, f, t), key in zip(TYPICAL, STAGE_KEYS)],
    *([{"Stage": "Share of the fragments in this qualifier vs the quantifier", "Fraction": rel,
        "Typical": "the ion ratio", "Source": "model: relative response (placeholder)"}]
      if tr.role is not TransitionRole.quantifier else []),
    {"Stage": "Arrives while this transition is being measured", "Fraction": dwell / cycle,
     "Typical": "dwell / cycle", "Source": f"model: {dwell:.1f} ms of a {cycle:g} ms cycle at {tR:.2f} min"},
])
edited = st.data_editor(
    stages, hide_index=True, width="stretch", key=f"budget_{compound}_{k_tr}",
    disabled=["Stage", "Typical", "Source"],
    column_config={"Fraction": st.column_config.NumberColumn("Fraction passing", min_value=0.0, max_value=1.0,
                                                              format="%.3g")},
)
fractions = edited["Fraction"].astype(float).to_numpy()
remaining = n0 * np.cumprod(fractions)
counted = float(remaining[-1])

stages = ["Injected", *edited["Stage"]]
fig = go.Figure(go.Bar(
    y=stages, x=[n0, *remaining], orientation="h",
    marker_color=[style.LILAC] * len(fractions) + [style.SAFFRON],
    text=[f"{v:.3g}" for v in (n0, *remaining)], textposition="outside",
    hovertemplate="%{y}: %{x:.3g}<extra></extra>"))
lo = math.log10(max(float(remaining.min()), 1.0))
fig.update_xaxes(type="log", title_text="Molecules or ions left (log scale)", range=[lo - 0.5, math.log10(n0) + 1.2])
fig.update_yaxes(autorange="reversed")
# Left margin from the longest stage name at ~7 px per PT Mono character: automargin leaves the longest one cut off.
fig.update_layout(height=80 + 42 * (len(fractions) + 1), margin=dict(t=10, b=40, l=16 + 7 * max(map(len, stages))),
                  template="room")
chart(fig, width="stretch")
sources_line(("page_2007", "kebarle_verkerk_2009", "matuszewski_2003", "bonfiglio_1999", "king_2000", "dawson_1995",
              "hoffmann_stroobant_2007", "douglas_french_1992"),
             label="Sources (spray and inlet; matrix suppression; Q1 and Q3 at unit resolution; the cell; the multiplier; "
                   "every fraction in the typical rows is an assumed value inside the sourced range)")


def short(n: float) -> str:
    """26.8 bn, 3.6 M, 1,015: short enough for a metric tile."""
    for size, word in ((1e12, "tn"), (1e9, "bn"), (1e6, "M")):
        if n >= size:
            return f"{n / size:.3g} {word}"
    return f"{n:,.0f}"


c = st.columns(2) + st.columns(2)
c[0].metric("Injected", short(n0), help=f"Molecules on the column: {n0:.3e}")
c[1].metric("Counted", f"{counted:,.0f}", help="Ions counted over the whole peak")
c[2].metric("1 counted in", short(n0 / counted) if counted > 0 else "-",
            help="Molecules injected per ion counted")
c[3].metric("Scatter", f"{100 / math.sqrt(counted):.1f}%" if counted > 0 else "-",
            help="Counting scatter, 1 / sqrt(N): the least scatter this many ions allow")

# From counted ions to the reported area.
st.subheader("From ions counted to the reported area")
scale = cycle / dwell if dwell > 0 else float("nan")
area_now = counted * gain * scale
st.markdown(f"""
1. **Each ion becomes a pulse.** An ion hitting the electron multiplier knocks off electrons from its first
   surface; each of those knocks off more down a chain of dynodes, roughly a million electrons per ion. That
   cascade is a current pulse big enough to count.
2. **The instrument reports a rate, not a count.** During its {dwell:.1f} ms dwell this transition counts
   {counted:,.0f} ions over the whole peak. The software divides by the time it was listening, so it reports
   ions per second as if it had listened all the time: x {scale:,.1f} (the {cycle:g} ms cycle / {dwell:.1f} ms
   dwell). That is why a shorter dwell gives the same area with more scatter, not a smaller area.
3. **Rate is converted to area units.** The detector gain sets how many area units one ion is worth:
   {gain:g} area units per ion here (placeholder until replicate areas are loaded).
""")
sources_line(("hoffmann_stroobant_2007", "knoll_2010", "sciex_scheduled_mrm", "agilent_5990_3595"),
             label="Sources (the multiplier cascade; counting scatter; dwell buys precision, not signal; reporting a rate "
                   "and the gain are the model's conventions)")
steps_tbl = pd.DataFrame([
    {"Step": "Ions counted over the peak", "Value": counted, "Unit": "ions"},
    {"Step": f"x cycle / dwell ({cycle:g} / {dwell:.1f} ms)", "Value": counted * scale, "Unit": "ions, as if always listening"},
    {"Step": f"x gain ({gain:g} area units per ion)", "Value": area_now, "Unit": "area units (the response)"},
])
st.dataframe(steps_tbl.style.format({"Value": "{:,.4g}"}), hide_index=True, width="stretch")
check = expected_counted_ions(area, gain, dwell, cycle)
st.caption(f"With the default fractions this reproduces the simulator's expected area for this injection, "
           f"{area:,.4g} ({check:,.0f} ions counted). Edited fractions show what the area would become. "
           "The response factor and gain are placeholders until a lab's area export is loaded, so the "
           "absolute counts are illustrative; the dwell, matrix and ion-ratio stages are the model's own.")
provenance_table(method_provenance(i))
