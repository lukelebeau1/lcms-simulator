"""Analytical, LC: change the LC conditions and see retention move against the method's windows."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import chart, Station, inputs, method_provenance, page, provenance_table, sources_line, targets  # noqa: E402

from lcms_sim.faults.library import LIBRARY, Scenario  # noqa: E402
from lcms_sim.io.schemas import TransitionRole  # noqa: E402
from lcms_sim.lc.gradient import LCConditions  # noqa: E402
from lcms_sim.lc.peaks import peak_sigma  # noqa: E402
from lcms_sim.lc.retention import retention_time  # noqa: E402

page("LC: gradient and retention",
     "Move the gradient, flow, dwell volume and pH. Peaks are drawn at equal height; grey is the method as written.", station=Station.column)

i = inputs()
base = LCConditions.from_method(i.system, i.lc_method)
ph0 = i.lc_method.ph or 3.0

with st.sidebar:
    st.subheader("LC conditions")
    flow = st.slider("Flow (mL/min)", 0.2, 1.0, float(i.lc_method.flow_ml_min), 0.05)
    dwell = st.slider("Dwell volume (mL)", 0.1, 2.0, float(i.system.dwell_volume_ml), 0.05)
    stretch = st.slider("Gradient time x", 0.5, 2.0, 1.0, 0.05, help="Multiplies every time in the gradient program")
    offset = st.slider("%B error (points)", -5.0, 5.0, 0.0, 0.5, help="Mobile phase B made stronger (+) or weaker (-)")
    ph = st.slider("Mobile phase pH", 2.0, 5.0, float(ph0), 0.1)
    show_is = st.checkbox("Show internal standards", value=False)

cond = LCConditions(
    times=tuple(t * stretch for t in base.times),
    phis=tuple(min(max(p + offset / 100.0, 0.0), 1.0) for p in base.phis),
    t0=i.system.void_volume_ml / flow, tD=dwell / flow, run_time=base.run_time,
)
analytes = LIBRARY["ph_error"].apply(Scenario(i), ph - ph0).inputs.analytes if ph != ph0 else i.analytes
window = {t.analyte: (t.retention_time_min, t.window_min) for t in i.transitions if t.role is TransitionRole.quantifier}
names = set(targets(i)) | ({a.name for a in i.analytes if a.is_internal_standard} if show_is else set())

rows, peaks0, peaks1 = [], [], []
for a0, a1 in zip(i.analytes, analytes):
    if a0.name not in names:
        continue
    t_old = retention_time(a0.log_kw, a0.S, base)
    t_new = retention_time(a1.log_kw, a1.S, cond)
    s_old = peak_sigma(a0.log_kw, a0.S, t_old, base, i.system.plate_number, i.system.extra_column_sigma_s)
    s_new = peak_sigma(a1.log_kw, a1.S, t_new, cond, i.system.plate_number, i.system.extra_column_sigma_s)
    rt, w = window[a0.name]
    inside = 0.5 * (math.erf((rt + w / 2 - t_new) / (s_new * math.sqrt(2))) - math.erf((rt - w / 2 - t_new) / (s_new * math.sqrt(2))))
    peaks0.append((t_old, s_old))
    peaks1.append((t_new, s_new))
    rows.append({"Analyte": a0.name, "Method RT": rt, "Now (min)": round(t_new, 3), "Shift (min)": round(t_new - t_old, 3),
                 "Width 4 sigma (s)": round(4 * s_new * 60, 1), "In window (%)": round(100 * inside, 1),
                 "Ionisable": a0.acid_base.value != "neutral"})

t = np.linspace(0, cond.run_time or 20, 6000)


def trace(peaks):
    y = np.zeros_like(t)
    for c, s in peaks:
        y += np.exp(-0.5 * ((t - c) / s) ** 2)
    return y


fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=trace(peaks0), name="Method", line=dict(color="lightgray", width=1)))
fig.add_trace(go.Scatter(x=t, y=trace(peaks1), name="Now", line=dict(width=1.2)))
pb = cond.phi_inlet(t) * 100
fig.add_trace(go.Scatter(x=t, y=pb, name="%B at column", yaxis="y2", line=dict(dash="dot", color="gray")))
fig.update_layout(height=430, margin=dict(t=10, b=40), xaxis_title="Time (min)", yaxis_title="Peaks (equal height)",
                  yaxis2=dict(overlaying="y", side="right", range=[0, 105], title="%B"), template="room",
                  legend=dict(orientation="h", y=1.08))
chart(fig, width="stretch")
sources_line(("snyder_dolan_2007", "snyder_kirkland_dolan_2010", "dolan_2006_dwell", "horvath_1977", "agilent_5990_3595"),
             label="Sources (LSS retention and the gradient delay; peak width; dwell volume shifts; ionisable retention, "
                   "the 1.5 log unit shift of the ionised form assumed; dMRM windows clip what drifts out)")

df = pd.DataFrame(rows).sort_values("Now (min)")
clipped = df[df["In window (%)"] < 99]
c1, c2, c3 = st.columns(3)
c1.metric("Median shift", f"{df['Shift (min)'].median():+.2f} min")
c2.metric("Largest shift", f"{df['Shift (min)'].abs().max():.2f} min")
c3.metric("Peaks partly outside window", f"{len(clipped)} of {len(df)}")
if len(clipped):
    st.error("Clipped peaks lose area in proportion to the share outside the window: "
             + ", ".join(f"{r.Analyte} ({r['In window (%)']:.0f}%)" for _, r in clipped.iterrows()))
st.dataframe(df, hide_index=True, width="stretch", height=420)

provenance_table(method_provenance(i) + [
    ("pH effect", "ionised form 1.5 log units less retained unless measured", "assumed")])
