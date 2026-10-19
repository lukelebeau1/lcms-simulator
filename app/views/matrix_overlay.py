"""Troubleshooting, matrix overlay: the suppression profile against retention time, with every analyte and its
internal standard drawn on it. Load a post-column infusion pair (matrix blank and solvent blank) or use the
bundled synthetic profile."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import chart, follow_index, inputs, matrix_profile, page, probe_links, sources_line, targets  # noqa: E402

from lcms_sim.io.schemas import InfusionPoint  # noqa: E402
from lcms_sim.lc.gradient import LCConditions  # noqa: E402
from lcms_sim.lc.retention import retention_time  # noqa: E402
from lcms_sim.quant.matrix import profile_from_traces  # noqa: E402

page("Matrix overlay",
     "The matrix factor is the ratio of two post-column infusion traces, matrix blank over solvent blank, read at "
     "each retention time: 1 means no effect, 0.4 means 60% suppression. Each analyte is drawn where it elutes, "
     "with its internal standard; the ratio of their factors is the part the IS does not correct.", show_rail=False)

i = inputs()
names = targets(i)


def read_trace(upload) -> list[InfusionPoint]:
    df = pd.read_csv(upload)
    cols = {c.lower().strip(): c for c in df.columns}
    t, s = cols.get("time_min") or cols.get("time"), cols.get("signal") or cols.get("intensity")
    if t is None or s is None:
        raise ValueError("the CSV needs columns time_min and signal")
    return [InfusionPoint(time_min=float(a), signal=float(b)) for a, b in zip(df[t], df[s])]


with st.sidebar:
    st.subheader("Profile")
    source = st.radio("Source", ["Bundled synthetic profile", "Upload infusion traces"])
    smooth = st.slider("Smoothing window (min)", 0.0, 0.5, 0.05, 0.01,
                       help="Median filter on the ratio before interpolation")
    scale = st.slider("Suppression scale (x)", 0.0, 3.0, 1.0, 0.1,
                      help="1 = as measured; 2 = a sample with twice the matrix")
    st.subheader("Highlight")
    highlight = st.selectbox("Analyte", names, index=follow_index(names))

profile = matrix_profile()
label = "bundled synthetic profile (assumed)"
if source == "Upload infusion traces":
    up_m = st.sidebar.file_uploader("Matrix blank trace (time_min, signal)", type="csv")
    up_s = st.sidebar.file_uploader("Solvent blank trace (time_min, signal)", type="csv")
    if up_m is not None and up_s is not None:
        try:
            profile = profile_from_traces(read_trace(up_m), read_trace(up_s), smooth_min=smooth)
            label = f"{up_m.name} / {up_s.name} (measured)"
        except (ValueError, KeyError) as e:
            st.sidebar.error(f"Could not read the traces: {e}")
    else:
        st.sidebar.info("Upload both traces; the bundled profile is shown until then.")
elif smooth != 0.05:
    profile = profile_from_traces(i.infusion_matrix, i.infusion_solvent, smooth_min=smooth)

cond = LCConditions.from_method(i.system, i.lc_method)
rt = {a.name: retention_time(a.log_kw, a.S, cond) for a in i.analytes}
istd_of = {a.name: a.internal_standard for a in i.analytes if not a.is_internal_standard}

t = np.linspace(0.0, i.lc_method.run_time_min, 600)
mf = profile.factor(t, scale)
rows = []
for a in names:
    s = istd_of.get(a)
    mfa = float(profile.factor(rt[a], scale))
    mfs = float(profile.factor(rt[s], scale)) if s else np.nan
    rows.append({"analyte": a, "RT (min)": rt[a], "matrix factor": mfa, "IS": s or "", "IS RT (min)": rt[s] if s else np.nan,
                 "IS matrix factor": mfs, "uncorrected (analyte / IS)": mfa / mfs if s and mfs > 0 else np.nan,
                 "RT gap to IS (min)": abs(rt[a] - rt[s]) if s else np.nan})
df = pd.DataFrame(rows)

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=mf, mode="lines", name="matrix factor", line=dict(color="#4fc3f7", width=2)))
fig.add_hline(y=1.0, line=dict(color="#888", dash="dot"))
fig.add_trace(go.Scatter(x=df["RT (min)"], y=df["matrix factor"], mode="markers", name="analytes",
                         marker=dict(size=8, color="#f2a33a"), text=df["analyte"],
                         hovertemplate="%{text}<br>RT %{x:.2f} min<br>MF %{y:.2f}<extra></extra>"))
istds = df.dropna(subset=["IS RT (min)"]).drop_duplicates("IS")
fig.add_trace(go.Scatter(x=istds["IS RT (min)"], y=istds["IS matrix factor"], mode="markers", name="internal standards",
                         marker=dict(size=10, symbol="diamond", color="#ce93d8"), text=istds["IS"],
                         hovertemplate="%{text}<br>RT %{x:.2f} min<br>MF %{y:.2f}<extra></extra>"))
h = df[df.analyte == highlight].iloc[0]
if h["IS"]:
    fig.add_trace(go.Scatter(x=[h["RT (min)"], h["IS RT (min)"]], y=[h["matrix factor"], h["IS matrix factor"]],
                             mode="lines+markers", name=f"{highlight} and its IS", line=dict(color="#ef5350", width=2),
                             marker=dict(size=12, color="#ef5350")))
fig.update_layout(xaxis_title="retention time (min)", yaxis_title="matrix factor (1 = no effect)", height=460,
                  margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.17, yanchor="top"),
                  yaxis=dict(range=[0, max(1.2, float(np.nanmax(mf)) * 1.05)]))
chart(fig, width="stretch")
st.caption(f"Profile: {label}. Retention times from the method's LSS parameters (fitted to its measured retention times).")
sources_line(("bonfiglio_1999", "matuszewski_2003", "king_2000", "stokvis_2005"),
             label="Sources (post-column infusion maps suppression against time; the matrix factor; suppression arises "
                   "in the droplets; an internal standard corrects only what it shares; the bundled profile is synthetic)")

st.markdown(f"**{highlight}**: matrix factor {h['matrix factor']:.2f} at {h['RT (min)']:.2f} min"
            + (f"; its IS {h['IS']} has {h['IS matrix factor']:.2f} at {h['IS RT (min)']:.2f} min, so the ratio "
               f"leaves x{h['uncorrected (analyte / IS)']:.2f} uncorrected." if h["IS"] else "; no internal standard."))

st.subheader("Every analyte against its internal standard")
st.caption("Uncorrected = analyte factor / IS factor: the share of the suppression the ratio does not remove. "
           "Sorted worst first. A scale above 1 stretches the deviation from 1, as a dirtier sample would.")
st.dataframe(df.sort_values("uncorrected (analyte / IS)"), hide_index=True, width="stretch", height=480,
             column_config={c: st.column_config.NumberColumn(format="%.3g") for c in df.columns if c not in ("analyte", "IS")})
probe_links([("views/fault_lab.py", "Fault lab: matrix suppression stronger in samples"),
             ("views/symptom_matcher.py", "Symptom matcher")])
