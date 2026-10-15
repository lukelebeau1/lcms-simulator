"""Troubleshooting, fault lab: pick a fault and a magnitude; before and after for every analyte."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import inputs, page, probe_links, seed_control, sources_line, targets  # noqa: E402

from lcms_sim.faults.library import LIBRARY, Scenario  # noqa: E402
from lcms_sim.faults.matcher import load_sweep_table, sweep_magnitudes, tolerances  # noqa: E402
from lcms_sim.faults.signatures import FEATURE_MEANING, FEATURES, extract  # noqa: E402
from lcms_sim.io.schemas import TransitionRole  # noqa: E402
from lcms_sim.pipeline import Kind  # noqa: E402
from lcms_sim.rng import make_rng  # noqa: E402

page("Fault lab",
     "Cause to effect. Pick a fault and its size here; the simulator injects it into the whole run and shows the "
     "symptoms it produces: before and after for every analyte in one injection, and the run's symptom vector. "
     "The symptom matcher runs the other way, from symptoms back to faults. Every number is simulated.",
     show_rail=False)

i = inputs()
rng, seed = seed_control()
keys = list(LIBRARY)
st.subheader("1. Choose the fault")
c_fault, c_size, c_run = st.columns([1.4, 1.2, 1.0], gap="large")
with c_fault:
    key = st.selectbox("Fault", keys, format_func=lambda k: LIBRARY[k].name)
    f = LIBRARY[key]
    st.caption(f"Stage: {f.stage}. " + (f"Compound-specific: follows {f.varies_with}." if f.scope == "compound"
                                         else "Same for every compound."))
with c_size:
    grid = [float(m) for m in sweep_magnitudes(f, 25)]
    if len(grid) > 1:
        default = min(grid, key=lambda m: abs(m - f.default))
        magnitude = st.select_slider(f"Size: {f.magnitude_label}", options=grid, value=default,
                                     format_func=lambda m: f"{m:.3g}")
    else:
        magnitude = grid[0]
        st.markdown(f"Size: on (this fault has no size)")
with c_run:
    expected = st.checkbox("Noise-free detector", value=True,
                           help="Off: counting and proportional noise with the seed in the sidebar")
    compare = [s.id for s in i.samples if s.kind in ("sample", "qc")]
    sid = st.selectbox("Injection to compare", compare, index=compare.index("PT sample") if "PT sample" in compare else 0)


@st.cache_resource(show_spinner="Simulating the run...")
def runs(key: str, magnitude: float, seed: int, expected: bool):
    detector = "expected" if expected else "simulated"
    base = Scenario.baseline(inputs(), detector=detector)
    r0 = base.simulate(make_rng(seed))
    r1 = LIBRARY[key].apply(base, magnitude).simulate(make_rng(seed))
    return r0, r1, extract(r0, inputs()), extract(r1, inputs())


r0, r1, f0, f1 = runs(key, magnitude, seed, expected)

st.markdown(f"**{f.name}** at {magnitude:.3g} {f.magnitude_label}. Expected signature: {f.signature}")
sources_line(f.sources)

# --- one injection, every analyte --------------------------------------------------------------------------
sample = next(s for s in i.samples if s.id == sid)
rec0, rec1 = r0.record(sid), r1.record(sid)
expected_ratio = {t.analyte: t.relative_response for t in i.transitions if t.role is TransitionRole.qualifier}
qual_idx = {t.analyte: j for j, t in enumerate(i.transitions) if t.role is TransitionRole.qualifier}


def ion_ratio(rec, a):
    j = qual_idx.get(a)
    q = rec.qualifier_areas.get(j) if j is not None else None
    base = rec.areas.get(a, 0.0)
    return q / base if q is not None and base > 0 else np.nan


present = [a for a in targets(i) if sample.true_concentration_ug_kg.get(a, 0.0) > 0]
rows = []
for a in present:
    true = sample.true_concentration_ug_kg.get(a, np.nan)
    rep0, rep1 = r0.reported_ug_kg.get(sid, {}).get(a, np.nan), r1.reported_ug_kg.get(sid, {}).get(a, np.nan)
    rows.append({"analyte": a, "RT before (min)": r0.retention_min[a], "RT after": r1.retention_min[a],
                 "area before": rec0.areas.get(a, 0.0), "area after": rec1.areas.get(a, 0.0),
                 "ion ratio before": ion_ratio(rec0, a), "ion ratio after": ion_ratio(rec1, a),
                 "true (ug/kg)": true, "reported before": rep0, "reported after": rep1,
                 "bias before": rep0 / true if true else np.nan, "bias after": rep1 / true if true else np.nan})
df = pd.DataFrame(rows)
df["area after / before"] = df["area after"] / df["area before"].replace(0, np.nan)
df["RT shift (min)"] = df["RT after"] - df["RT before (min)"]

st.subheader(f"2. The symptoms: {sid}, every analyte present, before and after")
st.caption(f"{len(present)} of {len(targets(i))} targets are in this injection; a QC carries them all, the PT sample a subset.")
c1, c2 = st.columns([1.6, 1.0], gap="large")
with c1:
    st.dataframe(df[["analyte", "RT before (min)", "RT shift (min)", "area before", "area after / before",
                     "ion ratio before", "ion ratio after", "true (ug/kg)", "reported before", "reported after",
                     "bias after"]], hide_index=True, width="stretch", height=420,
                 column_config={c: st.column_config.NumberColumn(format="%.3g") for c in df.columns if c != "analyte"})
with c2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["RT before (min)"], y=df["bias before"], mode="markers", name="before",
                             marker=dict(symbol="circle-open", size=9, color="#9aa0a6"), text=df["analyte"]))
    fig.add_trace(go.Scatter(x=df["RT after"], y=df["bias after"], mode="markers", name="after",
                             marker=dict(size=9, color="#f2a33a"), text=df["analyte"]))
    fig.add_hline(y=1.0, line=dict(color="#888", dash="dot"))
    fig.update_layout(title="Reported / true against retention time", xaxis_title="RT (min)",
                      yaxis_title="reported / true", height=420, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    scope_note = ("the pattern across compounds is the diagnosis" if f.scope == "compound"
                  else "every compound moves together, as a dilution would")
    st.caption(f"{'Compound-specific' if f.scope == 'compound' else 'Uniform'}: {scope_note}.")

# --- the feature vector ------------------------------------------------------------------------------------
st.subheader("3. The run's symptom vector, before and after")
try:
    tol = tolerances(load_sweep_table())
except (FileNotFoundError, ValueError):
    tol = {}
feat = pd.DataFrame([{"feature": k, "before": f0[k], "after": f1[k],
                      "change": (f1[k] - f0[k]) if np.isfinite(f0[k]) and np.isfinite(f1[k]) else np.nan,
                      "tolerance": tol.get(k, np.nan), "meaning": FEATURE_MEANING[k]} for k in FEATURES])
feat["moved"] = np.where(feat["change"].abs() > feat["tolerance"], "yes", "")
st.dataframe(feat, hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.3g") for c in ("before", "after", "change", "tolerance")})
st.caption("Moved: the change exceeds the matcher's tolerance for that feature (bench floor or twice the replicate noise).")
if st.button("Self-test: send these symptoms to the matcher"):
    st.session_state["observation"] = {k: float(f1[k]) for k in FEATURES if np.isfinite(f1[k])}
    st.session_state["observation_from"] = f"{f.name} at {magnitude:.3g}"
    st.success("Sent. Open the symptom matcher and load the preset 'From the fault lab': it should find "
               f"{f.name}, and it shows what else it cannot rule out.")
probe_links([("views/symptom_matcher.py", "Symptom matcher: which faults fit an observation"),
             ("views/matrix_overlay.py", "Matrix overlay: suppression against retention time")])
