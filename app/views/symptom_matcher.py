"""Troubleshooting, symptom matcher: enter what was observed; read which library faults are consistent with it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import page, probe_links  # noqa: E402

from lcms_sim.references import markdown_cite  # noqa: E402

from lcms_sim.faults.library import LIBRARY  # noqa: E402
from lcms_sim.faults.matcher import (BASELINE, TIE_MARGIN, confusion, default_checks, load_sweep_table, match,  # noqa: E402
                                     tolerances)
from lcms_sim.faults.signatures import FEATURE_MEANING, FEATURES  # noqa: E402

page("Symptom matcher",
     "Effect to cause. Enter the symptoms you have (any subset of the run's symptom vector) and read which library "
     "faults are consistent with them: a ranked "
     "list with the best-fit size, which features fit and which do not, and the bench check that separates the "
     "leading candidates. The matcher reports consistency, not probability, and a fault outside the library "
     "cannot be found.", show_rail=False)


@st.cache_resource(show_spinner=False)
def sweep():
    t = load_sweep_table()
    return t, tolerances(t), default_checks()


table, tol, checks = sweep()


@st.cache_data(show_spinner=False)
def confusion_table(margin: float):
    return confusion(table, checks, margin)

PRESETS = {
    "Enter your own": {},
    "Reference scenario: PT at 30%, uniform, dilution ruled out": {"sample_bias": 0.30, "sample_bias_spread": 0.0,
                                                                     "bias_rt_corr": 0.0},
    "Everything elutes half a minute late": {"rt_shift": 0.5, "rt_shift_spread": 0.02},
    "Blanks positive after the top standard": {"blank_positive": 0.3, "sample_bias": 1.05, "sample_bias_spread": 0.0},
}
if "observation" in st.session_state:
    PRESETS["From the fault lab: " + st.session_state.get("observation_from", "")] = st.session_state["observation"]

with st.sidebar:
    st.subheader("Observation")
    preset = st.selectbox("Preset", list(PRESETS))
    if st.button("Load preset"):
        for k in FEATURES:
            st.session_state[f"obs_on_{k}"] = k in PRESETS[preset]
            if k in PRESETS[preset]:
                st.session_state[f"obs_{k}"] = float(PRESETS[preset][k])
        if "dilution ruled out" in preset:
            st.session_state["rule_out"] = ["volume_error"]
    exclude = st.multiselect("Rule out (already checked)", list(LIBRARY), format_func=lambda k: LIBRARY[k].name,
                             key="rule_out")
    margin = st.number_input("Tie margin (tolerance units)", 0.0, 2.0, TIE_MARGIN, 0.05,
                             help="Faults closer than this are reported as not told apart")

st.subheader("What was observed")
st.caption("Tick a feature you know and enter its value; leave the rest unticked. Tolerances are the matcher's: "
           "a bench floor or twice the baseline's replicate noise, whichever is larger.")
cols = st.columns(3, gap="large")
observation: dict[str, float] = {}
for n, k in enumerate(FEATURES):
    with cols[n % 3]:
        on = st.checkbox(k, key=f"obs_on_{k}", help=FEATURE_MEANING[k])
        v = st.number_input(f"{k} value", key=f"obs_{k}", format="%.4g", disabled=not on,
                            label_visibility="collapsed")
        if on:
            observation[k] = float(v)
        st.caption(f"tolerance {tol[k]:.3g}")

if not observation:
    st.info("Tick at least one feature. The presets in the sidebar fill in a case.")
    st.stop()

res = match(observation, table, exclude=exclude, checks=checks, tie_margin=margin)
lead = res.leader()

st.subheader("Consistent faults, ranked")
if lead.key == BASELINE:
    st.markdown("**No fault in the library fits better than no fault at all** on the features supplied.")
elif lead.consistent:
    st.markdown(f"**{lead.name}** at about {lead.magnitude:.3g} {lead.magnitude_label} fits every feature supplied.")
else:
    st.markdown(f"**{lead.name}** fits best but not within tolerance on: {', '.join(lead.unmatched)}.")
if res.ties:
    st.warning("Not told apart on these features: " + "; ".join(
        f"{LIBRARY[a].name if a in LIBRARY else 'no fault'} and {LIBRARY[b].name if b in LIBRARY else 'no fault'}"
        for a, b in res.ties))
frame = res.frame()
st.dataframe(frame[["rank", "fault", "stage", "scope", "magnitude", "magnitude label", "distance", "matched", "unmatched"]],
             hide_index=True, width="stretch", height=min(60 + 36 * len(frame), 760),
             column_config={"magnitude": st.column_config.NumberColumn(format="%.3g"),
                            "distance": st.column_config.NumberColumn(format="%.2f",
                                                                      help="RMS difference in tolerance units; 1 = one tolerance per feature")})

# --- how each leading candidate sits against the observation ----------------------------------------------
st.subheader("Where the leading candidates fit and where they do not")
top = [m for m in res.ranked[:4] if np.isfinite(m.distance)]
fig = go.Figure()
for m in top:
    z = [(m.simulated[k] - observation[k]) / tol[k] if np.isfinite(m.simulated.get(k, np.nan)) else np.nan
         for k in observation]
    fig.add_trace(go.Bar(name=m.name if m.key != BASELINE else "no fault", x=list(observation), y=z))
fig.add_hrect(y0=-1, y1=1, fillcolor="#4caf50", opacity=0.12, line_width=0)
fig.update_layout(barmode="group", yaxis_title="(simulated - observed) / tolerance", height=380,
                  margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h"))
st.plotly_chart(fig, width="stretch")
st.caption("Inside the green band the feature fits within tolerance. A bar far outside it is the feature that rules the fault out.")

# --- the separating experiment --------------------------------------------------------------------------------
st.subheader("The bench check that separates them")
if not res.separations:
    st.markdown("No pair of leading candidates needs separating, or no listed check tells them apart.")
seen = set()
for s in res.separations:
    if (s.a, s.b) in seen:
        continue
    seen.add((s.a, s.b))
    pair = [x for x in res.separations if (x.a, x.b) == (s.a, s.b)]
    name_a = LIBRARY[s.a].name if s.a in LIBRARY else "no fault"
    name_b = LIBRARY[s.b].name if s.b in LIBRARY else "no fault"
    st.markdown(f"**{name_a}** against **{name_b}**")
    for x in pair[:2]:
        st.markdown(f"- {x.check.name}: {x.check.do}  \n  {name_a}: {x.outcome_a}. {name_b}: {x.outcome_b}."
                    + (f"  \n  Sources: {markdown_cite(x.check.sources)}" if x.check.sources else ""))

with st.expander("Confusion table: what the results alone cannot tell apart"):
    st.caption("Each fault near its default size, matched on every feature. Rank 1 means it is found; "
               "'confused with' lists the faults within the tie margin and the check that separates each.")
    st.dataframe(confusion_table(margin), hide_index=True, width="stretch")
probe_links([("views/fault_lab.py", "Fault lab: inject a fault and read its signature")])
