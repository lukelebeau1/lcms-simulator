"""Post-analytical, calibration: inject the levels through the detector model and compare two fit settings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from common import (chart, Station, acceptance, follow_index, inputs, method, method_provenance, method_sources, page,  # noqa: E402
                    provenance_table, seed_control, sources_line, targets)

from lcms_sim.io.schemas import CalibrationModel, OriginHandling, TransitionRole, Weighting  # noqa: E402
from lcms_sim.lc.gradient import LCConditions  # noqa: E402
from lcms_sim.lc.peaks import peak_sigma  # noqa: E402
from lcms_sim.lc.retention import retention_time  # noqa: E402
from lcms_sim.ms.acquisition import build_schedule  # noqa: E402
from lcms_sim.ms.noise import PeakSpec, acquire  # noqa: E402
from lcms_sim.quant.calibration import fit_calibration  # noqa: E402

page("Calibration",
     "Each level is injected through the detector model (counting noise, proportional noise, optional "
     "saturation). Compare two fit settings on the same data.")

i = inputs()
rng, seed = seed_control()
names = targets(i)
with st.sidebar:
    st.subheader("Data")
    analyte = st.selectbox("Analyte", names, index=follow_index(names))
    level_names = [lv.level for lv in i.calibration_design]
    use_levels = st.multiselect("Levels", level_names, default=level_names)
    reps = st.slider("Replicates per level", 1, 3, 1)
    prop = st.slider("Proportional noise per injection (%)", 0.0, 10.0, 100 * i.acquisition.injection_rsd, 0.5)
    dead = st.select_slider("Detector dead time (s)", options=[0.0, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6], value=0.0)
    log_axes = st.checkbox("Log axes", value=True)


def config(label, default_w):
    with st.sidebar:
        st.subheader(f"Fit {label}")
        m = st.selectbox(f"Model {label}", list(CalibrationModel), format_func=lambda x: x.value, key=f"m{label}")
        w = st.selectbox(f"Weighting {label}", list(Weighting), index=list(Weighting).index(default_w),
                         format_func=lambda x: x.value, key=f"w{label}")
        o = st.selectbox(f"Origin {label}", list(OriginHandling), format_func=lambda x: x.value, key=f"o{label}")
    return m, w, o


cfg = {"A": config("A", Weighting.inv_x), "B": config("B", Weighting.none)}

a = next(x for x in i.analytes if x.name == analyte)
cond = LCConditions.from_method(i.system, i.lc_method)
tR = retention_time(a.log_kw, a.S, cond)
sigma = peak_sigma(a.log_kw, a.S, tR, cond, i.system.plate_number, i.system.extra_column_sigma_s)
trs = [t for t in i.transitions if t.analyte == analyte and t.role is TransitionRole.quantifier]
acq = i.acquisition.model_copy(update={"injection_rsd": prop / 100, "detector_dead_time_s": dead or None})
sched = build_schedule(i.transitions, acq, i.lc_method.run_time_min)
j = next(k for k, t in enumerate(i.transitions) if t.analyte == analyte and t.role is TransitionRole.quantifier)
sched_one = type(sched)(sched.cycle_starts, sched.active[:, [j]], sched.n_active, sched.dwell_ms,
                        sched.sample_times[:, [j]], sched.cycle_time_ms)
inj_ul = i.prep.injection_volume_ul.nominal
frac = a.isomer_fraction or 1.0

xs, ys = [], []
for lv in i.calibration_design:
    if lv.level not in use_levels:
        continue
    c = lv.concentrations_ng_ml[analyte]
    area = a.response_factor * c * frac * inj_ul / 1000.0
    for _ in range(reps):
        sig = acquire([PeakSpec(analyte, tR, sigma, area)], trs, sched_one, acq, rng)[0]
        xs.append(c)
        ys.append(sig.area)
if len(set(xs)) < 3:
    st.error("Pick at least three levels.")
    st.stop()

x, y = np.array(xs), np.array(ys)
fits = {k: fit_calibration(x, y, *v) for k, v in cfg.items()}
grid = np.geomspace(x.min() * 0.8, x.max() * 1.1, 200) if log_axes else np.linspace(0, x.max() * 1.1, 200)

fig = make_subplots(rows=1, cols=2, subplot_titles=("Response", "Back-calculated accuracy (%)"))
fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Injections", marker=dict(color="#F3ECDD", size=7)), 1, 1)
colors = {"A": "steelblue", "B": "darkorange"}
for k, f in fits.items():
    fig.add_trace(go.Scatter(x=grid, y=f.predict(grid), name=f"Fit {k}", line=dict(color=colors[k])), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=f.accuracy_pct, mode="markers+lines", name=f"Accuracy {k}",
                             line=dict(color=colors[k], dash="dot"), showlegend=False), 1, 2)
ACC = acceptance()
LO, HI = ACC.get("calibration_point_recovery_pct", [80, 120])
R2_MIN = ACC.get("calibration_r2_min", 0.99)
for yv in (LO, HI):
    fig.add_hline(y=yv, line_dash="dash", line_color="gray", row=1, col=2)
fig.add_hline(y=100, line_color="lightgray", row=1, col=2)
axis_type = "log" if log_axes else "linear"
fig.update_xaxes(type=axis_type, title_text="Vial concentration (ng/mL)", row=1, col=1)
fig.update_yaxes(type=axis_type, title_text="Area", row=1, col=1)
fig.update_xaxes(type=axis_type, title_text="Vial concentration (ng/mL)", row=1, col=2)
fig.update_layout(height=420, template="room", margin=dict(t=40, b=40), legend=dict(orientation="h", y=-0.2))
chart(fig, width="stretch")

cols = st.columns(2)
for col, (k, f) in zip(cols, fits.items()):
    acc = np.array(f.accuracy_pct)
    fail = int(((acc < LO) | (acc > HI)).sum())
    with col:
        st.markdown(f"**Fit {k}**: {f.model.value}, weighting {f.weighting.value}, origin {f.origin.value}")
        st.metric("R²", f"{f.r2:.4f}", delta="pass" if f.r2 >= R2_MIN else f"fails {R2_MIN:g}", delta_color="off")
        st.metric(f"Levels outside {LO:g}-{HI:g}%", fail)
        st.caption("Coefficients (intercept, slope[, curvature]): " + ", ".join(f"{c:.4g}" for c in f.coef))

table = pd.DataFrame({"Vial ng/mL": x, "Area": np.round(y, 0),
                      **{f"Accuracy {k} (%)": np.round(f.accuracy_pct, 1) for k, f in fits.items()}})
st.dataframe(table, hide_index=True, width="stretch")
st.caption(f"{analyte}: RT {tR:.2f} min, response factor {a.response_factor:.3g} area/ng (assumed). "
           f"Acceptance ({ACC.get('source', method().short)}): R² >= {R2_MIN:g}, every level {LO:g}-{HI:g}%.")
sources_line(("knoll_2010", "almeida_2002", "gu_2014", *method_sources()),
             label="Sources (counting noise and dead time in the detector model; weighting and accuracy at the low end; "
                   "the acceptance criteria are the example method's own; the response factor is assumed)")
provenance_table(method_provenance(i))
