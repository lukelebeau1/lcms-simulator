"""Analytical, MS: how cycle time, windows and transition count set dwell, sampling and ion ratio precision."""

import dataclasses
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

import style  # noqa: E402
from common import (chart, Station, follow_index, inputs, method_provenance, method_sources, page, provenance_table,  # noqa: E402
                    sources_line)

from lcms_sim.faults.library import LIBRARY, Scenario  # noqa: E402
from lcms_sim.io.schemas import TransitionRole  # noqa: E402
from lcms_sim.lc.gradient import LCConditions  # noqa: E402
from lcms_sim.lc.peaks import gaussian, peak_sigma  # noqa: E402
from lcms_sim.lc.retention import retention_time  # noqa: E402
from lcms_sim.ms.acquisition import build_schedule, cycle_layout, points_across_peak  # noqa: E402
from lcms_sim.ms.crosstalk import crosstalk_map  # noqa: E402
from lcms_sim.ms.noise import expected_counted_ions  # noqa: E402
from lcms_sim.ms.ratios import counting_limit_rsd  # noqa: E402
from lcms_sim.paths import ROOT  # noqa: E402
from lcms_sim.pipeline import RunConditions, lc_peaks, run_sequence  # noqa: E402
from lcms_sim.quant.prep import analyte_vial_conc, on_column_ng, prep_values  # noqa: E402
from lcms_sim.rng import make_rng  # noqa: E402

page("MS: acquisition",
     "Dynamic MRM shares a fixed cycle among the transitions whose windows are open. "
     "Shorter dwell means fewer ions counted, so noisier areas and ion ratios; the mean does not move.", station=Station.detector)

i = inputs()
acq0 = i.acquisition
with st.sidebar:
    st.subheader("Acquisition")
    cycle = st.slider("Cycle time (ms)", 100, 1000, int(acq0.cycle_time_ms), 10)
    wscale = st.slider("Window width x", 0.25, 3.0, 1.0, 0.05, help="Multiplies every retention window")
    extra = st.slider("Extra transitions from other methods", 0, 300, 0, 10)
    st.subheader("Ion ratio precision")
    frac_limit = st.slider("Concentration (x action limit)", 0.05, 2.0, 0.5, 0.05)

scn = Scenario(i)
if extra:
    scn = LIBRARY["too_many_transitions"].apply(scn, extra)
trs = [t.model_copy(update={"window_min": t.window_min * wscale}) for t in scn.inputs.transitions]
acq = acq0.model_copy(update={"cycle_time_ms": float(cycle)})
try:
    sched = build_schedule(trs, acq, i.lc_method.run_time_min)
except ValueError as e:
    st.error(f"{e}. Lengthen the cycle or narrow the windows.")
    st.stop()

# Concurrency and dwell along the run
fig = make_subplots(specs=[[{"secondary_y": True}]])
busy = sched.n_active > 0
fig.add_trace(go.Scatter(x=sched.cycle_starts, y=sched.n_active, name="Concurrent transitions",
                         line=dict(color=style.LILAC)), secondary_y=False)
fig.add_trace(go.Scatter(x=sched.cycle_starts, y=np.where(busy, sched.dwell_ms, np.nan), name="Dwell (ms)",
                         line=dict(color=style.SAFFRON)), secondary_y=True)
fig.update_yaxes(title_text="Concurrent", secondary_y=False)
fig.update_yaxes(title_text="Dwell (ms)", type="log", tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
                 showgrid=False, secondary_y=True)
fig.update_layout(height=320, margin=dict(t=10, b=40), xaxis_title="Time (min)", template="room",
                  legend=dict(orientation="h", y=1.12))
chart(fig, width="stretch")
sources_line(("agilent_5990_3595", "knoll_2010"),
             label="Sources (dynamic MRM shares the cycle; counting noise falls as 1 / sqrt N; the polarity-switch "
                   "overhead is a user setting)")


# Per analyte: points across the peak, dwell at the apex, counted ions and ion ratio precision
mn = pd.read_csv(ROOT / "data" / "reference" / "mn_ocm_cta_v2_analytes.csv")
limit = {r.analyte: r.limit_flower * (1000 if r.unit == "ppm" else 1) for r in mn.itertuples()}
cond = LCConditions.from_method(i.system, i.lc_method)
nom = prep_values(i.prep, "nominal")
quant_j = {t.analyte: j for j, t in enumerate(trs) if t.role is TransitionRole.quantifier}
qual_rel = {}
for t in trs:
    if t.role is TransitionRole.qualifier and t.analyte not in qual_rel:
        qual_rel[t.analyte] = t.relative_response or 0.5

rows = []
for a in i.analytes:
    if a.is_internal_standard:
        continue
    tR = retention_time(a.log_kw, a.S, cond)
    sig = peak_sigma(a.log_kw, a.S, tR, cond, i.system.plate_number, i.system.extra_column_sigma_s)
    j = quant_j[a.name]
    on = sched.active[:, j]
    k = np.searchsorted(sched.cycle_starts, tR) - 1
    dwell = float(sched.dwell_ms[k]) if 0 <= k < len(sched.dwell_ms) and on[k] else 0.0
    c = limit.get(a.regulated_as, math.nan) * frac_limit * (a.isomer_fraction or 1.0)
    area = a.response_factor * on_column_ng(analyte_vial_conc(c, 1.0, nom), nom)
    n_q = area / acq.gain_area_per_ion * dwell / cycle if dwell > 0 else 0.0
    n_ql = n_q * qual_rel.get(a.name, 0.5)
    rsd = counting_limit_rsd(n_q, n_ql) if n_ql > 0 else math.inf
    p_fail = math.erfc(0.30 / (rsd * math.sqrt(2))) if math.isfinite(rsd) else 1.0
    rows.append({"Analyte": a.name, "RT (min)": round(tR, 2), "Points across peak": round(points_across_peak(sig, cycle), 1),
                 "Dwell at apex (ms)": round(dwell, 1), "Ions counted (quant)": int(n_q),
                 "Ion ratio RSD (%)": round(100 * rsd, 1), "Chance ratio outside +-30% (%)": round(100 * p_fail, 1)})
df = pd.DataFrame(rows).sort_values("RT (min)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Max concurrent", int(sched.n_active.max()))
c2.metric("Shortest dwell", f"{sched.dwell_ms[busy].min():.1f} ms")
low = df[df["Points across peak"] < acq.min_points_per_peak]
c3.metric(f"Peaks under {acq.min_points_per_peak} points", f"{len(low)} of {len(df)}")
risky = df[df["Chance ratio outside +-30% (%)"] > 5]
c4.metric("Ion ratios at risk (>5% fail)", f"{len(risky)} of {len(df)}")

bar = go.Figure(go.Bar(x=df["Analyte"], y=df["Points across peak"], marker_color=np.where(
    df["Points across peak"] < acq.min_points_per_peak, style.SAFFRON, style.TEAL_LIGHT)))
bar.add_hline(y=acq.min_points_per_peak, line_dash="dash", line_color=style.PALE)
bar.update_layout(height=300, margin=dict(t=10, b=120), yaxis_title="Points across peak (4 sigma / cycle)",
                  template="room")
chart(bar, width="stretch")
st.dataframe(df, hide_index=True, width="stretch", height=380)
st.caption("Ion ratio RSD is the counting limit, sqrt(1/N_quant + 1/N_qual); real scatter is at least this. "
           "Concentrations are the stated fraction of each analyte's MN action limit, in flower.")
sources_line(("knoll_2010", "sante_11312_2021", "waters_720001168", "zeng_2023", "sciex_scheduled_mrm", "mn_ocm_cta_v2"),
             label="Sources (the ratio's counting limit; the +-30% tolerance; the points-per-peak conventions behind "
                   "the flag at 10, a chosen threshold; action limits)")

# Inside one cycle: what Q1 and Q3 pass, one transition at a time
st.subheader("Inside one cycle")
st.caption("The quadrupoles pass one precursor > product pair at a time. Every compound eluting at that moment "
           "keeps sending ions; only the one being measured is counted. Blue: eluting now. Grey: window open, "
           "nothing eluting. Dark: polarity switch.")
sources_line(("yost_enke_1978", "agilent_5990_3595", "knoll_2010"),
             label="Sources (one pair at a time through Q1 and Q3; dwell and inter-scan delay; ions counted = "
                   "area / gain x dwell / cycle)")
with st.sidebar:
    st.subheader("Inside one cycle")
    t_now = st.slider("Moment in the run (min)", 0.0, float(i.lc_method.run_time_min), 7.0, 0.05)
    levels = [lv.level for lv in i.calibration_design]
    lvl = st.selectbox("Vial", levels, index=follow_index([f"Column: {lv}" for lv in levels], "injection"))

lay = cycle_layout(sched, acq, t_now)
if not lay.slots:
    st.info(f"No transitions are scheduled at {t_now:.2f} min.")
else:
    level = next(lv for lv in i.calibration_design if lv.level == lvl)
    inj_ul = i.prep.injection_volume_ul.nominal
    ng = {a.name: level.concentrations_ng_ml.get(a.name, 0.0) * (a.isomer_fraction or 1.0) * inj_ul / 1000
          for a in i.analytes}
    if i.prep.istd_spike_ul is not None and i.prep.vial_volume_ml is not None:
        ng |= {n: m * i.prep.istd_spike_ul.nominal / i.prep.vial_volume_ml.nominal * inj_ul / 1000
               for n, m in i.prep.istd_mix_ng_ul.items()}
    pk = {p.analyte: p for p in lc_peaks(i, ng, cond)}

    def eluting(name: str) -> bool:
        p = pk.get(name)
        return p is not None and p.quant_area > 0 and abs(t_now - p.center_min) < 3 * p.sigma_min

    def ions_per_s(name: str, rel: float = 1.0) -> float:
        """Ions of this transition reaching the detector now, if it were measured continuously."""
        p = pk[name]
        return float(gaussian(t_now, p.center_min, p.sigma_min, p.quant_area * rel)) / acq.gain_area_per_ion / 60.0

    tc = lay.cycle_time_ms
    k = st.slider("Step through the cycle (transition)", 1, len(lay.slots), 1)
    cur = lay.slots[k - 1]
    tr_cur = trs[cur.transition_index]

    strip = go.Figure()
    for x in lay.slots:
        tr = trs[x.transition_index]
        colour = "crimson" if x is cur else "steelblue" if eluting(tr.analyte) else "lightgray"
        strip.add_trace(go.Bar(
            y=[""], x=[x.dwell_ms], base=[x.start_ms], orientation="h", marker_color=colour, showlegend=False,
            hovertemplate=f"{tr.analyte} {tr.role.value}<br>{tr.precursor_mz} > {tr.product_mz}<br>"
                          f"{x.start_ms:.1f}-{x.start_ms + x.dwell_ms:.1f} ms<extra></extra>"))
    for b, e in lay.switches_ms:
        strip.add_trace(go.Bar(y=[""], x=[e - b], base=[b], orientation="h", marker_color="#7A6690", showlegend=False,
                               hovertemplate="Polarity switch<extra></extra>"))
    strip.update_layout(height=150, margin=dict(t=10, b=40), xaxis_title="ms into the cycle",
                        xaxis_range=[0, tc], barmode="overlay", template="room")
    chart(strip, width="stretch")

    on = eluting(tr_cur.analyte)
    rate = ions_per_s(tr_cur.analyte, tr_cur.relative_response or 1.0) if on else 0.0
    st.markdown(f"**{cur.start_ms:.1f} ms: Q1 {tr_cur.precursor_mz} > Q3 {tr_cur.product_mz}**, "
                f"{tr_cur.analyte} {tr_cur.role.value} ({tr_cur.polarity.value})"
                + ("" if on else ". Window open, nothing eluting."))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dwell", f"{cur.dwell_ms:.1f} ms")
    c2.metric("Share of cycle", f"{100 * cur.dwell_ms / tc:.1f}%")
    c3.metric("Ions/s arriving", f"{rate:,.0f}")
    c4.metric("Ions counted", f"{rate * cur.dwell_ms / 1000:,.1f}")

    # Every compound eluting now: what reaches the detector and what share of it this schedule counts.
    rows_c = []
    for name in sorted({trs[x.transition_index].analyte for x in lay.slots if eluting(trs[x.transition_index].analyte)},
                       key=lambda n: pk[n].center_min):
        p = pk[name]
        n_peak = expected_counted_ions(p.quant_area, acq.gain_area_per_ion, cur.dwell_ms, tc)
        rows_c.append({"Compound": name, "Measured now": name == tr_cur.analyte,
                       "RT (min)": round(p.center_min, 2),
                       "Quantifier ions/s arriving": round(ions_per_s(name)),
                       "Share counted (%)": round(100 * cur.dwell_ms / tc, 1),
                       "Ions counted over the peak": int(n_peak),
                       "Counting scatter (%)": round(100 / math.sqrt(n_peak), 1) if n_peak > 0 else math.inf})
    if rows_c:
        st.dataframe(pd.DataFrame(rows_c), hide_index=True, width="stretch")
    else:
        st.caption("No compound is eluting at this moment; every transition's dwell is spent waiting.")
    st.caption(f"{lvl}, noise-free expected peaks. Ions counted over the peak = area / gain x dwell / cycle; "
               "counting scatter = 1 / sqrt(N).")

# Crosstalk: another compound's ions through both filters of a transition
st.subheader("Crosstalk between transitions")
st.caption("A transition counts every ion that passes its Q1 setting and, after fragmenting, its Q3 setting. "
           "A co-eluting compound with the same precursor m/z and a product within Q3's window adds to it, "
           "whatever the dwell: the dwell decides which pair is measured, not which compound can pass it.")
sources_line(("hoffmann_stroobant_2007", "dawson_1995", *method_sources(), "sante_11312_2021",
              "ich_m10_2022"),
             label="Sources (both filters must pass; unit-resolution windows from the method, their Gaussian shape "
                   "assumed; the pairs from the method's transitions; the +-30% tolerance; the guideline's selectivity check for the IS "
                   "channel; the yield is assumed until measured)")
xmap = crosstalk_map(i.transitions, acq.q1_fwhm_mz, acq.q3_fwhm_mz)
tR_all = {a.name: retention_time(a.log_kw, a.S, cond) for a in i.analytes}
sg_all = {a.name: peak_sigma(a.log_kw, a.S, tR_all[a.name], cond, i.system.plate_number,
                             i.system.extra_column_sigma_s) for a in i.analytes}
xrows = []
for j, cs in xmap.items():
    t = i.transitions[j]
    for c in cs:
        if c.factor < 1e-3:
            continue
        sep = abs(tR_all[t.analyte] - tR_all[c.source]) / math.hypot(sg_all[t.analyte], sg_all[c.source])
        xrows.append({"j": j, "Transition": f"{t.analyte} {t.precursor_mz:g} > {t.product_mz:g} ({t.role.value})",
                      "Leaks in from": c.source, "Through both windows": c.factor,
                      "Elution apart (peak widths)": sep})
if not xrows:
    st.info("No two transitions in this method share a precursor and a product within the windows.")
else:
    is_istd = {a.name for a in i.analytes if a.is_internal_standard}
    xdf = pd.DataFrame(xrows)
    xdf["_istd"] = [i.transitions[j].analyte in is_istd for j in xdf.j]
    xdf = xdf.sort_values(["Elution apart (peak widths)", "_istd"]).drop(columns="_istd")   # meeting pairs, targets first
    st.dataframe(xdf.drop(columns="j").style.format({"Through both windows": "{:.3f}",
                                                    "Elution apart (peak widths)": "{:.1f}"}),
                 hide_index=True, width="stretch")
    st.caption(f"Through both windows: the share of the leaking compound's listed fragment that this transition "
               f"passes, with {acq.q1_fwhm_mz:g} m/z Q1 and Q3 windows (unit resolution, assumed), times its "
               "relative response in the method (a placeholder). Pairs more than ~3 peak widths apart never meet.")

    c1, c2 = st.columns(2)
    j_pick = c1.selectbox("Transition to check", list(xdf.j), format_func=lambda j: xdf.set_index("j").Transition[j])
    y_log = c2.slider("Crosstalk yield (log10)", -5.0, 0.0, -3.0, 0.25,
                      help="Share of the leaking compound's fragment that really forms under this transition's "
                           "collision energy, relative to its own listed ratio. Unknown until the leaking compound "
                           "is injected alone; 0 = -inf.")
    yld = 10.0 ** y_log
    t_pick = i.transitions[j_pick]

    @st.cache_resource(show_spinner="Running the sequence with crosstalk")
    def xt_ratios(y: float) -> dict[str, dict[int, float]]:
        """Qualifier / quantifier for every injection, relative to the expected ratio, noise-free."""
        inp = dataclasses.replace(i, acquisition=i.acquisition.model_copy(update={"crosstalk": True,
                                                                                  "crosstalk_yield": y}))
        r = run_sequence(inp, make_rng(0), conditions=RunConditions(detector="expected"))
        out = {}
        for rec in r.injections:
            row = {}
            for jj, area in rec.qualifier_areas.items():
                tt = i.transitions[jj]
                q = rec.areas.get(tt.analyte, 0.0)
                if q > 0 and tt.relative_response:
                    row[jj] = area / q / tt.relative_response
            out[rec.item.ref] = row
        return out

    now, full = xt_ratios(yld), xt_ratios(1.0)
    order = [lv.level for lv in i.calibration_design] + [s.id for s in i.samples]
    pts = [(ref, now[ref][j_pick], full[ref][j_pick]) for ref in order if j_pick in now.get(ref, {})]
    if t_pick.role is not TransitionRole.qualifier or not pts:
        st.info("This transition is a quantifier; crosstalk there biases the area itself rather than an ion ratio.")
    else:
        refs, vals, full_vals = zip(*pts)
        bar_c = ["firebrick" if abs(v - 1) > 0.30 else "steelblue" for v in vals]
        xfig = go.Figure(go.Bar(x=list(refs), y=list(vals), marker_color=bar_c,
                                hovertemplate="%{x}: %{y:.3g} x the expected ratio<extra></extra>"))
        xfig.add_hrect(y0=0.7, y1=1.3, fillcolor="seagreen", opacity=0.12, line_width=0)
        xfig.update_layout(height=320, margin=dict(t=10, b=60), template="room",
                           yaxis_title="Observed / expected ion ratio", yaxis_type="log")
        chart(xfig, width="stretch")
        cal1 = dict(zip(refs, full_vals)).get(i.calibration_design[0].level)
        threshold = 0.30 / (cal1 - 1) if cal1 and cal1 > 1 else None
        n_fail = sum(1 for v in vals if abs(v - 1) > 0.30)
        m = st.columns(3)
        m[0].metric("Yield", f"{yld:.2g}")
        m[1].metric("Injections failing +-30%", f"{n_fail} of {len(vals)}")
        m[2].metric(f"{i.calibration_design[0].level} fails above", f"{threshold:.2g}" if threshold else "-",
                    help="The yield at which the lowest calibrator's ion ratio leaves +-30%")
        st.caption(f"Noise-free run of the whole sequence with crosstalk on. The leak is about the same in every "
                   f"vial when it comes from an internal standard, so it matters most where {t_pick.analyte} is "
                   f"lowest. Check: inject the leaking compound alone and read this transition; the ratio of that "
                   f"area to the compound's own qualifier gives the yield.")

provenance_table(method_provenance(i))
