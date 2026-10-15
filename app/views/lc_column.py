"""Analytical, LC: inside the column. Every band's position and the %B around it, played through the run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

import c18_scene  # noqa: E402
import molecules  # noqa: E402
from common import (Station, follow_index, inputs, method, method_provenance, page, plotly_spec,  # noqa: E402
                    provenance_table, sources_line)

from lcms_sim.lc.column import band_sigma_fraction, band_trajectory, phi_in_column  # noqa: E402
from lcms_sim.lc.gradient import LCConditions  # noqa: E402
from lcms_sim.lc.retention import k_lss, retention_time  # noqa: E402
from lcms_sim.paths import ROOT  # noqa: E402

page("LC: inside the column",
     "Every compound is injected at the column head at time zero. A band moves at the mobile-phase speed "
     "divided by (1 + k), and k depends on the %B around it, so bands sit at the head until the gradient "
     "reaches them, then accelerate. Press Play to run it from the time set in the sidebar.", station=Station.column)

i = inputs()
cond = LCConditions.from_method(i.system, i.lc_method)
L = i.system.column_length_mm
N = i.system.plate_number
names_t = [a.name for a in i.analytes if not a.is_internal_standard]
SPEED_MS = {"Slow": 250, "Normal": 120, "Fast": 50}

with st.sidebar:
    st.subheader("Show")
    with_is = st.checkbox("Internal standards", value=False)
    focus = st.selectbox("Highlight", names_t, index=follow_index(names_t))
    st.subheader("Playback")
    speed = st.select_slider("Speed", list(SPEED_MS), value="Normal")
    step = st.select_slider("Time step (min)", [0.05, 0.1, 0.2], value=0.1)
COLOUR = {"At the head": "darkorange", "Moving": "steelblue", "Eluted": "lightgray"}
# Band state as a number on a discrete colour scale: one numeric array per frame, cheap to build and send.
STATE_SCALE = [[0.0, "darkorange"], [0.33, "darkorange"], [0.34, "steelblue"], [0.66, "steelblue"],
               [0.67, "lightgray"], [1.0, "lightgray"]]


@st.cache_resource(show_spinner="Computing band positions")
def build(with_is: bool, focus: str, step: float) -> tuple[dict, list[dict]]:
    """The figure as a plain spec, and every frame precomputed. Cached; never mutated by callers."""
    times = np.round(np.arange(0.0, i.lc_method.run_time_min + 1e-9, step), 3)
    shown = sorted((a for a in i.analytes if with_is or not a.is_internal_standard),
                   key=lambda a: retention_time(a.log_kw, a.S, cond))
    names = [a.name for a in shown]
    tR = np.array([retention_time(a.log_kw, a.S, cond) for a in shown])
    X = np.array([band_trajectory(a.log_kw, a.S, cond, times) for a in shown])      # (compounds, frames)
    PHI = phi_in_column(cond, X, times[None, :])
    K = np.array([k_lss(PHI[j], a.log_kw, a.S) for j, a in enumerate(shown)])
    ELUTED = X >= 1.0
    SIGMA = np.vectorize(band_sigma_fraction)(X, N) * L
    xs = np.linspace(0.0, 1.0, 201)
    fi = names.index(focus) if focus in names else None


    def frame_data(f: int):
        x, el = X[:, f], ELUTED[:, f]
        state = np.where(el, 2, np.where(x < 0.02, 0, 1))
        pos = np.where(el, L * 1.04, x * L)
        speed_mm = np.where(el, np.nan, L / (cond.t0 * (1 + K[:, f])))
        bands = dict(type="scatter", x=pos, y=names, mode="markers",
                     marker=dict(color=state, colorscale=STATE_SCALE, cmin=0, cmax=2, size=8),
                     error_x=dict(type="data", array=np.where(el, 0, 2 * SIGMA[:, f]), thickness=4, width=0,
                                  color="rgba(70,130,180,0.5)"),
                     customdata=np.stack([tR, np.where(el, np.nan, K[:, f]), np.where(el, np.nan, 100 * PHI[:, f]),
                                          speed_mm], axis=1),
                     hovertemplate="%{y}<br>%{x:.1f} mm<br>%B here %{customdata[2]:.0f}%, k %{customdata[1]:.3g}"
                                   "<br>speed %{customdata[3]:.3g} mm/min<br>elutes at %{customdata[0]:.2f} min"
                                   "<extra></extra>", showlegend=False)
        profile = dict(type="scatter", x=xs * L, y=100 * phi_in_column(cond, xs, times[f]))
        hi = dict(type="scatter", x=[pos[fi]] if fi is not None else [], y=[focus] if fi is not None else [])
        head = int(((x < 0.02) & ~el).sum())
        title = (f"{times[f]:.2f} min   |   %B at inlet {100 * float(cond.phi_inlet(times[f])):.0f}%, "
                 f"at outlet {100 * float(phi_in_column(cond, 1.0, times[f])):.0f}%   |   "
                 f"at the head {head}, eluted {int(el.sum())} of {len(names)}")
        if fi is not None:
            title += (f"<br>{focus}: eluted at {tR[fi]:.2f} min" if el[fi] else
                      f"<br>{focus}: {x[fi] * L:.1f} mm along in {100 * PHI[fi, f]:.0f}% B, k {K[fi, f]:.3g}, "
                      f"{L / (cond.t0 * (1 + K[fi, f])):.3g} mm/min")
        return [profile, bands, hi], title


    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.12, 0.88], vertical_spacing=0.02)
    (profile, bands, hi), title0 = frame_data(0)
    fig.add_trace(go.Scatter(profile, mode="lines", fill="tozeroy", line=dict(color="gray"), name="%B",
                             hovertemplate="%{x:.0f} mm: %{y:.0f}% B<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(bands), row=2, col=1)
    fig.add_trace(go.Scatter(hi, mode="markers", name=focus, marker=dict(color="crimson", size=13, symbol="diamond"),
                             hoverinfo="skip"), row=2, col=1)
    for s, c in COLOUR.items():   # legend only
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color=c, size=8), name=s), row=2, col=1)

    # Frames are plain dicts and the figure is rebuilt without per-property validation: the static
    # traces above were validated already, and validating ~150 frames costs over a second per rerun.
    frames = []
    for f in range(len(times)):
        data, title = frame_data(f)
        frames.append(dict(name=f"{times[f]:.2f}", data=data, traces=[0, 1, 2], layout=dict(title=dict(text=title))))

    fig.update_layout(
        title=dict(text=title0, font=dict(size=13), x=0.0, y=0.995, yanchor="top"),
        height=max(560, 15 * len(names) + 260), margin=dict(t=95, b=120), template="plotly_white",
        legend=dict(orientation="h", y=1.0, x=1.0, xanchor="right", yanchor="bottom"),
    )
    fig.add_vline(x=L, line_dash="dash", line_color="gray")
    fig.update_xaxes(range=[-0.02 * L, 1.08 * L], row=2, col=1,
                     title_text=f"Position along the column (mm): inlet 0, outlet {L:g}; eluted bands sit right of the outlet")
    fig.update_yaxes(title_text="%B", range=[0, 100], row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=10), autorange="reversed", row=2, col=1)
    return fig.to_dict(), frames


spec, frames = build(with_is, focus, step)
with st.sidebar:
    t_jump = st.slider("Jump to time (min)", 0.0, float(frames[-1]["name"]), 0.0, float(step),
                       help="Shows that moment; Play runs on from there.")
f = min(range(len(frames)), key=lambda k: abs(float(frames[k]["name"]) - t_jump))
# Show frame f: copy only what changes, so the cached spec stays untouched.
shown_spec = dict(spec, data=list(spec["data"]), layout=dict(spec["layout"]))
for k, d in zip(frames[f]["traces"], frames[f]["data"]):
    shown_spec["data"][k] = {**spec["data"][k], **d}
shown_spec["layout"]["title"] = {**spec["layout"]["title"], "text": frames[f]["layout"]["title"]["text"]}
# The frames travel inside the Play button, not as registered figure frames: Streamlit redraws the
# chart for fullscreen and registered frames do not survive that, while the layout does.
dur = SPEED_MS[speed]
play = dict(label="Play", method="animate",
            args=[frames[f:], dict(frame=dict(duration=dur, redraw=True), transition=dict(duration=0), mode="immediate")])
pause = dict(label="Pause", method="animate",
             args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
shown_spec["layout"]["updatemenus"] = [dict(type="buttons", direction="left", buttons=[play, pause], x=0.0, y=-0.02,
                                            xanchor="left", yanchor="top", showactive=False, pad=dict(t=30))]
plotly_spec(shown_spec, width="stretch")

st.caption(f"t0 = {cond.t0:.2f} min, dwell {cond.tD:.2f} min: the gradient reaches the column head "
           f"{cond.tD:.2f} min after the pump starts it and takes {cond.t0:.2f} min to cross the column. "
           f"Bars are +-2 sigma from the plate count (N = {N:g}); gradient compression and extra-column "
           "broadening are left out here. docs/lc.md.")

# --- Why it's retained: the molecules ------------------------------------------------------------
st.subheader("Why it's retained: the molecules")
st.caption("Drag to rotate, scroll to zoom (beside it, the usual 2D diagram). Coloured by each atom's share of log P (Wildman-Crippen): orange parts "
           "prefer the C18 chains and hold the molecule back, blue parts prefer the water/methanol and pull it "
           "forward. The table adds the shares up by functional group. One 3D shape is shown; real molecules flex.")
sources_line(("wildman_crippen_1999", "cheng_2007", "dorsey_dill_1989", "rafferty_2007"))
mt = molecules.table()
mol_names = [n for n in mt.index if n in {a.name for a in i.analytes}]
xlogp = pd.read_csv(ROOT / "data" / "reference" / "pesticide_xlogp.csv").set_index("compound")
quant = {t.analyte: t for t in i.transitions if t.role.value == "quantifier"}
mc1, mc2 = st.columns(2)
pick_a = mc1.selectbox("Molecule", mol_names, index=mol_names.index(focus) if focus in mol_names else 0,
                       help="Follows the Highlight choice in the sidebar")
pick_b = mc2.selectbox("Compare with", ["(none)"] + mol_names, index=0)
mo = st.columns([1.3, 1.1, 0.6, 0.8, 0.8])
as_ion = mo[0].radio("Show", ["Molecule", "Ion the method selects"], horizontal=True)
colour = mo[1].radio("Colour", ["Hydrophobicity", "Element"], horizontal=True)
spin = mo[2].toggle("Spin", value=False)
atom_labels = mo[3].toggle("Label atoms", value=True, help="Every atom that is not C or H")
labels = mo[4].toggle("Label groups", value=False)


def measured_rt(name: str) -> tuple[float, str]:
    """The retention time the method lists for the compound (its quantifier's), and where it comes from."""
    return float(quant[name].retention_time_min), method().short


def svg_html(svg: str) -> str:
    """Inline SVG for st.markdown (drop the XML declaration; scale to the column)."""
    body = svg[svg.index("<svg"):]
    return f'<div style="width:100%;max-width:520px">{body.replace("<svg ", '<svg style="width:100%;height:auto" ', 1)}</div>'


def show(name: str, where, side_by_side: bool) -> None:
    m = molecules.load(name)
    contrib, grp = molecules.contributions(m), molecules.groups(m)
    t = quant.get(name)
    kind, nd, iso = molecules.precursor_type(name, t.precursor_mz, t.polarity.value) if t else ("unknown", 0, 0)
    shown, added = (molecules.make_ion(m, kind) if as_ion != "Molecule" else (m, []))
    pad = np.r_[contrib, np.zeros(max(0, shown.GetNumAtoms() - len(contrib)))][:shown.GetNumAtoms()]
    html = molecules.viewer_html(shown, "hydrophobicity" if colour == "Hydrophobicity" else "element", added, spin,
                                 labels, 380, contrib=pad, label_groups=grp, atom_labels=atom_labels)
    flat = svg_html(molecules.draw_2d(shown))
    rt, src = measured_rt(name)
    with where:
        st.markdown(f"**{name}**")
        if side_by_side:
            v3, v2 = st.columns([0.58, 0.42])
            with v3:
                components.html(html, height=390)
            v2.markdown(flat, unsafe_allow_html=True)
        else:
            components.html(html, height=390)
            st.markdown(flat, unsafe_allow_html=True)
        ion_txt = f"{kind} at m/z {t.precursor_mz:g}" if t else "no quantifier"
        if kind == "[M+H-HF]+":
            ion_txt += " (HF lost in the source; the protonated molecule is drawn)"
        if iso:
            ion_txt += f" (the M+{2 * iso} peak, from 37Cl / 81Br)"
        if t is not None and t.polarity.value == "negative":
            ion_txt += ", negative mode"
        st.caption(f"Ion: {ion_txt}. Magenta: the added proton or adduct (or the atom that lost its proton).")
        k = st.columns(3)
        k[0].metric("Measured RT", f"{rt:.2f} min", help=src)
        k[1].metric("XLogP3", f"{xlogp.loc[name, 'xlogp']:.1f}" if name in xlogp.index else "-",
                    help="PubChem's log P estimate, used by the retention-by-analogy fit")
        k[2].metric("Crippen log P", f"{molecules.crippen_logp(m):.1f}", help="Sum of the atom colours")
        st.dataframe(pd.DataFrame({"group": [g.name for g in grp], "log P share": [round(g.logp, 2) for g in grp],
                                   "atoms": [sum(1 for j in g.atoms if m.GetAtomWithIdx(j).GetAtomicNum() > 1)
                                             for g in grp]}),
                     hide_index=True, width="stretch",
                     column_config={"log P share": st.column_config.NumberColumn(format="%+.2f")})
        notes = [s for s in (str(mt.loc[name, "note"]) if mt.loc[name, "note"] == mt.loc[name, "note"] else "",
                             "3D built with RDKit (PubChem has no 3D conformer)"
                             if mt.loc[name, "coords"].startswith("rdkit") else "") if s]
        if nd:
            notes.append(f"drawn without its {nd} deuterium (positions unknown)")
        if notes:
            st.caption("Note: " + "; ".join(notes) + ".")


if pick_b == "(none)":
    show(pick_a, st.container(), side_by_side=True)
else:
    ca, cb = st.columns(2)
    show(pick_a, ca, side_by_side=False)
    show(pick_b, cb, side_by_side=False)

# Where the molecule sits on retention against hydrophobicity, all targets.
rt_listed = pd.DataFrame({"rt_median_min": {t.analyte: t.retention_time_min for t in quant.values()}})
xt = xlogp.join(rt_listed, how="inner")
fr = go.Figure(go.Scatter(x=xt.xlogp, y=xt.rt_median_min, mode="markers", text=xt.index, hoverinfo="text+x+y",
                          marker=dict(color="#bbb", size=7), name="method compounds"))
for nm, colr in ((pick_a, "crimson"), (pick_b, "royalblue")):
    if nm in xt.index:
        fr.add_trace(go.Scatter(x=[xt.loc[nm, "xlogp"]], y=[xt.loc[nm, "rt_median_min"]], mode="markers+text",
                                text=[nm], textposition="top center", marker=dict(color=colr, size=12), name=nm))
fr.update_xaxes(title_text="XLogP3 (hydrophobicity)")
fr.update_yaxes(title_text=f"measured RT (min), {method().short}")
fr.update_layout(height=340, margin=dict(t=20, b=40, l=10, r=10), template="plotly_white", showlegend=False)
st.plotly_chart(fr, width="stretch")
st.caption("More hydrophobic, later: the main trend. Off the trend: bases charged at pH 3 elute early (imazalil, "
           "spinosyns), very large molecules late (avermectins), and log P estimates have their own errors "
           "(docs/retention_chemistry.md, docs/lc.md).")



# --- Down at the C18 surface ---------------------------------------------------------------------
st.subheader("Down at the C18 surface")
st.caption("Drawn as C18, the most common reversed-phase surface (a phenyl-hexyl phase such as the application "
           "note's adds pi-pi contact with aromatic rings; docs/retention_chemistry.md). "
           "A molecule among the C18 chains (grey) bonded to the silica (tan and red), with water and methanol "
           "flowing past. Free, it drifts with the flow. Held, it sits partitioned into the chain layer: oily end among the chains, polar end at the top "
           "facing the solvent. The share of time held is the real k / (1 + k) at the %B around its band, and its "
           "average speed down the column is 1 / (1 + k) of the solvent's. Each molecule is caught and released "
           "thousands of times on the way down (real hops last far less than a millisecond; time is compressed "
           "here), and retention is the average of all of them.")
an_by = {a.name: a for a in i.analytes}
pk1, pk2 = st.columns(2)
scene_a = pk1.selectbox("Molecule in the scene", mol_names, index=mol_names.index(pick_a),
                        help="Starts as the molecule picked above. The window follows this compound's band.")
scene_b = pk2.selectbox("Add a second compound", ["(none)"] + mol_names, index=0, key="scene_compare",
                        help="Shown in the same window while its band overlaps; it leaves as the bands separate")
scene_names = [scene_a] + ([scene_b] if scene_b not in ("(none)", scene_a) else [])
tRs = {n: retention_time(an_by[n].log_kw, an_by[n].S, cond) for n in scene_names}
sc1, sc3, sc2 = st.columns([1, 1, 2])
mode = sc1.radio("Show", ["Through the run (gradient)", "One moment"], horizontal=False)
copies = sc3.radio("Molecules", [1, 8], index=1 if len(scene_names) > 1 else 0, horizontal=True,
                   format_func=lambda n: "one each" if n == 1 else "eight each",
                   help="Eight copies: at any instant about k/(1+k) of them are held, so the time average "
                        "shows as an average over molecules")
t_one = sc2.slider("Moment (min)", 0.0, float(round(tRs[scene_a], 2)), float(round(0.85 * tRs[scene_a], 2)), 0.05,
                   disabled=mode != "One moment",
                   help="The followed band's position and the %B around it at this time")


@st.cache_data(show_spinner="Building the surface scene", max_entries=8)
def surface_scene(names: tuple[str, ...], mode_: str, t_: float, n_copies: int,
                  height: int) -> tuple[str, list[float], dict]:
    a0 = an_by[names[0]]
    t_stop = retention_time(a0.log_kw, a0.S, cond) * 1.06
    times = np.full(160, t_) if mode_ == "One moment" else np.linspace(0.0, t_stop, 230)
    xs = {nm: np.asarray(band_trajectory(an_by[nm].log_kw, an_by[nm].S, cond, times), dtype=float) for nm in names}
    # Band width: plate broadening, with a floor at the injected plug (the band is never narrower than the
    # injection spread over the column's open cross-section; no on-column focusing assumed).
    eps = cond.t0 * i.lc_method.flow_ml_min / (np.pi * (i.system.column_id_mm / 20) ** 2 * L / 10)
    plug_mm = i.prep.injection_volume_ul.nominal / 1000 / (np.pi * (i.system.column_id_mm / 20) ** 2 * eps) * 10
    floor2 = (plug_mm / np.sqrt(12) / L) ** 2
    sig = {nm: np.sqrt(np.array([band_sigma_fraction(min(v, 1.0), N) for v in xs[nm]]) ** 2 + floor2) for nm in names}
    species, curves = [], {}
    for j, nm in enumerate(names):
        a = an_by[nm]
        x = xs[nm]
        phi_s = np.asarray(phi_in_column(cond, np.minimum(x, 1.0), times), dtype=float)
        k_s = np.asarray(k_lss(phi_s, a.log_kw, a.S), dtype=float)
        d = x - xs[names[0]]                                    # + downstream of the followed band
        if j == 0:
            presence, note = np.ones(len(times)), [f"(band {min(v, 1.0) * L:.0f} mm down)" for v in x]
        else:
            s2 = sig[nm] ** 2 + sig[names[0]] ** 2
            presence = np.where(np.abs(d) < 1e-6, 1.0, np.exp(-d**2 / (2 * np.maximum(s2, 1e-12))))
            note = [("(same place)" if abs(dd) * L < 0.5 else
                     f"(band {abs(dd) * L:.0f} mm {'ahead' if dd > 0 else 'behind'})") for dd in d]
        species.append(c18_scene.Species(molecules.load(nm), nm, phi_s, k_s, presence, np.where(d >= 0, 1, -1),
                                         note, x >= 1.0))
        curves[nm] = (list(map(float, phi_s)), list(map(float, k_s)))
    labels = [f"t = {t:.2f} min" for t in times]
    sc = c18_scene.build_multi(species, np.random.default_rng(11), labels=labels, n_copies=n_copies)
    return (c18_scene.html(sc, height=height, colour="cmp" if len(names) > 1 else "hyd"),
            list(map(float, times)), curves)


scene_html, s_times, s_curves = surface_scene(tuple(scene_names), mode, float(t_one), int(copies), 566)
components.html(scene_html, height=636)
if len(scene_names) > 1:
    st.caption(f"The window follows the {scene_a} band, and the solvent drawn is the mix there. At the start every band "
               f"sits together at the column head, so both compounds share the window. Once the gradient reaches them "
               f"they separate, and {scene_names[1]} leaves the window (downstream if its band pulls ahead, upstream "
               f"if it falls behind). Each compound holds and hops by its own k, at its own band's %B.")
if mode != "One moment":
    fs = make_subplots(specs=[[{"secondary_y": True}]])
    line_cols = ["#f2a33a", "#3fc7b0"]
    for j, n in enumerate(scene_names):
        s_phi, s_k = (np.array(v) for v in s_curves[n])
        fs.add_trace(go.Scatter(x=s_times, y=100 * s_phi, name=f"%B around the band: {n}",
                                line=dict(color="#8fb3ff", dash="solid" if j == 0 else "dash")), secondary_y=False)
        fs.add_trace(go.Scatter(x=s_times, y=100 * s_k / (1 + s_k), name=f"time held in C18 (%): {n}",
                                line=dict(color=line_cols[j])), secondary_y=True)
        if tRs[n] <= s_times[-1]:
            fs.add_vline(x=tRs[n], line=dict(color=line_cols[j], dash="dot"))
    fs.update_xaxes(title_text="min")
    fs.update_yaxes(title_text="%B", range=[0, 100], secondary_y=False)
    fs.update_yaxes(title_text="held (%)", range=[0, 100.5], secondary_y=True)
    fs.update_layout(height=280, margin=dict(t=20, b=40, l=10, r=10), legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fs, width="stretch")
    st.caption("Held about 100% of the time while the %B is low: the band waits at the column head. As the gradient "
               "reaches it, k falls, the molecule spends more and more time in the mobile phase, and the band "
               "accelerates out (dotted lines: elution).")
    sources_line(("dorsey_dill_1989", "snyder_kirkland_dolan_2010", "lippa_2005", "rafferty_2008", "rafferty_2007",
                  "rafferty_2011", "pubchem_methanol"),
                 label="Sources (held share, chain layer, methanol in the layer, molar volumes; the hop timing is a drawing device)")


provenance_table(method_provenance(i))
