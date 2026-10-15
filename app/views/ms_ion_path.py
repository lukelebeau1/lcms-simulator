"""Analytical, MS: the ion path, as a labelled cartoon. Spray, desolvation, Q1, collision cell, Q3, detector.

This is an illustration of what each stage does, not a model of it: droplet shrinkage, ion
trajectories and fragmentation are drawn, not computed (ESI and quadrupole ion optics are out of
scope in the build plan). The numbers on it are real: the method's m/z values, voltages, polarity
and dwell, and the compounds the LC model has eluting at that moment.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from common import (Station, follow_index, inputs, matrix_profile, method_provenance, page, plotly_spec,  # noqa: E402
                    provenance_table, recipe, sources_line, targets)
from ion_stream import (BACKGROUND_COLOUR, MATRIX_COLOUR, Q1_FWHM, Q1_PEAK, background_density,  # noqa: E402
                        background_in_window, population, q1_transmission)
from spray_cartoon import SpraySettings, solvent_state  # noqa: E402
from quad_stability import (APEX_A, APEX_RATIO, Q_END, effective_a, paths, ratio_for_width,  # noqa: E402
                            rf_amplitude_v, upper_boundary, window)
from spray_cartoon import simulate as spray_simulate  # noqa: E402

from lcms_sim.io.schemas import Polarity, TransitionRole  # noqa: E402
from lcms_sim.lc.gradient import LCConditions  # noqa: E402
from lcms_sim.lc.column import phi_in_column  # noqa: E402
from lcms_sim.lc.peaks import peak_sigma  # noqa: E402
from lcms_sim.lc.retention import retention_time  # noqa: E402
from lcms_sim.ms.acquisition import build_schedule, cycle_layout, points_across_peak  # noqa: E402
from lcms_sim.quant.ledger import RecipeError, run_recipe, with_actual  # noqa: E402
from lcms_sim.rng import make_rng  # noqa: E402

page("MS: ion path",
     "A cartoon of one transition being measured: the spray turns eluent into charged droplets, the solvent "
     "evaporates until ions are released, Q1 passes only the precursor m/z, the collision cell breaks it into "
     "fragments, Q3 passes only the chosen product, and the electron multiplier counts what arrives. "
     "Motion is illustrative; the m/z values, voltages and co-eluting compounds come from the method and the model.", station=Station.source)

i = inputs()
cond = LCConditions.from_method(i.system, i.lc_method)
names = targets(i)
SPEED_MS = {"Slow": 120, "Normal": 60, "Fast": 30}

with st.sidebar:
    st.subheader("Transition")
    analyte = st.selectbox("Compound", names, index=follow_index(names))
    trs = [t for t in i.transitions if t.analyte == analyte]
    labels = [f"{t.precursor_mz:g} > {t.product_mz:g} ({t.role.value})" for t in trs]
    k_sel = st.selectbox("Q3 set to", range(len(trs)), format_func=lambda k: labels[k])
    a = next(x for x in i.analytes if x.name == analyte)
    tR = retention_time(a.log_kw, a.S, cond)
    st.subheader("Moment and injection")
    t_now = st.slider("Moment in the run (min)", 0.0, float(i.lc_method.run_time_min), round(tR, 2), 0.01,
                      help=f"Defaults to the apex of {analyte} ({tR:.2f} min)")
    columns = [s.vessel for s in recipe().steps if s.op == "inject"]
    column = st.selectbox("Vial injected", columns, format_func=lambda c: c.removeprefix("Column: "),
                          index=follow_index(columns, "injection"),
                          help="Amounts from Bench steps, including any edits made there")
    st.subheader("Playback")
    speed = st.select_slider("Speed", list(SPEED_MS), value="Normal")

tr = trs[k_sel]
try:
    vessels, _ = run_recipe(with_actual(recipe(), st.session_state.get("bench_actual", {})), "actual")
except RecipeError as e:
    st.error(f"The bench steps cannot be run as edited: {e}")
    st.stop()
# The ions entering Q1 at this moment: shared by the spray and Q1 close-ups (same species, same colours).
pop = population(i, cond, dict(vessels[column].ng), t_now, analyte, matrix_profile(), vessels[column].matrix_g > 0)

# Compounds the LC model has eluting at this compound's apex (within 3 sigma of their own apex).
co = []
for b in i.analytes:
    if b.name == analyte:
        continue
    tb = retention_time(b.log_kw, b.S, cond)
    sb = peak_sigma(b.log_kw, b.S, tb, cond, i.system.plate_number, i.system.extra_column_sigma_s)
    if abs(tb - tR) < 3 * sb:
        co.append(b.name)
sched = build_schedule(i.transitions, i.acquisition, i.lc_method.run_time_min)
kc = int(np.clip(np.searchsorted(sched.cycle_starts, tR, side="right") - 1, 0, len(sched.cycle_starts) - 1))
dwell = float(sched.dwell_ms[kc])

# --- Geometry of the cartoon (arbitrary units along the path) --------------------------------
X_TIP, X_FISSION, X_CAP = 6.0, 18.0, (20.0, 30.0)
Q1, Q2, Q3 = (34.0, 50.0), (54.0, 66.0), (70.0, 84.0)
X_EM, ROD = 88.0, 3.0
N_FRAMES, V_DROP, V_ION = 160, 0.55, 1.0
N_WARM = 100     # frames run before recording, so every stage is already busy at frame 0
HIT_FRAMES = 5   # how long a detector hit stays drawn

# Products of this compound in the collision cell, weighted by the method's relative responses.
products = [(t.product_mz, t.relative_response or 0.5) for t in trs]
pw = np.array([w for _, w in products])
pw = pw / pw.sum()


@st.cache_resource(show_spinner="Drawing the ion path")
def simulate(analyte: str, k_sel: int, n_co: int) -> list[dict]:
    """Particle positions per frame. Kinematics are a cartoon; the seed is fixed so it replays the same."""
    rng = make_rng(20261008)
    drops, ions, hits = [], [], []
    frames = []
    counted = 0
    for f in range(-N_WARM, N_FRAMES):
        for _ in range(2):   # two droplets leave the tip every frame
            drops.append(dict(x=X_TIP, slope=rng.uniform(-0.45, 0.45), size=14.0))
        # Droplets: fly, shrink as solvent evaporates, release ions at the fission point.
        for d in drops:
            d["x"] += V_DROP
            d["size"] = 14.0 - 10.0 * (d["x"] - X_TIP) / (X_FISSION - X_TIP)
        for d in [d for d in drops if d["x"] >= X_FISSION]:
            for _ in range(rng.integers(1, 3)):
                r = rng.random()
                # Share of ions: this compound, co-eluting compounds, solvent and matrix background.
                kind = "target" if r < 0.35 else ("coeluting" if (r < 0.55 and n_co) else "background")
                ions.append(dict(x=d["x"], y=d["slope"] * (d["x"] - X_TIP), kind=kind, phase=rng.uniform(0, 6.3),
                                 grow=rng.uniform(0.18, 0.3), product=None, lost=False))
        drops = [d for d in drops if d["x"] < X_FISSION]

        arrived = []
        for p in ions:
            p["x"] += V_ION
            x = p["x"]
            if x < X_CAP[1]:      # capillary and lenses focus ions onto the axis
                p["y"] *= 0.8
            elif Q1[0] <= x < Q1[1]:
                stable = p["kind"] == "target"
                amp = 0.5 if stable else 0.5 * np.exp(p["grow"] * (x - Q1[0]))
                p["y"] = amp * np.sin(1.3 * x + p["phase"])
                if abs(p["y"]) > ROD:
                    p["lost"] = True
            elif Q2[0] <= x < Q2[1]:
                if p["product"] is None:
                    # Collision with nitrogen: most precursors break into one of this compound's products.
                    p["product"] = (int(rng.choice(len(products), p=pw)) if rng.random() < 0.85 else -1)
                p["y"] = 0.9 * np.sin(2.1 * x + p["phase"]) * np.exp(-(x - Q2[0]) / 6)
            elif Q3[0] <= x < Q3[1]:
                stable = p["product"] == k_sel
                amp = 0.5 if stable else 0.5 * np.exp(p["grow"] * (x - Q3[0]))
                p["y"] = amp * np.sin(1.3 * x + p["phase"])
                if abs(p["y"]) > ROD:
                    p["lost"] = True
            elif x >= X_EM:
                arrived.append(p)
        ions = [p for p in ions if not p["lost"] and p["x"] < X_EM]
        hits = [(age + 1, y) for age, y in hits if age + 1 < HIT_FRAMES] + [(0, rng.uniform(-1.5, 1.5)) for _ in arrived]
        if f < 0:
            continue
        counted += len(arrived)

        def pts(sel):
            sub = [p for p in ions if sel(p)]
            return [p["x"] for p in sub], [p["y"] for p in sub]

        frames.append(dict(
            drops=([d["x"] for d in drops], [d["slope"] * (d["x"] - X_TIP) for d in drops], [d["size"] for d in drops]),
            target=pts(lambda p: p["kind"] == "target" and p["product"] is None),
            other=pts(lambda p: p["kind"] != "target"),
            chosen=pts(lambda p: p["product"] == k_sel),
            frag=pts(lambda p: p["product"] is not None and p["product"] != k_sel),
            hits=([X_EM + 2.0] * len(hits), [y for _, y in hits]),
            counted=counted,
        ))
    return frames


frames = simulate(analyte, k_sel, len(co))
TRACE = [("drops", "Droplets (solvent evaporating)", "lightskyblue"),
         ("target", f"{analyte} precursor ions, m/z {tr.precursor_mz:g}", "crimson"),
         ("other", "Other ions (co-eluting compounds, solvent, matrix)", "gray"),
         ("chosen", f"Product m/z {tr.product_mz:g} (Q3 setting)", "seagreen"),
         ("frag", "Other fragments", "darkorange"),
         ("hits", "Detector hits", "gold")]
sign = "+" if tr.polarity is Polarity.positive else "-"


def trace_data(fr: dict) -> list[dict]:
    out = []
    for key, _, _ in TRACE:
        if key == "drops":
            x, y, s = fr["drops"]
            out.append(dict(type="scatter", x=x, y=y, marker=dict(size=s)))
        else:
            x, y = fr[key]
            out.append(dict(type="scatter", x=x, y=y))
    return out


def title(fr: dict) -> str:
    return (f"Q1 passes m/z {tr.precursor_mz:g}   |   collision energy {tr.collision_energy_v:g} V   |   "
            f"Q3 passes m/z {tr.product_mz:g}   |   ions counted: {fr['counted']}")


fig = go.Figure()
for (key, name, colour), d in zip(TRACE, trace_data(frames[0])):
    style = dict(mode="markers", name=name, marker=dict(color=colour, size=7, opacity=0.85), hoverinfo="skip")
    if key == "drops":
        style["marker"] = dict(color=colour, size=d["marker"]["size"], opacity=0.6, line=dict(width=0))
    if key == "hits":
        style["marker"] = dict(color=colour, size=12, symbol="star")
    fig.add_trace(go.Scatter(x=d["x"], y=d["y"], **style))

# Static hardware: needle, capillary, rods, collision cell, detector.
grey, rod_w = "#888", 7
shapes = [
    dict(type="line", x0=0, x1=X_TIP, y0=0, y1=0, line=dict(color=grey, width=6)),
    dict(type="rect", x0=X_CAP[0], x1=X_CAP[1], y0=-1.4, y1=1.4, line=dict(color=grey), fillcolor="rgba(128,128,128,0.12)"),
    dict(type="rect", x0=Q2[0], x1=Q2[1], y0=-ROD - 0.6, y1=ROD + 0.6, line=dict(color=grey),
         fillcolor="rgba(70,130,180,0.08)"),
    dict(type="rect", x0=X_EM, x1=X_EM + 4, y0=-2.5, y1=2.5, line=dict(color=grey), fillcolor="rgba(218,165,32,0.25)"),
]
for x0, x1 in (Q1, Q3):
    for yy in (ROD + 0.3, -ROD - 0.3):
        shapes.append(dict(type="line", x0=x0, x1=x1, y0=yy, y1=yy, line=dict(color=grey, width=rod_w)))
stage_labels = [((X_TIP + X_FISSION) / 2, "Spray: charged droplets,<br>solvent evaporates"),
          (sum(X_CAP) / 2, "Capillary and<br>drying gas"),
          (sum(Q1) / 2, f"Q1: mass filter<br>passes m/z {tr.precursor_mz:g}"),
          (sum(Q2) / 2, f"Collision cell (N2)<br>{tr.collision_energy_v:g} V: fragments"),
          (sum(Q3) / 2, f"Q3: mass filter<br>passes m/z {tr.product_mz:g}"),
          (X_EM + 2, "Electron<br>multiplier")]
annotations = [dict(x=x, y=6.3, text=t, showarrow=False, font=dict(size=11)) for x, t in stage_labels]

spec = fig.to_dict()
spec["layout"].update(
    shapes=shapes, annotations=annotations,
    title=dict(text=title(frames[0]), font=dict(size=13), x=0.0),
    xaxis=dict(range=[-1, 94], visible=False), yaxis=dict(range=[-7, 8], visible=False),
    height=430, margin=dict(t=60, b=90, l=10, r=10),
    legend=dict(orientation="h", y=-0.02, x=0, yanchor="top", font=dict(size=11)),
)
# Frames travel inside the Play button: Streamlit's fullscreen redraw drops registered frames.
anim = [dict(name=str(f), data=trace_data(fr), traces=list(range(len(TRACE))),
             layout=dict(title=dict(text=title(fr)))) for f, fr in enumerate(frames)]
dur = SPEED_MS[speed]
spec["layout"]["updatemenus"] = [dict(
    type="buttons", direction="left", x=0.0, y=1.0, xanchor="left", yanchor="bottom", showactive=False,
    pad=dict(b=30),
    buttons=[dict(label="Play", method="animate", args=[anim, dict(frame=dict(duration=dur, redraw=True),
                                                                    transition=dict(duration=0), mode="immediate")]),
             dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                                       mode="immediate", transition=dict(duration=0))])])]
spec["layout"]["title"].update(x=0.16, xanchor="left")
plotly_spec(spec, width="stretch")
st.caption("Cartoon motion. In the instrument each transition is measured for its dwell only; "
           "this shows what happens during one dwell, slowed down enormously.")
sources_line(("yost_enke_1978", "paul_steinwedel_1953", "agilent_5990_3595", "hoffmann_stroobant_2007"))

# --- What each stage does, with this transition's numbers -------------------------------------
role = "quantifier" if tr.role is TransitionRole.quantifier else "qualifier"
rows = [
    ("Spray (ESI)", "The eluent is sprayed from a charged needle into droplets. Solvent evaporates until the "
     f"charge pushes ions out of the droplet. {analyte} forms {'[M+H]+' if sign == '+' else '[M-H]-'} here.",
     f"{tr.polarity.value} mode", "Response factor x matrix factor (suppression by what co-elutes)"),
    ("Capillary and lenses", "Drying gas finishes desolvation; ions are pulled into the vacuum and focused. The "
     "fragmentor voltage sets how hard they are pushed in, and can break fragile ions early.",
     f"fragmentor {tr.fragmentor_v:g} V" if tr.fragmentor_v else "-", "Part of the response factor"),
    ("Q1", "Four rods with combined DC and RF voltages. Only one m/z has a stable path; every other ion "
     "oscillates more and more until it hits a rod.", f"m/z {tr.precursor_mz:g}",
     "Selectivity: only this precursor goes on"),
    ("Collision cell", "Precursors are accelerated into nitrogen. Collisions break bonds and give product ions.",
     f"CE {tr.collision_energy_v:g} V", "Relative response: the share that becomes this product (ion ratio)"),
    ("Q3", "A second mass filter. Only the chosen product reaches the detector.",
     f"m/z {tr.product_mz:g} ({role})", "One transition = one Q1/Q3 pair"),
    ("Electron multiplier", "Each ion knocks electrons off a dynode; the cascade becomes a countable pulse.",
     f"dwell {dwell:.1f} ms of a {i.acquisition.cycle_time_ms:g} ms cycle at {tR:.2f} min",
     "Counting noise (1/sqrt N), gain, dead time"),
]
st.dataframe(pd.DataFrame(rows, columns=["Stage", "What happens", "This transition", "In the simulator"]),
             hide_index=True, width="stretch")
st.caption(f"Co-eluting with {analyte} at {tR:.2f} min in the LC model: "
           + (", ".join(co) if co else "nothing") + ". Their ions reach Q1 too and are filtered out there.")
# --- Close-up: the spray ----------------------------------------------------------------------
st.subheader("Close-up: the spray")
st.caption("Where the charge comes from, why the solvent leaves and the analyte stays, and what the drying gas does. "
           "A cartoon steered by real solvent properties and textbook rules; it does not feed the simulated signal.")
pct_b_elution = 100 * float(phi_in_column(cond, 1.0, t_now))
with st.sidebar:
    st.subheader("Spray close-up")
    pct_b = st.slider("Organic (methanol, % B)", 0, 100, int(round(pct_b_elution)),
                      help=f"Defaults to the %B leaving the column at {t_now:.2f} min ({pct_b_elution:.0f}%).")
    gas_t = st.slider("Drying gas temperature (C)", 100, 350, 200, 10, help="Application note 5994-1734EN: 200 C.")
    gas_f = st.slider("Drying gas flow (L/min)", 3, 16, 13, help="Application note 5994-1734EN: 13 L/min.")
    neb = st.slider("Nebuliser pressure (psi)", 15, 60, 55, help="Application note 5994-1734EN: 55 psi.")
    volts = st.slider("Spray voltage (V)", 2000, 6000, 5000, 250,
                      help="Application note 5994-1734EN: 5000 V positive, 3500 V negative.")

sp = SpraySettings(percent_b=pct_b, gas_temp_c=gas_t, gas_flow_l_min=gas_f, nebuliser_psi=neb, voltage_v=volts,
                   positive=tr.polarity is Polarity.positive)
ss = solvent_state(sp)
c = st.columns(4)
c[0].metric("Surface tension", f"{ss.surface_tension_mn_m:.0f} mN/m", help="Water 72.8, methanol 22.6")
c[1].metric("Vapour pressure", f"{ss.vapour_pressure_kpa:.1f} kPa", help="At 20 C. Water 2.34, methanol 13.0")
c[2].metric("Evaporation, relative", f"{ss.evaporation_rel:.1f}x", help="Water with 200 C gas at 10 L/min = 1")
c[3].metric("Starting droplet size", f"{ss.droplet_size_rel:.2f}x", help="Water at the method's 55 psi = 1")


@st.cache_resource(show_spinner="Drawing the spray")
def spray_frames(sp: SpraySettings, species: tuple[tuple[str, float], ...]) -> list[dict]:
    return spray_simulate(sp, make_rng(20261009), species)


# Analyte molecules in the droplets are drawn from the same population as Q1 sees (not to scale: rate^0.3).
sp_weights: dict[str, float] = {}
for s_ in pop:
    sp_weights[s_.colour] = sp_weights.get(s_.colour, 0.0) + s_.rate ** 0.3
sframes = spray_frames(sp, tuple(sorted(sp_weights.items())) or (("crimson", 1.0),))
q = "+" if sp.positive else "-"
SPRAY_TRACES = [("gas", "Drying gas (N2)", dict(color="rgba(160,160,160,0.35)", size=3)),
                ("vapour", "Solvent vapour", dict(color="rgba(135,206,250,0.6)", size=3)),
                ("solvent", "Solvent molecules", dict(color="steelblue", size=3)),
                ("analyte", "Analyte molecules (coloured by compound, as in Q1)", dict(color="crimson", size=7)),
                ("charges", f"Excess charge ({q})", None),
                ("ions", "Analyte ions (same colours enter Q1)", dict(color="crimson", size=9, line=dict(color="gold", width=2))),
                ("sions", f"Solvent and additive ions ({'H+, NH4+, Na+' if sp.positive else 'HCOO-, Cl-'})",
                 dict(color="gold", size=5))]


def spray_data(fr: dict) -> list[dict]:
    out = []
    for key, _, _ in SPRAY_TRACES:
        x, y = fr[key]
        extra = {"text": [q] * len(x)} if key == "charges" else {}
        if key == "analyte":
            extra = {"marker": dict(color=fr["analyte_c"], size=7)}
        if key == "ions":
            extra = {"marker": dict(color=fr["ions_c"], size=9, line=dict(color="gold", width=2))}
        out.append(dict(type="scatter", x=np.round(np.asarray(x, dtype=float), 2).tolist(),
                        y=np.round(np.asarray(y, dtype=float), 2).tolist(), **extra))
    return out


needle = dict(type="rect", x0=-12, x1=0, y0=-1.6, y1=1.6, fillcolor="#888", line=dict(width=0))
cone = dict(type="path", path="M 0,-1.6 L 7,0 L 0,1.6 Z", fillcolor="rgba(135,206,250,0.5)", line=dict(width=0))
inlet = [dict(type="rect", x0=100, x1=108, y0=y0, y1=y1, fillcolor="#888", line=dict(width=0))
         for y0, y1 in ((1.2, 9), (-9, -1.2))]
static_shapes = [needle, cone, *inlet]


def spray_shapes(fr: dict) -> list[dict]:
    return static_shapes + [dict(type="circle", x0=cx - r, x1=cx + r, y0=cy - r, y1=cy + r,
                                 line=dict(color="lightskyblue", width=1), fillcolor="rgba(135,206,250,0.12)")
                            for cx, cy, r in fr["circles"]]


def spray_title(fr: dict) -> str:
    return (f"Droplets in flight {fr['n_drops']}   |   fissions {fr['fissions']}   |   "
            f"analyte ions released {fr['released']}   |   droplets reaching the inlet still wet {fr['wet']}")


sfig = go.Figure()
for (key, name, marker), d in zip(SPRAY_TRACES, spray_data(sframes[0])):
    if key == "charges":
        sfig.add_trace(go.Scatter(x=d["x"], y=d["y"], text=d["text"], mode="text", name=name,
                                  textfont=dict(color="gold", size=11), hoverinfo="skip"))
    else:
        sfig.add_trace(go.Scatter(x=d["x"], y=d["y"], mode="markers", name=name, marker=d.get("marker", marker),
                                  hoverinfo="skip"))
sanim = [dict(name=str(k), data=spray_data(fr), traces=list(range(len(SPRAY_TRACES))),
              layout=dict(shapes=spray_shapes(fr), title=dict(text=spray_title(fr)))) for k, fr in enumerate(sframes)]
sspec = sfig.to_dict()
sspec["layout"].update(
    shapes=spray_shapes(sframes[0]),
    annotations=[dict(x=-6, y=-4.5, text=f"Needle at {volts:g} V", showarrow=False, font=dict(size=11)),
                 dict(x=5, y=6, text="Taylor cone", showarrow=False, font=dict(size=11)),
                 dict(x=104, y=12, text="Capillary inlet", showarrow=False, font=dict(size=11)),
                 dict(x=80, y=-26, text="Drying gas flows out of the inlet, toward the spray", showarrow=False,
                      font=dict(size=11))],
    title=dict(text=spray_title(sframes[0]), font=dict(size=13), x=0.16, xanchor="left"),
    xaxis=dict(range=[-13, 109], visible=False),
    yaxis=dict(range=[-29, 29], visible=False, scaleanchor="x", scaleratio=1),
    height=470, margin=dict(t=60, b=80, l=10, r=10),
    legend=dict(orientation="h", y=-0.02, x=0, yanchor="top", font=dict(size=11)),
    updatemenus=[dict(type="buttons", direction="left", x=0.0, y=1.0, xanchor="left", yanchor="bottom",
                      showactive=False, pad=dict(b=30), buttons=[
                          dict(label="Play", method="animate",
                               args=[sanim, dict(frame=dict(duration=SPEED_MS[speed], redraw=True),
                                                 transition=dict(duration=0), mode="immediate")]),
                          dict(label="Pause", method="animate",
                               args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                                  transition=dict(duration=0))])])],
)
plotly_spec(sspec, width="stretch")

charge_src = ("the voltage pulls positive ions (H+ from formic acid, NH4+ from ammonium formate, Na+) to the liquid "
              "surface at the tip, while negative ions are pushed back and discharged at the metal "
              "(the needle works as an electrolytic cell: water is oxidised there)") if sp.positive else (
              "the voltage pulls negative ions (formate, deprotonated analyte) to the liquid surface, while positive "
              "ions are pushed back and discharged at the metal (the needle works as an electrolytic cell)")
st.markdown(f"""
**Where the charge comes from.** At the needle tip {charge_src}. The surface fills with excess {q} charge, the
charges repel each other and the field pulls the liquid into a cone (the Taylor cone). A thin jet leaves its tip
and breaks into droplets that each carry excess {q} charge. Like charges repel, so the charge sits at the droplet
surface. The nebuliser gas shears the liquid into smaller droplets; more pressure, smaller droplets.

**Why the solvent leaves and the analyte stays.** Water and methanol are small, light molecules with high vapour
pressure; pesticides are large and barely volatile. As the droplet flies, solvent molecules escape from its surface
and the analyte is left behind, more and more concentrated. Methanol evaporates faster than water (vapour pressure
13.0 vs 2.34 kPa at 20 C), so the higher the %B, the faster the drying: at {pct_b}% B the evaporation is
{ss.evaporation_rel:.1f}x water's at the default gas settings. A droplet's area shrinks at a steady rate
(the d-squared law), so small droplets vanish quickly.

**What the drying gas does.** Hot nitrogen flows out of the inlet toward the spray. It supplies the heat that
evaporation takes ({gas_t} C here: hotter means faster drying, but very hot gas can break fragile compounds) and it
sweeps the neutral vapour away from the inlet, while the electric field still pulls the ions in.

**Why droplets split.** The droplet loses solvent but keeps its charge, so the charge gets more crowded. Past a limit
set by surface tension and size (the Rayleigh limit), the repulsion beats surface tension and the droplet throws off
small offspring that carry about 2% of its mass but 15% of its charge. Lower surface tension
({ss.surface_tension_mn_m:.0f} mN/m here; water is 72.8) means the limit is reached sooner. Repeated splits end in
droplets so small that single ions escape: analyte ions, and ions from the solvent and additives.

**Why it matters for signal.** Droplets that reach the inlet still wet carry analyte that never became an ion. That is
one reason response changes along the gradient, and why gas temperature and flow are tuned per method.
""")
sources_line(("blades_1991", "van_berkel_1995", "taylor_1964", "fenn_1989", "bruins_1987", "crc_handbook_97",
              "langmuir_1918", "yamashita_fenn_1984", "agilent_5990_3494", "agilent_5994_1734en", "rayleigh_1882",
              "kebarle_verkerk_2009", "gomez_tang_1994", "iribarne_thomson_1976", "dole_1968", "page_2007", "page_2009"),
             label="Sources (charge, cone and droplets; solvent properties and the d-squared law; drying gas and the "
                   "Ultivo's source; the Rayleigh limit; ion release; the inlet. Gas-flow and voltage scalings are "
                   "illustrative, and 'very hot gas can break fragile compounds' is stated without a source)")

# --- Close-up: Q1 -----------------------------------------------------------------------------
st.subheader("Close-up: Q1, the first mass filter")
st.caption("The same ions that left the spray above, now entering Q1. During each dwell Q1 is set to one precursor "
           "m/z and passes only ions within about 0.35 m/z of it; everything else spirals into the rods. Step "
           "through the cycle to move Q1 from one m/z to the next. Colours match the spray close-up.")
sources_line(("agilent_5994_1734en", "sante_11312_2021", "dawson_1995", "paul_steinwedel_1953"),
             label="Sources (unit resolution 0.7 m/z, the mass filter; the 0.3 peak transmission is assumed)")
with st.sidebar:
    st.subheader("Q1 close-up")
    mode = st.radio("Dwell set by", ["Cycle time (dynamic MRM)", "Fixed dwell"], index=0,
                    help="Dynamic MRM shares a fixed cycle among the open windows; fixed dwell sets it directly "
                         "and the cycle grows instead")
    if mode.startswith("Cycle"):
        cyc_ms = float(st.slider("Cycle time (ms)", 100, 1000, int(i.acquisition.cycle_time_ms), 10))
    else:
        fixed_ms = float(st.slider("Dwell per transition (ms)", 1, 200, 20))

acq0 = i.acquisition
lay0 = cycle_layout(build_schedule(i.transitions, acq0, i.lc_method.run_time_min), acq0, t_now)
if not lay0.slots:
    st.info(f"No transitions are scheduled at {t_now:.2f} min.")
else:
    overhead = acq0.cycle_time_ms - sum(x.dwell_ms + x.delay_ms for x in lay0.slots)
    if not mode.startswith("Cycle"):
        cyc_ms = len(lay0.slots) * (fixed_ms + acq0.inter_scan_delay_ms) + overhead
    acq_q = acq0.model_copy(update={"cycle_time_ms": cyc_ms})
    try:
        lay = cycle_layout(build_schedule(i.transitions, acq_q, i.lc_method.run_time_min), acq_q, t_now)
    except ValueError as e:
        st.error(f"{e}. Lengthen the cycle.")
        st.stop()
    slot_tr = [i.transitions[x.transition_index] for x in lay.slots]
    mf_now = float(matrix_profile().factor(t_now))
    supp = 1.0 - mf_now

    def passing(setpoint: float, pol) -> tuple[float, dict[str, float]]:
        """Ions per s Q1 passes at this setpoint: total, and per compound."""
        per: dict[str, float] = {}
        for s in pop:
            if s.polarity is pol:
                per[s.name] = per.get(s.name, 0.0) + s.rate * float(q1_transmission(s.mz, setpoint))
        return sum(per.values()) + background_in_window(setpoint, supp), per

    sig_t = peak_sigma(a.log_kw, a.S, tR, cond, i.system.plate_number, i.system.extra_column_sigma_s)
    target_slots = [k for k, x in enumerate(slot_tr) if x.analyte == analyte]
    c = st.columns(2) + st.columns(2)
    c[0].metric("Transitions in this cycle", len(lay.slots))
    c[1].metric("Dwell each", f"{lay.slots[0].dwell_ms:.1f} ms")
    c[2].metric("Cycle", f"{cyc_ms:.0f} ms")
    c[3].metric(f"Points across {analyte}", f"{points_across_peak(sig_t, cyc_ms):.1f}",
                help="4 sigma / cycle; the method flags below 10")

    k_default = (target_slots[0] + 1) if target_slots else 1
    k_step = st.slider("Step through the cycle (transition)", 1, len(lay.slots), k_default)

    # Ions shown as particles: the same species as the spray, plus background; counts are not to scale.
    @st.cache_resource(show_spinner="Placing the ions")
    def q1_particles(key: tuple, species: tuple[tuple[str, float, str, str], ...], n: int = 70) -> list[dict]:
        rng = make_rng(20261010)
        grid = np.linspace(100.0, 1000.0, 1801)
        dens = background_density(grid, supp)
        cdf = np.cumsum(dens) / dens.sum()
        w = np.array([wt for _, wt, _, _ in species])
        w = w / w.sum()
        out = []
        for _ in range(n):
            if rng.random() < 0.4 or not species:
                mz = float(np.interp(rng.random(), cdf, grid))
                colour = MATRIX_COLOUR if abs(mz - 380) < 90 and rng.random() < supp * 3 else BACKGROUND_COLOUR
                out.append(dict(mz=mz, colour=colour, pol="any", name="background"))
            else:
                mz, _, colour, pol = species[int(rng.choice(len(species), p=w))]
                out.append(dict(mz=mz, colour=colour, pol=pol, name=colour))
            out[-1].update(phase=float(rng.uniform(0, 6.3)), y0=float(rng.normal(0, 0.6)))
        return out

    sp_species = tuple((s.mz, s.rate ** 0.3, s.colour, s.polarity.value) for s in pop)
    parts = q1_particles((analyte, column, round(t_now, 3)), sp_species)
    XS = np.linspace(0.0, 100.0, 41)
    ROD_IN, ROD_OUT, ROD_Y = 12.0, 92.0, 4.5

    def rods_data(setpoint: float, pol) -> list[dict]:
        """Paths through the rods: stable near the setpoint, growing oscillation elsewhere (drawn, not computed)."""
        lines: dict[str, tuple[list, list]] = {}
        ends_x, ends_y, ends_c, ends_s = [], [], [], []
        for p in parts:
            if p["pol"] not in ("any", pol.value):
                continue
            dm = abs(p["mz"] - setpoint)
            stable = float(q1_transmission(p["mz"], setpoint)) >= 0.5 * Q1_PEAK
            g = 0.0 if stable else 0.03 + 0.09 * min(dm / 40.0, 1.0)
            ys = []
            for x in XS:
                inside = max(x - ROD_IN, 0.0)
                amp = 0.9 * math.exp(g * inside) if x >= ROD_IN else 0.3
                y = p["y0"] * (1 - min(x, ROD_IN) / ROD_IN) + amp * math.sin(0.22 * x + p["phase"])
                if abs(y) >= ROD_Y:
                    ys.append(math.copysign(ROD_Y, y))
                    break
                ys.append(y)
            xs = list(XS[:len(ys)])
            lx, ly = lines.setdefault(p["colour"], ([], []))
            lx += [round(v, 1) for v in xs] + [None]
            ly += [round(v, 2) for v in ys] + [None]
            ends_x.append(round(xs[-1], 1))
            ends_y.append(round(ys[-1], 2))
            ends_c.append(p["colour"])
            ends_s.append("circle" if len(ys) == len(XS) else "x")
        data = [dict(type="scatter", x=lines.get(col, ([], []))[0], y=lines.get(col, ([], []))[1]) for col in colour_keys]
        data.append(dict(type="scatter", x=ends_x, y=ends_y, marker=dict(color=ends_c, symbol=ends_s, size=7)))
        return data

    colour_keys = list(dict.fromkeys([s.colour for s in pop] + [BACKGROUND_COLOUR, MATRIX_COLOUR]))
    names_by_colour = {s.colour: s.name for s in pop} | {BACKGROUND_COLOUR: "Solvent and additive ions",
                                                         MATRIX_COLOUR: "Matrix ions"}

    def slot_title(k: int) -> str:
        x, trk = lay.slots[k], slot_tr[k]
        tot, per = passing(trk.precursor_mz, trk.polarity)
        own = per.get(trk.analyte, 0.0)
        n_pass = tot * x.dwell_ms / 1000.0
        return (f"Q1 at m/z {trk.precursor_mz:g}: {trk.analyte} ({trk.role.value}, {trk.polarity.value}), "
                f"{x.start_ms:.0f} ms into the cycle<br>dwell {x.dwell_ms:.1f} ms   |   {n_pass:,.3g} ions pass "
                f"this dwell, {(100 * own / tot if tot else 0):.2g}% of them {trk.analyte}")

    def window_shapes(setpoint: float) -> list[dict]:
        half = Q1_FWHM / 2
        return [dict(type="rect", xref=xr, yref=yr, x0=setpoint - half, x1=setpoint + half, y0=0, y1=1,
                     fillcolor="rgba(220,20,60,0.25)", line=dict(width=0)) for xr, yr in (("x", "y domain"),
                                                                                       ("x2", "y2 domain"))]

    rod_shapes = [dict(type="line", xref="x3", yref="y3", x0=ROD_IN, x1=ROD_OUT, y0=yy, y1=yy,
                       line=dict(color="#888", width=8)) for yy in (ROD_Y + 0.4, -ROD_Y - 0.4)]

    qfig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{"colspan": 2}, None]], column_widths=[0.62, 0.38],
                         row_heights=[0.5, 0.5], vertical_spacing=0.16,
                         subplot_titles=("Ions arriving now, by m/z", "Around the Q1 setting", ""))
    grid = np.linspace(100.0, 1000.0, 1801)
    dens = background_density(grid, supp)
    # Plain lists: numpy arrays reach the browser as encoded typed arrays, which the animation's redraw drops.
    grid_l, dens_l = grid.round(2).tolist(), dens.round(1).tolist()
    for col in (1, 2):
        qfig.add_trace(go.Scatter(x=grid_l, y=dens_l, mode="lines", line=dict(color=BACKGROUND_COLOUR, width=1),
                                  fill="tozeroy", fillcolor="rgba(255,215,0,0.15)", name="Background (per m/z)",
                                  showlegend=col == 1,
                                  hovertemplate="m/z %{x:.1f}: %{y:.3g} ions/s per m/z<extra></extra>"), row=1, col=col)
        qfig.add_trace(go.Bar(x=[s.mz for s in pop], y=[s.rate for s in pop], width=0.9,
                              marker_color=[s.colour for s in pop], showlegend=False,
                              customdata=[[s.name, s.polarity.value] for s in pop],
                              hovertemplate="%{customdata[0]} (%{customdata[1]}), m/z %{x:g}: %{y:.3g} ions/s"
                                            "<extra></extra>"), row=1, col=col)
    first_rod = len(qfig.data)
    k0 = k_step - 1
    for d, col in zip(rods_data(slot_tr[k0].precursor_mz, slot_tr[k0].polarity), colour_keys + [None]):
        if col is None:
            qfig.add_trace(go.Scatter(x=d["x"], y=d["y"], mode="markers", marker=d["marker"], showlegend=False,
                                      hoverinfo="skip"), row=2, col=1)
        else:
            qfig.add_trace(go.Scatter(x=d["x"], y=d["y"], mode="lines", line=dict(color=col, width=1.5),
                                      opacity=0.75, name=names_by_colour.get(col, ""), hoverinfo="skip"), row=2, col=1)
    rod_traces = list(range(first_rod, len(qfig.data)))
    qfig.update_yaxes(type="log", range=[0, 9], title_text="ions/s", row=1, col=1)
    qfig.update_yaxes(type="log", range=[0, 9], row=1, col=2)
    qfig.update_xaxes(range=[100, 950], title_text="m/z", row=1, col=1)
    qfig.update_xaxes(title_text="m/z", row=1, col=2)
    qfig.update_xaxes(visible=False, range=[0, 100], row=2, col=1)
    qfig.update_yaxes(visible=False, range=[-6.5, 6.5], row=2, col=1)

    def zoom(setpoint: float) -> list[float]:
        return [setpoint - 4.0, setpoint + 4.0]

    qspec = qfig.to_dict()
    qspec["layout"].update(
        shapes=window_shapes(slot_tr[k0].precursor_mz) + rod_shapes,
        title=dict(text=slot_title(k0), font=dict(size=12), x=0.22, xanchor="left", y=0.985, yanchor="top"),
        height=660, margin=dict(t=100, b=60, l=10, r=10), barmode="overlay",
        legend=dict(orientation="h", y=-0.02, x=0, yanchor="top", font=dict(size=11)),
    )
    qspec["layout"]["xaxis2"]["range"] = zoom(slot_tr[k0].precursor_mz)
    qspec["layout"]["annotations"] = list(qspec["layout"].get("annotations", [])) + [
        dict(xref="x3", yref="y3", x=5, y=5.6, text="from the capillary", showarrow=False, font=dict(size=11)),
        dict(xref="x3", yref="y3", x=(ROD_IN + ROD_OUT) / 2, y=6.1, text="Q1 rods: DC + RF", showarrow=False,
             font=dict(size=11)),
        dict(xref="x3", yref="y3", x=97, y=5.6, text="to the collision cell", showarrow=False, font=dict(size=11))]
    slow = {"Slow": 40, "Normal": 20, "Fast": 8}[speed]
    qanim = []
    for k in range(k0, len(lay.slots)):
        trk = slot_tr[k]
        qanim.append(dict(name=str(k), data=rods_data(trk.precursor_mz, trk.polarity), traces=rod_traces,
                          layout={"shapes": window_shapes(trk.precursor_mz) + rod_shapes,
                                  "title": {"text": slot_title(k)}, "xaxis2": {"range": zoom(trk.precursor_mz)}}))
    durs = [max(60, (lay.slots[k].dwell_ms + lay.slots[k].delay_ms) * slow) for k in range(k0, len(lay.slots))]
    qspec["layout"]["updatemenus"] = [dict(
        type="buttons", direction="left", x=0.0, y=1.0, xanchor="left", yanchor="bottom", showactive=False,
        pad=dict(b=40), buttons=[
            dict(label="Play", method="animate",
                 args=[qanim, dict(frame=[dict(duration=d, redraw=True) for d in durs],
                                   transition=dict(duration=0), mode="immediate")]),
            dict(label="Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                    transition=dict(duration=0))])])]
    plotly_spec(qspec, width="stretch")
    st.caption(f"Play steps Q1 through the cycle from the transition set above, each step lasting its dwell x {slow} "
               "(slowed down). Paths are drawn, not computed: Q1 is a Mathieu-stability filter, and only ions near "
               "the setting have bounded paths. Dots are not to scale: every species gets a few so the trace ones "
               "show; the spectrum and the table carry the real rates.")

    # Numbers for the transition the slider is on.
    trk, x = slot_tr[k0], lay.slots[k0]
    tot, per = passing(trk.precursor_mz, trk.polarity)
    arriving = sum(s.rate for s in pop if s.polarity is trk.polarity) + float(
        np.trapezoid(background_density(grid, supp), grid))
    rows = [
        {"Quantity": "All ions arriving at Q1 during this dwell", "Ions": arriving * x.dwell_ms / 1000},
        {"Quantity": "Ions Q1 passes during this dwell", "Ions": tot * x.dwell_ms / 1000},
        *[{"Quantity": f"  of which {n}", "Ions": v * x.dwell_ms / 1000} for n, v in sorted(per.items(), key=lambda z: -z[1])
          if v * x.dwell_ms / 1000 >= 0.01],
        {"Quantity": "  of which background", "Ions": background_in_window(trk.precursor_mz, supp) * x.dwell_ms / 1000},
    ]
    own_rate = sum(s.rate for s in pop if s.name == trk.analyte and s.polarity is trk.polarity
                   and abs(s.mz - trk.precursor_mz) < 0.5)
    rows.append({"Quantity": f"{trk.analyte} ions arriving in the rest of the cycle (Q1 elsewhere, lost)",
                 "Ions": own_rate * (cyc_ms - x.dwell_ms) / 1000})
    st.dataframe(pd.DataFrame(rows).style.format({"Ions": "{:,.4g}"}), hide_index=True, width="stretch")
    same = [s.name for s in pop if s.polarity is trk.polarity and abs(s.mz - trk.precursor_mz) <= Q1_FWHM
            and s.name != trk.analyte]
    st.caption(f"Q1 set to m/z {trk.precursor_mz:g} at {t_now:.2f} min, {column.removeprefix('Column: ')} injected. "
               + (f"Also inside the window: {', '.join(same)}. Q1 cannot separate them; the collision cell and Q3 "
                  "must. " if same else "")
               + "Rates use the model's response factor and gain (placeholders) and an assumed background.")

    # Why Q1 passes a band: the stability diagram at this setting.
    st.markdown("#### Why Q1 passes a band: the stability diagram")
    st.caption("Each ion's sideways motion between the rods depends on two numbers, a (from the DC voltage U) and q "
               "(from the RF amplitude V), both divided by its m/z. Inside the shaded region both rod pairs keep it "
               "bounded and it flies through; outside, its wobble grows until it hits a rod. Fixed U and V put every "
               "ion on one line from the origin (slope 2U/V), heavier ions nearer the origin. The stretch of line "
               "inside the region is the window; tilting the line toward the apex narrows it.")
    width = st.slider("Ideal window width at this setting (m/z)", 0.1, 3.0, Q1_FWHM, 0.05,
                      help="Sets U/V. Narrower means closer to the apex, where even stable ions wobble wide and "
                           "more of them clip the rods: higher resolution, lower transmission.")
    mz_set = trk.precursor_mz
    ratio = ratio_for_width(width, mz_set)
    lo_mz, hi_mz, q_mid = window(ratio, mz_set)
    near = [s for s in pop if s.polarity is trk.polarity and abs(s.mz - mz_set) <= 1.5]
    refs = [(mz_set - 1.0, f"m/z {mz_set - 1:g} (one below)"), (mz_set + 1.0, f"m/z {mz_set + 1:g} (one above)")]

    sfig_s = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55],
                           subplot_titles=("Whole stability region", f"Around the apex, Q1 at m/z {mz_set:g}"))
    qq = np.linspace(0.0, Q_END, 400)
    zx = (q_mid * (1 - 1.6 / mz_set), q_mid * (1 + 1.6 / mz_set))   # +-1.6 m/z around the setting
    for col, (x0, x1) in ((1, (0.0, 0.95)), (2, zx)):
        qs = np.linspace(x0, x1, 400) if col == 2 else qq
        sfig_s.add_trace(go.Scatter(x=qs.tolist(), y=upper_boundary(qs).tolist(), mode="lines", fill="tozeroy",
                                    line=dict(color="#7fb3d5", width=1), fillcolor="rgba(127,179,213,0.3)",
                                    name="Stable for both rod pairs", showlegend=col == 1, hoverinfo="skip"),
                         row=1, col=col)
        sfig_s.add_trace(go.Scatter(x=[x0, x1], y=[ratio * x0, ratio * x1], mode="lines",
                                    line=dict(color="#888", dash="dash"), name="Operating line (all m/z)",
                                    showlegend=col == 1, hoverinfo="skip"), row=1, col=col)
        ql, qr = q_mid * mz_set / hi_mz, q_mid * mz_set / lo_mz
        sfig_s.add_trace(go.Scatter(x=[ql, qr], y=[ratio * ql, ratio * qr], mode="lines",
                                    line=dict(color="seagreen", width=5), name=f"Window m/z {lo_mz:.2f}-{hi_mz:.2f}",
                                    showlegend=col == 1, hoverinfo="skip"), row=1, col=col)
        pts = [(s.mz, s.name, s.colour) for s in near] + [(m, lab, "#bbbbbb") for m, lab in refs]
        qx = [q_mid * mz_set / m for m, _, _ in pts]
        inside = [lo_mz <= m <= hi_mz for m, _, _ in pts]
        sfig_s.add_trace(go.Scatter(
            x=qx, y=[ratio * x for x in qx], mode="markers", showlegend=False,
            marker=dict(color=[c for _, _, c in pts], size=11, symbol=["circle" if k else "x" for k in inside],
                        line=dict(color=["seagreen" if k else "firebrick" for k in inside], width=2)),
            customdata=[[n, m, "passes" if k else "hits the rods"] for (m, n, _), k in zip(pts, inside)],
            hovertemplate="%{customdata[0]}, m/z %{customdata[1]:g}: %{customdata[2]}<extra></extra>"),
            row=1, col=col)
    sfig_s.update_xaxes(title_text="q (RF)", range=[0.0, 0.95], row=1, col=1)
    sfig_s.update_yaxes(title_text="a (DC)", range=[0.0, 0.26], row=1, col=1)
    zy = (min(ratio * zx[0], float(upper_boundary(zx[0])), float(upper_boundary(zx[1]))) * 0.9995,
          max(APEX_A, ratio * zx[1]) * 1.0004)
    sfig_s.update_xaxes(title_text="q (RF)", range=list(zx), row=1, col=2)
    sfig_s.update_yaxes(range=list(zy), row=1, col=2)
    sfig_s.add_shape(type="rect", xref="x", yref="y", x0=zx[0], x1=zx[1], y0=zy[0], y1=zy[1],
                     line=dict(color="crimson", width=1))
    sfig_s.update_layout(height=420, margin=dict(t=40, b=60, l=10, r=10), template="plotly_white",
                         legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=11)))
    st.plotly_chart(sfig_s, width="stretch")

    v_amp = rf_amplitude_v(mz_set, q_mid)
    cs_ = st.columns(2) + st.columns(2)
    cs_[0].metric("Window", f"{lo_mz:.2f}-{hi_mz:.2f}")
    cs_[1].metric("U / V", f"{ratio / 2:.5f}", help=f"The apex is at {APEX_RATIO / 2:.5f}: zero width, zero transmission")
    cs_[2].metric("RF amplitude", f"{v_amp:,.0f} V", help="0-peak, for an assumed 4 mm rod gap at 1 MHz")
    cs_[3].metric("To move 0.1 m/z", f"{100 * 0.1 / mz_set:.3f}% of V",
                  help="The passed m/z is proportional to V, so mass accuracy is RF stability and calibration")
    st.caption(f"Ideal edges: an ion just outside the region grows slowly and can leave the rods before it hits one, "
               f"so real edges are sloped and the real width depends on how many RF cycles an ion spends inside "
               f"(roughly m/dm ~ n^2/12). Rod gap and frequency are assumed; the region and line are exact. "
               f"Grey crosses mark m/z {mz_set - 1:g} and {mz_set + 1:g} for scale. Illustration; not used by the signal model.")
    sources_line(("paul_steinwedel_1953", "dawson_1995", "abramowitz_stegun_1964"),
                 label="Sources (Mathieu equation and stability region; n^2/12 is Dawson's rough rule; rod gap and frequency assumed)")

    # Inside the rods: one U, one V, two pairs with opposite sign.
    st.markdown("#### Inside the rods: one U, one V, opposite signs on the two pairs")
    st.caption("Looking down the rods. Opposite rods are wired together: one pair gets +(U + V cos wt), the other "
               "-(U + V cos wt), so the colours swap every RF cycle. The + pair's plane throws out ions that are too "
               "light (the RF shakes them hardest); the - pair's plane lets ions that are too heavy drift out (the "
               "RF barely holds them). Only the setting survives both. Paths computed for these U and V.")
    sources_line(("paul_steinwedel_1953", "dawson_1995"))
    delta = st.slider("Compare ions this far from the setting (m/z)", 0.5, 3.0, 1.0, 0.5)
    N_CYC, PER = 80, 2
    ions3 = [(mz_set, "seagreen", f"m/z {mz_set:g} (the setting)"),
             (mz_set - delta, "darkorange", f"m/z {mz_set - delta:g} (lighter)"),
             (mz_set + delta, "mediumpurple", f"m/z {mz_set + delta:g} (heavier)")]

    @st.cache_resource(show_spinner="Computing paths")
    def rod_paths(ratio_: float, q_mid_: float, mz_: float, d_: float, n_: int, per_: int) -> list[dict]:
        out = []
        for m in (mz_, mz_ - d_, mz_ + d_):
            qi = q_mid_ * mz_ / m
            out.append(paths(ratio_ * qi, qi, n_, per_cycle=per_))
        return out

    P = rod_paths(ratio, q_mid, mz_set, delta, N_CYC, PER)
    cyc = P[0]["cycles"]
    ROD_C, ROD_R = 2.15, 1.15                     # rod centres and radius, in units of the gap r0
    uv = ratio / 2

    ROD_XY = [(ROD_C, 0.0), (-ROD_C, 0.0), (0.0, ROD_C), (0.0, -ROD_C)]

    def frame_data(k: int) -> list[dict]:
        """What moves: the three ions, and one text layer with the rods' signs and the status.

        Kept to two kinds of update so the animation stays light (every moving trace costs a redraw).
        """
        out = []
        for pth, (_, col, _) in zip(P, ions3):
            lo_k = max(0, k - 3 * PER)
            out.append(dict(x=pth["x"][lo_k:k + 1].round(3).tolist(), y=pth["y"][lo_k:k + 1].round(3).tolist(),
                            marker=dict(color=col, size=[3] * (k - lo_k) + [11])))
        plus = uv + math.cos(2 * math.pi * cyc[k]) >= 0
        a_, b_ = ("+", "-") if plus else ("-", "+")
        red, blue = "rgb(230,80,80)", "rgb(80,130,230)"
        out.append(dict(text=[a_, a_, b_, b_, rods_title(k)],
                        textfont=dict(size=[22, 22, 22, 22, 10],
                                      color=[red if plus else blue] * 2 + [blue if plus else red] * 2 + ["#bbb"])))
        return out

    def rods_title(k: int) -> str:
        bits = []
        for pth, (m, _, _) in zip(P, ions3):
            hx, hy = pth["hit_x"], pth["hit_y"]
            hits = [(h, "+") for h in (hx,) if h is not None] + [(h, "-") for h in (hy,) if h is not None]
            first = min(hits) if hits else None
            bits.append(f"{m:g}: hit {first[1]} rod" if first and cyc[k] >= first[0] else f"{m:g}: flying")
        return f"cycle {cyc[k]:.0f}<br>" + "<br>".join(bits)

    # End-on view: animated on its own (fast animations drop traces in subplots they do not update).
    rfig = go.Figure()
    f0 = frame_data(0)
    for d, (_, col, _) in zip(f0[:3], ions3):
        rfig.add_trace(go.Scatter(x=d["x"], y=d["y"], mode="lines+markers", line=dict(color=col, width=1.5),
                                  marker=d["marker"], showlegend=False, hoverinfo="skip"))
    for (cx, cy), lab in zip(ROD_XY, ("+ pair", "+ pair", "- pair", "- pair")):   # the rods themselves, to scale
        rfig.add_shape(type="circle", x0=cx - ROD_R, x1=cx + ROD_R, y0=cy - ROD_R, y1=cy + ROD_R,
                       line=dict(color="#888", width=2), fillcolor="rgba(150,150,150,0.15)")
        rfig.add_annotation(x=cx, y=cy - 0.75 if cy <= 0 else cy + 0.75, text=lab, showarrow=False, font=dict(size=10))
    rfig.add_trace(go.Scatter(x=[x for x, _ in ROD_XY] + [-3.3], y=[y for _, y in ROD_XY] + [3.3], mode="text",
                              text=f0[3]["text"], textfont=f0[3]["textfont"],
                              textposition=["middle center"] * 4 + ["bottom right"], showlegend=False,
                              hoverinfo="skip"))
    moving = list(range(len(rfig.data)))
    rfig.update_xaxes(range=[-3.4, 3.4], visible=False)
    rfig.update_yaxes(range=[-3.4, 3.4], visible=False, scaleanchor="x", scaleratio=1)
    rfig.update_layout(height=430, margin=dict(t=50, b=10, l=10, r=10), template="plotly_white")
    rspec = rfig.to_dict()
    ranim = [dict(name=str(k), data=frame_data(k), traces=moving) for k in range(len(cyc))]
    rspec["layout"]["updatemenus"] = [dict(
        type="buttons", direction="left", x=0.0, y=1.0, xanchor="left", yanchor="bottom", showactive=False,
        pad=dict(b=10), buttons=[
            dict(label="Play", method="animate", args=[ranim, dict(frame=dict(duration=60, redraw=False),
                                                                   transition=dict(duration=0), mode="immediate")]),
            dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                                      mode="immediate", transition=dict(duration=0))])])]

    # Both planes over the whole flight: static, beside it.
    sfig2 = make_subplots(rows=2, cols=1, vertical_spacing=0.2,
                          subplot_titles=("Plane of the + pair (x)", "Plane of the - pair (y)"))
    for pth, (_, col, lab) in zip(P, ions3):
        sfig2.add_trace(go.Scatter(x=cyc.tolist(), y=np.round(pth["x"], 3).tolist(), mode="lines",
                                   line=dict(color=col, width=1.2), name=lab), row=1, col=1)
        sfig2.add_trace(go.Scatter(x=cyc.tolist(), y=np.round(pth["y"], 3).tolist(), mode="lines",
                                   line=dict(color=col, width=1.2), showlegend=False), row=2, col=1)
    for r_ in (1, 2):
        sfig2.add_hrect(y0=1.0, y1=1.1, fillcolor="#888", line_width=0, row=r_, col=1)
        sfig2.add_hrect(y0=-1.1, y1=-1.0, fillcolor="#888", line_width=0, row=r_, col=1)
        sfig2.update_yaxes(range=[-1.1, 1.1], title_text="off axis / gap", row=r_, col=1)
        sfig2.update_xaxes(range=[0, N_CYC], row=r_, col=1)
    sfig2.update_xaxes(title_text="RF cycles", row=2, col=1)
    sfig2.update_layout(height=430, margin=dict(t=60, b=40, l=10, r=10), template="plotly_white",
                        legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=11)))
    col_a, col_b = st.columns([0.45, 0.55])
    with col_a:
        plotly_spec(rspec, width="stretch")
    with col_b:
        st.plotly_chart(sfig2, width="stretch")
    st.caption(f"The + pair carries +(U + V cos wt), the - pair -(U + V cos wt). U / V = {uv:.5f}, the setting at "
               f"q = {q_mid:.4f}. "
               "Red: rods positive at that instant; blue: negative (V is much larger than U, so each pair flips sign "
               "every cycle). "
               "Each ion starts 1% of the gap off axis, at rest. Near the apex even the setting's ion swings wide, so "
               "ions that enter further off axis are lost too: the transmission cost of a narrow window.")

    # The saddle: the potential between the rods, and what each ion feels on average.
    st.markdown("#### The saddle")
    st.caption("The potential across the gap between the rods is a saddle: (U + V cos wt)(x^2 - y^2) / r0^2, uphill along "
               "one pair's axis and downhill along the other's. A static field can only make saddles (Laplace's "
               "equation: no bowl in every direction), so DC alone cannot hold an ion. The RF flips the saddle every "
               "cycle; on average that acts like a bowl whose depth falls with m/z.")
    sources_line(("dehmelt_1967", "gerlich_1992", "douglas_french_1992"),
                 label="Sources (pseudopotential, RF-only ion guides and their cutoff, collisional focusing)")
    mode_s = st.radio("Voltages on the rods", ["DC + RF (Q1, Q3)", "DC only", "RF only (collision cell)"],
                      horizontal=True)
    dc_on, rf_on = mode_s != "RF only (collision cell)", mode_s != "DC only"
    N_GRID, N_SADDLE = 31, 48                       # grid points across the gap; frames over two RF cycles
    gx = np.linspace(-1.0, 1.0, N_GRID)
    GX, GY = np.meshgrid(gx, gx)
    shape_ = np.where(GX**2 + GY**2 <= 1.0, GX**2 - GY**2, np.nan)   # inside the circle the rods enclose

    def phi_t(k: int) -> float:
        """(U + V cos wt) / V at frame k; DC only freezes it, RF only drops U."""
        wt = 2 * math.pi * 2 * k / N_SADDLE
        return (uv if dc_on else 0.0) + (math.cos(wt) if rf_on else 0.0)

    zmax = (uv if dc_on else 0.0) + (1.0 if rf_on else 0.0)

    def surf_z(k: int) -> list:
        return np.round(phi_t(k) * shape_, 4).tolist()

    sfig3 = go.Figure(go.Surface(x=gx.tolist(), y=gx.tolist(), z=surf_z(0), cmin=-zmax, cmax=zmax,
                                 colorscale=[[0, "rgb(60,110,220)"], [0.5, "rgb(235,235,235)"], [1, "rgb(220,60,60)"]],
                                 showscale=False, hoverinfo="skip"))
    sfig3.update_layout(
        height=420, margin=dict(t=40, b=0, l=0, r=0), template="plotly_white",
        scene=dict(xaxis=dict(title="x (+ pair)", range=[-1, 1]), yaxis=dict(title="y (- pair)", range=[-1, 1]),
                   zaxis=dict(title="potential", range=[-zmax, zmax], showticklabels=False),
                   camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9)), aspectmode="manual",
                   aspectratio=dict(x=1, y=1, z=0.6)))
    s3spec = sfig3.to_dict()
    if rf_on:
        s3anim = [dict(name=str(k), data=[dict(z=surf_z(k))], traces=[0]) for k in range(N_SADDLE)]
        s3spec["layout"]["updatemenus"] = [dict(
            type="buttons", direction="left", x=0.0, y=1.0, xanchor="left", yanchor="bottom", showactive=False,
            pad=dict(b=5), buttons=[
                dict(label="Play", method="animate", args=[s3anim, dict(frame=dict(duration=90, redraw=True),
                                                                        transition=dict(duration=0), mode="immediate")]),
                dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                                          mode="immediate", transition=dict(duration=0))])])]

    # The averaged restoring strength each ion feels in each plane, exactly (quad_stability.effective_a).
    rr = np.linspace(-1.0, 1.0, 81)
    efig = make_subplots(rows=2, cols=1, vertical_spacing=0.22,
                         subplot_titles=("Averaged, along x (+ pair)", "Averaged, along y (- pair)"))
    notes, in_legend = [], set()
    for m, col, lab in ions3:
        qi = (q_mid * mz_set / m) if rf_on else 0.0
        ai = (ratio * q_mid * mz_set / m) if dc_on else 0.0
        for r_, plane in ((1, "x"), (2, "y")):
            k_, kind = effective_a(ai, qi, plane)
            if kind == "rf":
                notes.append(f"m/z {m:g} along {plane}: driven out by the RF itself (no averaged bowl or hill)")
                efig.add_trace(go.Scatter(x=[0.0], y=[0.0], mode="markers", marker=dict(color=col, symbol="x", size=12),
                                          name=lab, showlegend=lab not in in_legend, hoverinfo="skip"), row=r_, col=1)
                in_legend.add(lab)
                continue
            efig.add_trace(go.Scatter(x=rr.tolist(), y=(k_ * rr**2).tolist(), mode="lines", line=dict(color=col),
                                      name=lab, showlegend=lab not in in_legend), row=r_, col=1)
            in_legend.add(lab)
    for r_ in (1, 2):
        efig.update_yaxes(title_text="strength", zeroline=True, row=r_, col=1)
    efig.update_xaxes(title_text="off axis / gap", row=2, col=1)
    efig.update_layout(height=420, margin=dict(t=40, b=40, l=10, r=10), template="plotly_white",
                       legend=dict(orientation="h", y=-0.3, x=0, font=dict(size=11)))
    ca, cb = st.columns([0.5, 0.5])
    with ca:
        plotly_spec(s3spec, width="stretch")
    with cb:
        st.plotly_chart(efig, width="stretch")
    verdict = {
        "DC + RF (Q1, Q3)": "Along y the RF's averaged bowl must beat the DC hill. The setting just manages (a shallow "
                            "bowl); the heavier ion feels the RF less and gets a hill, so it drifts out, as in the rods "
                            "view. The lighter ion is lost along x a different way: the RF shakes it too hard (marked x).",
        "DC only": "No RF, no averaged bowl: along x the DC holds every ion, along y every ion rolls downhill to the "
                   "- rods, whatever its m/z. Nothing passes.",
        "RF only (collision cell)": "No DC: every ion sits in an averaged bowl in both planes, shallower for heavier "
                                    "ions but never a hill. Everything above a low-m/z cutoff passes: an ion guide, "
                                    "not a filter."}[mode_s]
    st.caption(verdict + " Strength is each ion's averaged restoring force as an equivalent static curvature, "
                         "computed exactly from one RF cycle (bowl: beta^2; hill: minus the growth rate squared). "
                         + ("; ".join(notes) + "." if notes else ""))

provenance_table(method_provenance(i))
