"""Analytical, MS: how a quadrupole filters, in five steps. A teaching view; not part of the signal model."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import plotly.figure_factory as ff  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from collision_cell import KB, T_GAS, Cell, simulate  # noqa: E402
from common import chart, Station, inputs, page, plotly_spec, sources_line  # noqa: E402
from lcms_sim.rng import make_rng  # noqa: E402
from quad_stability import (APEX_RATIO, E_CHARGE, Q_END, a0, b1, effective_a, mathieu_aq, paths,  # noqa: E402
                            ratio_for_width, rf_amplitude_v, segment, slow_swing_cycles,
                            upper_boundary, window)

page("MS: how a quadrupole filters",
     "Five steps: why the field between the rods is a saddle, why flipping it holds an ion and which knobs decide "
     "that, how a little DC turns the hold into a filter, what one ion near the setting does, and the collision cell. The same physics as the Q1 close-up on the ion path page, one idea at a time.", station=Station.analyser)

# --- Step 1 ------------------------------------------------------------------------------------
st.subheader("1. The field between the rods is a saddle")
st.caption("Each rod is a line of charge; its push on a positive ion falls off with distance. Add up the four pushes "
           "and you get the map below: a height map of the potential (red high, blue low) and arrows for the force "
           "on a positive ion. Switch rods on and off to see what each pair contributes.")
c_a, c_b = st.columns(2)
rods_on = c_a.radio("Rods switched on", ["All four", "+ pair only", "- pair only"], horizontal=True)
flipped = c_b.toggle("Flip the voltages (half an RF cycle later)", value=False)
show_f = c_b.toggle("Show the force on a + ion", value=True)
s = -1.0 if flipped else 1.0
D, R_ROD = 1.95, 0.35
rods = [((D, 0.0), s), ((-D, 0.0), s), ((0.0, D), -s), ((0.0, -D), -s)]      # (position, charge sign)
active = [r for r in rods if (rods_on == "All four") or (rods_on == "+ pair only" and r[0][1] == 0.0)
          or (rods_on == "- pair only" and r[0][0] == 0.0)]


def potential(x, y):
    """2D line charges: phi = -sum lambda ln r (arbitrary units)."""
    return sum(-lam * np.log(np.hypot(x - cx, y - cy)) for (cx, cy), lam in active)


def force(x, y):
    """Force on a + ion: sum lambda (r - r_i) / |r - r_i|^2, pushing away from + rods, toward - rods."""
    fx = sum(lam * (x - cx) / ((x - cx) ** 2 + (y - cy) ** 2) for (cx, cy), lam in active)
    fy = sum(lam * (y - cy) / ((x - cx) ** 2 + (y - cy) ** 2) for (cx, cy), lam in active)
    return fx, fy


g = np.linspace(-1.55, 1.55, 125)
GX, GY = np.meshgrid(g, g)
inside = GX**2 + GY**2 <= 1.55**2
Z = potential(GX, GY) - potential(0.0, 0.0)
lim = float(np.nanpercentile(np.abs(np.where(inside, Z, np.nan)), 95)) or 1.0
f1 = go.Figure(go.Contour(x=g.tolist(), y=g.tolist(), z=np.where(inside, Z, np.nan).round(4).tolist(), zmin=-lim,
                          zmax=lim, ncontours=18, showscale=False, hoverinfo="skip", contours=dict(coloring="heatmap"),
                          colorscale=[[0, "rgb(60,110,220)"], [0.5, "rgb(240,240,240)"], [1, "rgb(220,60,60)"]]))
for (cx, cy), lam in rods:
    on = ((cx, cy), lam) in active
    f1.add_shape(type="circle", x0=cx - R_ROD, x1=cx + R_ROD, y0=cy - R_ROD, y1=cy + R_ROD,
                 fillcolor=("rgb(220,60,60)" if lam > 0 else "rgb(60,110,220)") if on else "rgba(0,0,0,0)",
                 line=dict(color="#666", dash="solid" if on else "dot"))
    f1.add_annotation(x=cx, y=cy, text=("+" if lam > 0 else "-") if on else "off", showarrow=False,
                      font=dict(color="white" if on else "#888", size=16 if on else 10))
if show_f:
    qx, qy = np.meshgrid(np.linspace(-1.2, 1.2, 9), np.linspace(-1.2, 1.2, 9))
    keep = (qx**2 + qy**2 <= 1.25**2)
    qx, qy = qx[keep], qy[keep]
    fx, fy = force(qx, qy)
    mag = np.hypot(fx, fy)
    scale = 0.4 / float(mag.max()) if mag.max() > 0 else 0.0
    q = ff.create_quiver(qx.tolist(), qy.tolist(), (fx * scale).tolist(), (fy * scale).tolist(), scale=1.0,
                         arrow_scale=0.3, line=dict(color="#111", width=1.5))
    f1.add_trace(go.Scatter(x=q.data[0].x, y=q.data[0].y, mode="lines", line=dict(color="#111", width=1.5),
                            hoverinfo="skip", showlegend=False))
f1.update_xaxes(visible=False, range=[-2.35, 2.35])
f1.update_yaxes(visible=False, range=[-2.35, 2.35], scaleanchor="x", scaleratio=1)
f1.update_layout(height=470, margin=dict(t=10, b=10, l=10, r=10), template="room")
c1, c2 = st.columns([0.55, 0.45])
with c1:
    chart(f1, width="stretch")
with c2:
    st.markdown({
        "+ pair only": """
- Two **plus** rods repel the ion. Toward either rod the nearer one pushes harder: the arrows point **back to the
  centre** along their line.
- Sideways, both pushes tilt away from the line: their sideways parts cancel, the outward parts add. The arrows
  point **away from the centre**.
- Already a saddle, from just two rods.""",
        "- pair only": """
- Two **minus** rods attract the ion. Toward either rod its pull grows: the arrows point **away from the centre**
  along their line.
- Sideways, both pulls tilt back toward their line: the arrows point **back to the centre**.
- The same saddle, the same way round as the plus pair's.""",
        "All four": """
- Both pairs make the same pattern, so they **add**: squeezed in along the line of the plus rods, pushed out
  along the line of the minus rods, stronger the further from the centre.
- At the very centre every push cancels: zero force. Off centre it is never zero, and never inward in every
  direction (a static field can only make saddles).
- Flip the voltages and every arrow reverses: that is the RF, a million times a second."""}[rods_on])
sources_line(("paul_steinwedel_1953", "dawson_1995", "paul_1990", "gerlich_1992"),
             label="Sources (two rod pairs at opposite potentials; the saddle and the rotating-saddle picture; no static "
                   "field holds a charge, Earnshaw's theorem; the line-charge drawing is a simplification of the field "
                   "round rods make)")

# --- Step 2 ------------------------------------------------------------------------------------
st.subheader("2. Why the flip holds: one ion, every knob")
st.caption("One ion between the rods, starting off axis at rest. Each flip pushes it one way, then the other. Its "
           "speed carries over, so it is furthest out just when the push turns inward, and the push is stronger out "
           "there: on balance it is pushed back to the centre. How far it swings in one flip decides everything, and "
           "each knob below changes that swing.")
k1, k2, k3 = st.columns(3)
mass1 = k1.slider("Ion mass (Da)", 20, 2000, 289, 1)
z1 = k1.slider("Charge z", 1, 4, 1)
v1 = k2.slider("RF amplitude V (volts, 0-peak)", 0, 2000, 120, 10)
f1_mhz = k2.slider("RF frequency (MHz)", 0.30, 3.00, 1.00, 0.05)
r0_mm = k3.slider("Rod gap r0, axis to rod (mm)", 2.0, 8.0, 4.0, 0.1)
u1 = k3.slider("DC U (volts)", 0, 400, 0, 1, help="0 = RF only, like the collision cell")
k4, k5, k6 = st.columns(3)
start1 = k4.slider("Starting offset (fraction of the gap)", 0.01, 0.50, 0.30, 0.01)
n1 = k5.slider("RF cycles to follow", 10, 200, 60, 5)
watch = k6.radio("Watch", ["Every flip", "Once per RF cycle"], horizontal=True,
                 help="Once per RF cycle shows the ion at the same moment of each flip: the jiggle disappears and "
                      "only the slow swing back through the centre (the cup) is left.")
a1, q1 = mathieu_aq(mass1, z1, v1, u1, f1_mhz * 1e6, r0_mm * 1e-3)
PER1 = 8


@st.cache_data(show_spinner="Computing the path")
def one_ion(a_: float, q_: float, n_: int, u0_: float) -> dict:
    p_ = paths(a_, q_, n_, u0=u0_, per_cycle=PER1)
    p_["swing_x"], p_["swing_y"] = slow_swing_cycles(a_, q_, "x"), slow_swing_cycles(a_, q_, "y")
    p_["kind_x"], p_["kind_y"] = effective_a(a_, q_, "x")[1], effective_a(a_, q_, "y")[1]
    return p_


P1 = one_ion(a1, q1, n1, start1)
cyc1 = P1["cycles"]
hits1 = [(h, pl) for h, pl in ((P1["hit_x"], "+"), (P1["hit_y"], "-")) if h is not None]
hit1 = min(hits1) if hits1 else None
held = P1["kind_x"] == "bowl" and P1["kind_y"] == "bowl"
us1 = 1.0 / f1_mhz                                  # microseconds per RF cycle
if v1 == 0 and u1 == 0:
    verdict1 = "No voltage, no field: the ion just carries on straight down the rods, wherever it started."
elif v1 == 0:
    verdict1 = ("DC only: a frozen saddle. Held between the + rods, but it runs away toward a - rod, whatever its "
                "mass. Nothing can be held by a static field.")
elif held and hit1 is None:
    slow = max(P1["swing_x"], P1["swing_y"])
    verdict1 = (f"Trapped. The flips hold it: it swings slowly back through the centre about every {slow:.0f} RF "
                f"cycles ({slow * us1:.0f} µs), with the fast jiggle on top.")
elif held:
    verdict1 = ("Held on average, but it started so far off axis that its swing reaches a rod anyway. Start it "
                "nearer the centre: the transmission cost of a wide entrance.")
elif P1["kind_x"] == "rf" or P1["kind_y"] == "rf":
    verdict1 = (f"Shaken out. The flip is too strong for this ion (q = {q1:.2f}): it overshoots further on every "
                "flip until it hits a rod. Less voltage, a heavier ion or faster flips would hold it.")
else:
    verdict1 = ("The DC wins. The flips give this ion only a shallow cup, and the steady pull toward the - rods is "
                "stronger. Less DC, more RF or a lighter ion would hold it.")
if not held and hit1 is None and not (v1 == 0 and u1 == 0):
    verdict1 += " (It has not reached a rod in the cycles shown, but it is on its way.)"
st.markdown(f"**{verdict1}**")

# End-on, animated in its own figure (fast animations drop moving traces in other subplots).
ROD_C1, ROD_R1 = 2.15, 1.15
ROD_XY1 = [(ROD_C1, 0.0), (-ROD_C1, 0.0), (0.0, ROD_C1), (0.0, -ROD_C1)]
step1 = PER1 if watch == "Once per RF cycle" else 1
ks = list(range(0, len(cyc1), step1))
trail = 10 if step1 > 1 else 2 * PER1
alive = max(1, int(np.sum(np.isfinite(P1["x"]) & np.isfinite(P1["y"]))))


def ion_frame(k: int) -> list[dict]:
    """What moves: the ion with a short trail, and one text layer with the rods' signs and the clock."""
    kk = min(k, alive - 1)
    idx = [i for i in ks if kk - trail * step1 <= i <= kk] or [kk]
    xs, ys = P1["x"][idx].round(3).tolist(), P1["y"][idx].round(3).tolist()
    plus = u1 + v1 * math.cos(2 * math.pi * cyc1[k]) >= 0
    a_, b_ = ("+", "-") if plus else ("-", "+")
    red, blue = "rgb(230,80,80)", "rgb(80,130,230)"
    gone = hit1 is not None and cyc1[k] >= hit1[0]
    status = f"cycle {cyc1[k]:.0f} ({cyc1[k] * us1:.0f} µs)" + (f"<br>hit a {hit1[1]} rod" if gone else "")
    return [dict(x=xs, y=ys, marker=dict(color="seagreen", size=[3] * (len(xs) - 1) + [12])),
            dict(text=[a_, a_, b_, b_, status],
                 textfont=dict(size=[22, 22, 22, 22, 11],
                               color=[red if plus else blue] * 2 + [blue if plus else red] * 2 + ["#888"]))]


ef = go.Figure()
fr0 = ion_frame(0)
ef.add_trace(go.Scatter(x=fr0[0]["x"], y=fr0[0]["y"], mode="lines+markers", line=dict(color="seagreen", width=1.5),
                        marker=fr0[0]["marker"], showlegend=False, hoverinfo="skip"))
for cx, cy in ROD_XY1:
    ef.add_shape(type="circle", x0=cx - ROD_R1, x1=cx + ROD_R1, y0=cy - ROD_R1, y1=cy + ROD_R1,
                 line=dict(color="#888", width=2), fillcolor="rgba(150,150,150,0.15)")
ef.add_trace(go.Scatter(x=[x for x, _ in ROD_XY1] + [-3.3], y=[y for _, y in ROD_XY1] + [3.3], mode="text",
                        text=fr0[1]["text"], textfont=fr0[1]["textfont"],
                        textposition=["middle center"] * 4 + ["bottom right"], showlegend=False, hoverinfo="skip"))
ef.update_xaxes(range=[-3.4, 3.4], visible=False)
ef.update_yaxes(range=[-3.4, 3.4], visible=False, scaleanchor="x", scaleratio=1)
ef.update_layout(height=430, margin=dict(t=50, b=10, l=10, r=10), template="room")
espec = ef.to_dict()
eanim = [dict(name=str(k), data=ion_frame(k), traces=[0, 1]) for k in ks]
espec["layout"]["updatemenus"] = [dict(
    type="buttons", direction="left", x=0.0, y=1.0, xanchor="left", yanchor="bottom", showactive=False,
    pad=dict(b=10), buttons=[
        dict(label="Play", method="animate",
             args=[eanim, dict(frame=dict(duration=40 if step1 == 1 else 120, redraw=False),
                               transition=dict(duration=0), mode="immediate")]),
        dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                                  mode="immediate", transition=dict(duration=0))])])]

# Both planes over the whole flight, static: thin line every flip, dots once per cycle (the slow swing).
tf = make_subplots(rows=2, cols=1, vertical_spacing=0.2,
                   subplot_titles=("Toward the + rods (x)", "Toward the - rods (y)"))
for r_, key in ((1, "x"), (2, "y")):
    tf.add_trace(go.Scatter(x=cyc1.tolist(), y=np.round(P1[key], 3).tolist(), mode="lines",
                            line=dict(color="seagreen", width=1), name="every flip", showlegend=r_ == 1),
                 row=r_, col=1)
    tf.add_trace(go.Scatter(x=cyc1[::PER1].tolist(), y=np.round(P1[key][::PER1], 3).tolist(), mode="markers",
                            marker=dict(color="#F3ECDD", size=4), name="once per cycle: the slow swing",
                            showlegend=r_ == 1), row=r_, col=1)
    tf.add_hrect(y0=1.0, y1=1.1, fillcolor="#888", line_width=0, row=r_, col=1)
    tf.add_hrect(y0=-1.1, y1=-1.0, fillcolor="#888", line_width=0, row=r_, col=1)
    tf.update_yaxes(range=[-1.1, 1.1], title_text="off axis / gap", row=r_, col=1)
    tf.update_xaxes(range=[0, n1], row=r_, col=1)
tf.update_xaxes(title_text="RF cycles", row=2, col=1)
tf.update_layout(height=430, margin=dict(t=60, b=40, l=10, r=10), template="room",
                 legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=11)))
ea, eb = st.columns([0.45, 0.55])
with ea:
    plotly_spec(espec, width="stretch")
with eb:
    chart(tf, width="stretch")

# Where the knobs put it: every knob only moves one dot on the stability diagram.
qd, txt = st.columns([0.45, 0.55])
with qd:
    qq = np.linspace(0, Q_END, 300)
    sd = go.Figure(go.Scatter(x=qq.tolist(), y=np.round(upper_boundary(qq), 4).tolist(), fill="tozeroy",
                              mode="lines", line=dict(color="seagreen"), fillcolor="rgba(46,139,87,0.2)",
                              name="held in both planes"))
    on_chart = q1 <= 1.4 and a1 <= 0.4
    sd.add_trace(go.Scatter(x=[min(q1, 1.4)], y=[min(a1, 0.4)], mode="markers+text",
                            marker=dict(size=13, color="crimson", symbol="circle" if on_chart else "triangle-right"),
                            text=["this ion" if on_chart else f"q {q1:.1f}, a {a1:.2f}"], textposition="top center",
                            name="this ion"))
    sd.update_xaxes(title_text="q (from the RF)", range=[0, 1.45])
    sd.update_yaxes(title_text="a (from the DC)", range=[0, 0.42])
    sd.update_layout(height=300, margin=dict(t=20, b=40, l=10, r=10), template="room", showlegend=False)
    chart(sd, width="stretch")
with txt:
    sw = [f"{s_:.0f} cycles" if s_ else "not held" for s_ in (P1["swing_x"], P1["swing_y"])]
    mm = st.columns(3)
    mm[0].metric("m/z", f"{mass1 / z1:.1f}")
    mm[1].metric("q", f"{q1:.3f}")
    mm[2].metric("a", f"{a1:.4f}")
    st.caption(f"Slow swing: {sw[0]} toward the + rods, {sw[1]} toward the - rods. Faster swing = deeper cup.")
    st.markdown(
        "Every knob only moves the red dot: **q = 4zeV / (m r0² Ω²)** and **a = 8zeU / (m r0² Ω²)**. Inside the "
        "green the ion is held.\n"
        "- **Heavier ion:** less swing per flip, a shallower cup. **More charge** does the opposite, so only m/z "
        "counts: double both and the path is identical.\n"
        "- **More RF volts:** a bigger swing and a deeper cup, until past q = 0.908 the ion overshoots and is "
        "shaken out.\n"
        "- **Faster flips:** less time to move in each, so a much smaller swing (q falls as 1/f²).\n"
        "- **Wider rods:** a weaker field for the same volts (1/r0²).\n"
        "- **DC:** a steady pull toward the - rods, the same for every mass. It lifts the dot; out of the green, "
        "the DC wins.")
st.caption("Paths are the Mathieu equation for these knobs; the ion starts at rest, the same distance off axis in "
           "both planes. No fringing fields at the rod entrance. Illustration; not used by the signal model.")
sources_line(("paul_steinwedel_1953", "dawson_1995", "abramowitz_stegun_1964", "dehmelt_1967", "gerlich_1992"),
             label="Sources (a and q and the 0.908 limit; Floquet and the slow swing; the pseudopotential cup; the "
                   "default rod gap and frequency are assumed)")

# --- Step 3 ------------------------------------------------------------------------------------
st.subheader("3. The contest: the RF's cup against the DC's pull")
st.caption("Flipping fast leaves each ion, on average, in a cup. Heavier ions respond less, so their cup is "
           "shallower. The DC adds a small steady pull toward the - rods, the same for every ion. An ion passes "
           "when its cup beats that pull and it is not so light that the flipping shakes it out.")
cc = st.columns(2)
mz_set = cc[0].slider("Q1 set to m/z", 50.0, 1000.0, 289.1, 0.1)
# The RF amplitude is what "set to m/z" fixes (calibrated at unit resolution); the DC slider then moves U only.
ratio_unit = ratio_for_width(0.7, mz_set)
q_set = window(ratio_unit, mz_set)[2]
V = rf_amplitude_v(mz_set, q_set)
dc_pct = cc[1].slider("DC as % of RF (U/V)", 0.0, round(100 * APEX_RATIO / 2, 2), round(50 * ratio_unit, 2), 0.01,
                      help=f"Default: unit resolution at this setting. {100 * APEX_RATIO / 2:.2f}% is the tip: the "
                           "window closes completely")
uv = dc_pct / 100.0
ratio = 2 * uv
U = uv * V
seg = segment(ratio)
if seg is None:
    st.warning("At this much DC no ion's cup can beat the pull: the window is closed.")
    st.stop()
lo = mz_set * q_set / seg[1]
hi = mz_set * q_set / seg[0] if seg[0] > 0 else math.inf


def hold_v(m):
    """Strongest steady pull (V) the RF cup of an ion of m/z m can resist: the - pair's stability edge."""
    q = q_set * mz_set / np.asarray(m, dtype=float)
    return -a0(q) / (2 * q) * V


def shake_v(m):
    """Most DC (V) an ion can take before the flipping shakes it out: the + pair's stability edge."""
    q = q_set * mz_set / np.asarray(m, dtype=float)
    return b1(q) / (2 * q) * V


def contest_fig(m_lo: float, m_hi: float, title: str, y_range: list | None = None) -> go.Figure:
    m = np.linspace(m_lo, m_hi, 600)
    hv, sv = hold_v(m), shake_v(m)
    ok = (U <= hv) & (U <= sv)
    f = go.Figure()
    f.add_trace(go.Scatter(x=m.tolist(), y=hv.tolist(), mode="lines", line=dict(color="seagreen", width=2),
                           name="RF cup: pull it can resist"))
    f.add_trace(go.Scatter(x=m.tolist(), y=sv.tolist(), mode="lines", line=dict(color="darkorange", width=2),
                           name="Shake-out limit"))
    f.add_trace(go.Scatter(x=[m_lo, m_hi], y=[U, U], mode="lines", line=dict(color="crimson", width=2, dash="dash"),
                           name=f"DC pull (U = {U:,.0f} V)"))
    # Shade where the ion passes, along the bottom.
    edges = np.flatnonzero(np.diff(ok.astype(int)))
    starts = [m[0]] if ok[0] else []
    stops = []
    for e in edges:
        (starts if not ok[e] else stops).append(m[e + 1] if not ok[e] else m[e])
    if ok[-1]:
        stops.append(m[-1])
    for a_, b_ in zip(starts, stops):
        f.add_vrect(x0=a_, x1=b_, fillcolor="seagreen", opacity=0.15, line_width=0)
    top = float(np.nanmax(np.clip(hv, 0, None))) if np.isfinite(hv).any() else U
    f.update_yaxes(title_text="volts", range=y_range or [0, max(top, U) * 1.15])
    f.update_xaxes(title_text="ion m/z")
    f.update_layout(title=dict(text=title, font=dict(size=13)), height=330, margin=dict(t=40, b=40, l=10, r=10),
                    template="room", legend=dict(orientation="h", y=-0.3, x=0, font=dict(size=11)))
    return f


m_min = mz_set * q_set / 0.95                       # lighter than this is far past the shake-out limit
cw1, cw2 = st.columns(2)
with cw1:
    chart(contest_fig(m_min, 4 * mz_set, "Wide view"), width="stretch")
with cw2:
    yz = [0.985 * U, 1.015 * U] if U > 0 else None          # tight around the DC line, where the edges are
    chart(contest_fig(mz_set - 2.0, mz_set + 2.0, f"Zoomed on m/z {mz_set:g}", yz), width="stretch")
mc = st.columns(2) + st.columns(2)
mc[0].metric("Passes", f"{lo:.1f}-{hi:.1f}" if hi - lo < 50 else f"above {lo:.0f}" if math.isinf(hi)
             else f"{lo:.0f}-{hi:.0f}")
mc[1].metric("Window width", f"{hi - lo:.2f} m/z" if hi - lo < 50 else "no upper limit" if math.isinf(hi) else "very wide")
mc[2].metric("RF amplitude", f"{V:,.0f} V", help="Assumed 4 mm rod gap at 1 MHz")
st.caption("Green: where the ion passes (its cup beats the DC pull and it is not shaken out). Set the DC to 0: "
           "everything above a cutoff passes, a guide like the collision cell. Raise it: the green narrows to the "
           "window. Change the m/z setting: the RF and DC scale together and the window moves. The curves are the "
           "exact edges of the stability diagram, expressed as volts.")
sources_line(("paul_steinwedel_1953", "dawson_1995", "abramowitz_stegun_1964", "agilent_5994_1734en"),
             label="Sources (the scan line and the stability edges; unit resolution 0.7 m/z from the example method)")

# --- Step 4 ------------------------------------------------------------------------------------
st.subheader("4. Try one ion near the setting")
st.caption("Pick an ion near the setting and follow it for 80 RF cycles in each rod pair's plane. "
           "Grey bands are the rods.")
off = st.slider("Ion m/z, relative to the setting", -3.0, 3.0, 1.0, 0.1)
m_ion = mz_set + off
q_ion = q_set * mz_set / m_ion
p = st.cache_data(show_spinner=False)(paths)(ratio * q_ion, q_ion, 80, per_cycle=12)
hx, hy = p["hit_x"], p["hit_y"]
if hx is None and hy is None:
    verdict = f"m/z {m_ion:.1f} rides through: its cup beats the DC pull and the flipping does not shake it out."
elif hy is not None and (hx is None or hy <= hx):
    verdict = (f"m/z {m_ion:.1f} slides out toward a - rod after {hy:.0f} cycles: too heavy, its cup is too shallow "
               "to beat the DC pull.")
else:
    verdict = (f"m/z {m_ion:.1f} is shaken out into a + rod after {hx:.0f} cycles: too light, it overreacts to the "
               "flipping.")
st.markdown(f"**{verdict}**")
f3 = make_subplots(rows=1, cols=2, subplot_titles=("Toward the + rods (x)", "Toward the - rods (y)"),
                   horizontal_spacing=0.1)
for col, key in ((1, "x"), (2, "y")):
    f3.add_trace(go.Scatter(x=p["cycles"].tolist(), y=np.round(p[key], 3).tolist(), mode="lines",
                            line=dict(color="mediumpurple" if off > 0 else "darkorange" if off < 0 else "seagreen"),
                            showlegend=False), row=1, col=col)
    f3.add_hrect(y0=1.0, y1=1.15, fillcolor="#888", line_width=0, row=1, col=col)
    f3.add_hrect(y0=-1.15, y1=-1.0, fillcolor="#888", line_width=0, row=1, col=col)
    f3.update_yaxes(range=[-1.15, 1.15], title_text="off axis / gap" if col == 1 else None, row=1, col=col)
    f3.update_xaxes(title_text="RF cycles", range=[0, 80], row=1, col=col)
f3.update_layout(height=300, margin=dict(t=40, b=40, l=10, r=10), template="room")
chart(f3, width="stretch")
st.caption("Paths computed for these voltages (Mathieu equation), each ion starting 1% of the gap off axis. "
           "The ion path page's Q1 close-up has the full detail: stability diagram, rods and saddle.")
sources_line(("paul_steinwedel_1953", "dawson_1995"))

# --- Step 5 ------------------------------------------------------------------------------------
st.subheader("5. The collision cell: RF only, gas, and a fragment")
st.caption("The cell is the same kind of rods with the DC locked at zero, so it holds everything above a low-mass "
           "cutoff. It is filled with nitrogen. The precursor enters at the collision energy, hits gas molecules, "
           "slows down and gets knocked sideways, and at some collision it breaks. The fragment keeps the "
           "precursor's speed but is lighter, so its q jumps up by precursor / product. Here it breaks at a "
           "collision you choose: which bond breaks, and when, is not predicted.")
TRS = sorted({(t.analyte, t.precursor_mz, t.product_mz, t.collision_energy_v or 20.0)
              for t in inputs().transitions if t.polarity.value == "positive"}, key=lambda r: (r[0], -r[2]))
labels = [f"{a} {p:g} > {d:g} (CE {ce:g} V)" for a, p, d, ce in TRS]
default_tr = next((k for k, (a, p, d, _) in enumerate(TRS) if a == "Acephate" and d == 49.0), 0)
c5a, c5b = st.columns([0.5, 0.5])
pick = c5a.selectbox("Transition from the method", range(len(TRS)), index=default_tr, format_func=lambda k: labels[k])
an5, prec5, prod5, ce_m = TRS[pick]
seed5 = c5b.number_input("Seed (same seed, same ions)", 0, 10**6, 1, 1)
g1, g2, g3 = st.columns(3)
v5 = g1.slider("Cell RF amplitude V (volts)", 10, 300, 60, 5, help="Assumed range; the Ultivo's cell RF is not published")
p5 = g1.slider("N2 pressure (mTorr)", 0.0, 20.0, 5.0, 0.5, help="0 = no gas: nothing slows, nothing breaks")
ce5 = g2.slider("Collision energy (V)", 0, 60, int(round(ce_m)), 1, help="Entry energy = CE x charge, in eV")
ax5 = g2.slider("Axial push across the cell (V)", 0.0, 10.0, 2.0, 0.5,
                help="A gentle field along the cell. At 0, ions that have lost their energy wander and stall")
brk5 = g3.slider("Breaks at collision number", 1, 30, 3, 1)
off5 = g3.slider("Entry offset (fraction of the gap)", 0.01, 0.50, 0.10, 0.01)
cell5 = Cell(v_rf=float(v5), pressure_mtorr=float(p5), axial_v=float(ax5))


@st.cache_data(show_spinner="Flying ions through the cell")
def fly(prec: float, prod: float, ce: float, brk: int, cell: Cell, off: float, seed: int, n: int) -> dict:
    return simulate(prec, prod, ce, brk, cell, n, off, make_rng(seed))


one5 = fly(prec5, prod5, float(ce5), brk5, cell5, off5, int(seed5), 1)
many5 = fly(prec5, prod5, float(ce5), brk5, cell5, off5, int(seed5) + 1, 200)
F5 = one5["first"]
st5, tend5 = one5["status"][0], one5["t_end_s"][0] * 1e6
n_coll5 = int(one5["ncoll"][0])
broke5 = bool(F5["prod"].any()) if F5["prod"].size else False
VERD5 = {
    "q3": f"The fragment m/z {prod5:g} reaches Q3 after {tend5:.0f} µs and {n_coll5} collisions.",
    "unbroken": (f"The precursor crossed the cell unbroken ({n_coll5} collisions, fewer than {brk5}). More gas or a "
                 "longer path would give it more chances to break."),
    "rod_prec": f"The precursor hit a rod after {tend5:.0f} µs, before it broke: knocked sideways too hard for its cup.",
    "rod_prod": (f"The fragment m/z {prod5:g} hit a rod after {tend5:.0f} µs: "
                 + ("it is past the cutoff, so the RF shakes it out." if one5["q_prod"] > Q_END else
                    "light and near the edge, collisions knocked it out of its cup.")),
    "back": "Scattered back out of the entrance: a head-on collision reversed it.",
    "stalled": (f"Stalled: still in the cell after {cell5.t_max_s * 1e3:.0f} ms. The gas took all its energy and "
                "nothing pushes it along. Raise the axial push."),
}
st.markdown(f"**{VERD5[st5]}**")

# Side view, animated in its own figure: distance along the cell against distance toward the + rods.
tt = F5["t"]
nfr = min(200, max(2, tt.size))
fidx = np.unique(np.linspace(0, tt.size - 1, nfr).astype(int)) if tt.size else np.array([0])
keep = np.unique(np.linspace(0, max(tt.size - 1, 0), min(tt.size, 600)).astype(int)) if tt.size else np.array([], int)
HZ, HX = one5["hits"]["z"], one5["hits"]["x"]
ht = one5["hits"]["t"]


def cell_frame(k: int) -> list[dict]:
    """Moving traces: precursor trail, fragment trail, the ion now, the collisions so far, the clock."""
    upto = keep[keep <= k]
    pre = upto[~F5["prod"][upto]] if upto.size else upto
    pro = upto[F5["prod"][upto]] if upto.size else upto
    now_c = "seagreen" if F5["prod"][k] else "darkorange"
    hk = ht <= tt[k]
    clock = f"{tt[k] * 1e6:.1f} µs, {int(hk.sum())} collisions, {F5['ke_ev'][k]:.2g} eV"
    return [dict(x=F5["z"][pre].round(4).tolist(), y=F5["x"][pre].round(4).tolist()),
            dict(x=F5["z"][pro].round(4).tolist(), y=F5["x"][pro].round(4).tolist()),
            dict(x=[round(float(F5["z"][k]), 4)], y=[round(float(F5["x"][k]), 4)], marker=dict(color=now_c, size=12)),
            dict(x=HZ[hk].round(4).tolist(), y=HX[hk].round(4).tolist()),
            dict(text=[clock])]


if tt.size:
    sv = go.Figure()
    c0 = cell_frame(int(fidx[0]))
    sv.add_trace(go.Scatter(x=c0[0]["x"], y=c0[0]["y"], mode="lines", line=dict(color="darkorange", width=1.2),
                            name=f"precursor m/z {prec5:g}"))
    sv.add_trace(go.Scatter(x=c0[1]["x"], y=c0[1]["y"], mode="lines", line=dict(color="seagreen", width=1.2),
                            name=f"fragment m/z {prod5:g}"))
    sv.add_trace(go.Scatter(x=c0[2]["x"], y=c0[2]["y"], mode="markers", marker=c0[2]["marker"], showlegend=False))
    sv.add_trace(go.Scatter(x=c0[3]["x"], y=c0[3]["y"], mode="markers", name="collision",
                            marker=dict(symbol="x", size=7, color="#F3ECDD")))
    sv.add_trace(go.Scatter(x=[0.0], y=[1.32], mode="text", text=c0[4]["text"], textposition="middle right",
                            showlegend=False, textfont=dict(size=11, color="#888")))
    sv.add_hrect(y0=1.0, y1=1.18, fillcolor="#888", line_width=0)
    sv.add_hrect(y0=-1.18, y1=-1.0, fillcolor="#888", line_width=0)
    sv.update_xaxes(title_text="along the cell (entrance 0, exit 1)", range=[-0.02, 1.02])
    sv.update_yaxes(title_text="toward the + rods / gap", range=[-1.2, 1.4])
    sv.update_layout(height=380, margin=dict(t=50, b=40, l=10, r=10), template="room",
                     legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=11)))
    svspec = sv.to_dict()
    svanim = [dict(name=str(k), data=cell_frame(int(k)), traces=[0, 1, 2, 3, 4]) for k in fidx]
    svspec["layout"]["updatemenus"] = [dict(
        type="buttons", direction="left", x=0.0, y=1.0, xanchor="left", yanchor="bottom", showactive=False,
        pad=dict(b=10), buttons=[
            dict(label="Play", method="animate", args=[svanim, dict(frame=dict(duration=60, redraw=False),
                                                                    transition=dict(duration=0), mode="immediate")]),
            dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                                      mode="immediate", transition=dict(duration=0))])])]
    plotly_spec(svspec, width="stretch")

    # Energy and distance off axis over time, static.
    ef5 = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Energy (eV, log scale)", "Distance off axis / gap"))
    tus = tt * 1e6
    ef5.add_trace(go.Scatter(x=tus[keep].tolist(), y=np.clip(F5["ke_ev"][keep], 1e-3, None).tolist(), mode="lines",
                             line=dict(color="#7A6690", width=1), showlegend=False), row=1, col=1)
    ef5.add_hline(y=1.5 * KB * T_GAS / E_CHARGE, line=dict(color="#aaa", dash="dot"), row=1, col=1,
                  annotation_text="room-temperature gas", annotation_position="bottom right")
    r_ = np.hypot(F5["x"], F5["y"])
    ef5.add_trace(go.Scatter(x=tus[keep].tolist(), y=r_[keep].round(4).tolist(), mode="lines",
                             line=dict(color="#7A6690", width=1), showlegend=False), row=1, col=2)
    ef5.add_hrect(y0=1.0, y1=1.1, fillcolor="#888", line_width=0, row=1, col=2)
    if broke5:
        tb = float(tus[np.argmax(F5["prod"])])
        for cc in (1, 2):
            ef5.add_vline(x=tb, line=dict(color="seagreen", dash="dash"), row=1, col=cc)
    ef5.update_yaxes(type="log", row=1, col=1)
    ef5.update_yaxes(range=[0, 1.1], row=1, col=2)
    ef5.update_xaxes(title_text="µs in the cell")
    ef5.update_layout(height=300, margin=dict(t=40, b=40, l=10, r=10), template="room")
    chart(ef5, width="stretch")
    st.caption("Green dashed line: the moment it breaks. Each collision with N2 takes a share of the energy: a lot "
               "for a light ion, little for a heavy one. After enough of them the ion is as slow as the gas.")

# 200 ions with the same settings.
sts = many5["status"]
pct = {k: 100 * float(np.mean(sts == k)) for k in ("q3", "unbroken", "rod_prec", "rod_prod", "back", "stalled")}
t_q3 = many5["t_end_s"][sts == "q3"] * 1e6
st.markdown(f"**200 ions, the same settings:**")
mb = st.columns(4)
mb[0].metric("Fragment reaches Q3", f"{pct['q3']:.0f}%")
mb[1].metric("Lost on the rods", f"{pct['rod_prod'] + pct['rod_prec']:.0f}%",
             help=f"Fragment {pct['rod_prod']:.0f}%, precursor {pct['rod_prec']:.0f}%")
mb[2].metric("Unbroken / back / stalled", f"{pct['unbroken']:.0f} / {pct['back']:.0f} / {pct['stalled']:.0f}%")
mb[3].metric("Time in the cell (median)", f"{np.median(t_q3):.0f} µs" if t_q3.size else "-",
             help=f"The method pauses {inputs().acquisition.inter_scan_delay_ms:g} ms between transitions; a "
                  "fragment still in the cell when the next transition starts is lost, or lands in it")
mq = st.columns(3)
mq[0].metric("Low-mass cutoff", f"m/z {many5['cutoff_mz']:.0f}", help="q = 0.908 at this RF")
mq[1].metric(f"q of the precursor ({prec5:g})", f"{many5['q_prec']:.2f}")
mq[2].metric(f"q of the fragment ({prod5:g})", f"{many5['q_prod']:.2f}")
st.markdown(
    "- **Cell RF** is a trade-off. More volts give every ion a deeper cup, but raise the cutoff: light fragments "
    "go past q = 0.908 and are shaken out. Fewer volts keep them, but leave heavy fragments in shallow cups.\n"
    "- **With gas the edge is soft.** Fragments well inside the cutoff (q above about 0.7) are still knocked out by "
    "collisions at awkward moments of the RF. The practical cutoff is below 0.908.\n"
    "- **Light fragments scatter hardest.** Hitting N2 (28 Da), an m/z 49 fragment loses or redirects a big share "
    "of its speed; a heavy precursor barely deflects.\n"
    "- **Gas pressure** gives more collisions: more breaking, more cooling (focusing onto the axis), and more "
    "scattering. **No axial push** and cooled ions stall in the cell.")
st.caption("Assumed, not the Ultivo's: a quadrupole cell (Agilent does not publish the Vortex cell's geometry; a "
           "hexapole or octopole would have a softer cutoff), 4 mm, 1 MHz, 8 cm, the pressure, and a collision "
           "cross-section of 150 A^2 at 200 Da. Collisions are elastic hard spheres with N2 at 300 K; the fragment "
           "keeps the precursor's velocity. Which bond breaks and how often are not modelled. Illustration; not used "
           "by the signal model.")
sources_line(("yost_enke_1978", "dawson_1995", "gerlich_1992", "douglas_french_1992", "hoffmann_stroobant_2007",
              "skullerud_1968", "picache_2019"),
             label="Sources (the RF-only cell and its cutoff; collisional cooling and focusing; collision energy and the "
                   "centre-of-mass share; null-collision sampling; the cross-section's order of magnitude; geometry, "
                   "pressure and cell RF assumed)")
