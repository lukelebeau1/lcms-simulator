"""The control-room look: palette, page CSS and the chart template. docs/design.md.

Sand panels, violet-black ink, deep violet for pressed controls and the chart screens, saffron for the title band,
labels and anything that needs attention, teal for low. The theme colours and fonts are in .streamlit/config.toml;
this file adds what the theme cannot express (title band, tiles, annunciator, rail) and the plotly template.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

INK, SYS, LILAC, PALE = "#2E2433", "#574066", "#BA9DDB", "#EDE5F5"
CREAM, SAFFRON, SAFFRON_PALE, SAFFRON_EDGE = "#F3ECDD", "#D9A441", "#F3DFA8", "#B8862B"
TEAL, TEAL_LIGHT = "#3F8A8C", "#7CC4C6"
PANEL, SECTION, SEAM, ROOM, MUTED = "#E7DFD0", "#DED4C2", "#A89C86", "#CDC4B6", "#5C5262"
SCREEN, SCREEN_GRID = "#3A2C48", "#574066"

# Marks on a screen: lilac first, then saffron, light teal, cream, pale lilac. Mid violet stays off the screens
# (it merges with teal under deuteranopia).
SCREEN_COLORWAY = [LILAC, SAFFRON, TEAL_LIGHT, CREAM, PALE]

_axis = dict(automargin=True, gridcolor=SCREEN_GRID, zerolinecolor="#7A6690", linecolor="#7A6690", tickcolor="#7A6690",
             tickfont=dict(family="PT Mono, monospace", size=11, color=PALE),
             title=dict(font=dict(family="PT Sans Narrow, sans-serif", size=14, color=CREAM)))
_axis3d = {k: v for k, v in _axis.items() if k != "automargin"}   # 3D axes take no automargin
pio.templates["room"] = go.layout.Template(layout=dict(
    paper_bgcolor=SCREEN, plot_bgcolor=SCREEN, colorway=SCREEN_COLORWAY,
    font=dict(family="PT Sans Narrow, sans-serif", size=13, color=CREAM),
    title=dict(font=dict(family="PT Sans Narrow, sans-serif", size=15, color=CREAM)),
    xaxis=_axis, yaxis=_axis,
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=CREAM)),
    hoverlabel=dict(bgcolor=INK, bordercolor=SAFFRON, font=dict(family="PT Mono, monospace", color=CREAM)),
    coloraxis=dict(colorbar=dict(tickfont=dict(color=PALE), title=dict(font=dict(color=CREAM)))),
    annotationdefaults=dict(font=dict(color=CREAM)),
    polar=dict(bgcolor=SCREEN), ternary=dict(bgcolor=SCREEN),
    scene=dict(xaxis=_axis3d, yaxis=_axis3d, zaxis=_axis3d, bgcolor=SCREEN),
))
pio.templates.default = "room"

CSS = f"""
<style>
:root {{ --ink:{INK}; --sys:{SYS}; --lilac:{LILAC}; --pale:{PALE}; --cream:{CREAM}; --saffron:{SAFFRON};
  --saffron-pale:{SAFFRON_PALE}; --saffron-edge:{SAFFRON_EDGE}; --teal:{TEAL}; --panel:{PANEL}; --section:{SECTION};
  --seam:{SEAM}; --room:{ROOM}; --muted:{MUTED}; --screen:{SCREEN}; }}

/* Title band */
[data-testid="stMain"] h1 {{ background:var(--saffron); border:1px solid var(--saffron-edge); padding:6px 12px 4px;
  text-transform:uppercase; letter-spacing:.06em; font-size:1.9rem; font-weight:700; color:var(--ink); }}
[data-testid="stMain"] h1 a {{ display:none; }}

/* Section labels: numbered sections read as saffron caption strips */
[data-testid="stMain"] h2 {{ display:inline-block; background:var(--saffron-pale); border:1px solid var(--saffron);
  padding:3px 10px 1px; font-size:1.05rem; letter-spacing:.08em; text-transform:uppercase; font-weight:700;
  margin-top:1.2rem; }}
[data-testid="stMain"] h3 {{ font-size:1rem; letter-spacing:.08em; text-transform:uppercase; font-weight:700;
  border-bottom:1px solid var(--seam); padding-bottom:2px; }}
[data-testid="stMain"] h2 a, [data-testid="stMain"] h3 a {{ display:none; }}

/* Annunciator: the simulated warning and every other alert */
[data-testid="stAlertContainer"] {{ border:2px solid var(--ink); border-radius:0; }}
[data-testid="stAlertContentWarning"] p {{ font-family:"PT Mono", monospace; font-size:.85rem; letter-spacing:.04em;
  text-transform:uppercase; }}

/* Captions in muted ink */
[data-testid="stCaptionContainer"] {{ color:var(--muted); }}

/* Readout tiles */
[data-testid="stMetric"] {{ background:var(--cream); border:3px solid var(--ink); padding:6px 10px; }}
[data-testid="stMetricLabel"] p {{ font-size:.78rem; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); }}
[data-testid="stMetricValue"] {{ font-family:"PT Mono", monospace; font-size:1.6rem; }}

/* Push-buttons: cream with an ink rim; pressed (primary) is deep violet */
[data-testid="stBaseButton-secondary"] {{ background:var(--cream); border:2px solid var(--ink); font-family:"PT Mono", monospace; }}
[data-testid="stBaseButton-secondary"]:hover {{ border-color:var(--sys); color:var(--sys); }}
[data-testid="stBaseButton-primary"] {{ background:var(--sys); border:2px solid var(--sys); color:var(--cream);
  font-family:"PT Mono", monospace; }}

/* Expanders as panel sections */
[data-testid="stExpander"] details {{ background:var(--section); border:1px solid var(--seam); border-radius:0; }}
[data-testid="stExpander"] summary p {{ letter-spacing:.03em; }}

/* Sidebar: group headers as engraved labels */
[data-testid="stNavSectionHeader"] {{ text-transform:uppercase; letter-spacing:.08em; font-weight:700; color:var(--ink); }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ font-size:.95rem;
  text-transform:uppercase; letter-spacing:.08em; font-weight:700; }}

/* The rail: one backlit tile per station, this page's lit saffron */
.st-key-rail [data-testid="stColumn"] > div {{ background:var(--cream); border:2px solid var(--ink); padding:6px 8px;
  height:100%; }}
.st-key-rail [data-testid="stColumn"]:has(.st-key-rail-here) > div {{ background:var(--saffron); }}
.st-key-rail [data-testid="stMarkdownContainer"] p {{ font-family:"PT Mono", monospace; font-size:.85rem; }}
.st-key-rail [data-testid="stMarkdownContainer"] strong {{ font-family:"PT Sans Narrow", sans-serif; }}
.st-key-rail [data-testid="stPageLink"] p {{ font-family:"PT Sans Narrow", sans-serif; font-size:.9rem;
  white-space:nowrap; }}
.st-key-rail [data-testid="stPageLink"] a {{ padding-left:0; }}
</style>
"""


def apply() -> None:
    """Inject the page CSS. Call once per page run, before any content."""
    st.html(CSS)
