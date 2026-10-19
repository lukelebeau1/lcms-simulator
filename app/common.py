"""Shared pieces for every view: cached inputs, the seed control, and the 'simulated' and provenance labels."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from streamlit.errors import StreamlitAPIException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lcms_sim import factors as F  # noqa: E402
from lcms_sim.io.loaders import InputSet, load_containers, load_input_set, load_prep_recipe, method_info  # noqa: E402
from lcms_sim.io.schemas import ContainerCatalogue, MethodInfo, PrepRecipe  # noqa: E402
from lcms_sim.paths import EXAMPLES  # noqa: E402
from lcms_sim.quant.ledger import RecipeError, with_actual  # noqa: E402
from lcms_sim.quant.matrix import MatrixProfile, profile_from_traces  # noqa: E402
from lcms_sim.references import markdown_cite  # noqa: E402
from lcms_sim.rng import make_rng, new_seed  # noqa: E402
from lcms_sim.trace import STATIONS, Station, Trace, build_trace  # noqa: E402
import style  # noqa: E402  (registers the chart template; page() puts the CSS in)

PROVENANCE_ICON = {"measured": "measured", "fitted": "fitted", "assumed": "assumed"}


def local_mode() -> bool:
    """True when the app runs on the user's own machine (scripts/run_app.py sets LCMS_SIM_LOCAL=1). Anywhere else,
    a hosted copy shared by many visitors, nothing is read from or written to data/lab: uploads stay in the
    visitor's session and results come back as downloads."""
    return os.environ.get("LCMS_SIM_LOCAL") == "1"


def input_folder() -> Path:
    """Where the app reads its inputs: the bundled example, or (locally) a lab input set chosen on the Import page."""
    if not local_mode():
        return EXAMPLES
    return Path(st.session_state.get("input_folder", EXAMPLES))


def _in_inputs(name: str) -> Path:
    """A companion file (recipe, containers) from the input folder, or the example's when it has none."""
    p = input_folder() / name
    return p if p.exists() else EXAMPLES / name


@st.cache_resource
def _inputs(folder: str) -> InputSet:
    return load_input_set(folder)


def inputs() -> InputSet:
    return _inputs(str(input_folder()))


@st.cache_resource
def _recipe(path: str) -> PrepRecipe:
    return load_prep_recipe(path)


def recipe() -> PrepRecipe:
    return _recipe(str(_in_inputs("prep_recipe.yaml")))


@st.cache_resource
def _acceptance(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def acceptance() -> dict:
    """The method's acceptance criteria (acceptance.yaml): r2, level and QC recovery windows, ion ratio tolerance."""
    return _acceptance(str(_in_inputs("acceptance.yaml")))


@st.cache_resource
def _containers(path: str) -> ContainerCatalogue:
    return load_containers(path)


def containers() -> ContainerCatalogue:
    return _containers(str(_in_inputs("containers.yaml")))


@st.cache_resource
def matrix_profile() -> MatrixProfile:
    i = inputs()
    return profile_from_traces(i.infusion_matrix, i.infusion_solvent)


def page(title: str, what: str, station: Station | None = None, show_rail: bool = True) -> None:
    """Title, the simulated warning, the intro, and the rail with this page's station marked."""
    st.set_page_config(page_title=title, layout="wide")
    style.apply()
    st.title(title)
    st.warning("SIMULATED. Not validated for reporting. For learning and planning experiments only.", icon=None)
    st.caption(what)
    if show_rail:
        rail(station)


def probe_links(pages: list[tuple[str, str]]) -> None:
    """Links to the pages that go deeper (skipped when a view runs on its own, outside the navigation)."""
    for path, label in pages:
        try:
            st.page_link(path, label=label)
        except StreamlitAPIException:
            pass


def sources_line(keys, label: str = "Sources") -> None:
    """A caption of short citations with links, or the assumed label when a claim cites nothing."""
    if keys:
        st.caption(f"{label}: {markdown_cite(keys)}")
    else:
        st.caption(f"{label}: none; assumed (see References)")


def seed_control():
    """Sidebar seed: shown, fixable, or redrawn."""
    if "seed" not in st.session_state:
        st.session_state.seed = new_seed()
    with st.sidebar:
        st.subheader("Random seed")
        st.session_state.seed = int(st.number_input("Seed", min_value=0, max_value=2**32 - 1,
                                                    value=st.session_state.seed, step=1))
        if st.button("New seed"):
            st.session_state.seed = new_seed()
            st.rerun()
    return make_rng(st.session_state.seed), st.session_state.seed


def provenance_table(rows: list[tuple[str, str, str]]) -> None:
    """rows of (input, value, provenance)."""
    with st.expander("Where these inputs come from"):
        st.dataframe(pd.DataFrame(rows, columns=["Input", "Value", "Provenance"]), hide_index=True,
                     width="stretch")


@st.cache_resource
def _method(folder: str) -> MethodInfo:
    return method_info(folder)


def method() -> MethodInfo:
    """What the loaded input set is: its name, sources and where each input comes from (method.yaml)."""
    return _method(str(input_folder()))


def method_sources() -> tuple[str, ...]:
    """Reference keys of the loaded method, for a page's Sources line."""
    return tuple(method().sources)


def method_provenance(i: InputSet) -> list[tuple[str, str, str]]:
    rows = [(r.input, r.value, r.provenance) for r in method().provenance]
    return rows or [("Inputs", str(input_folder()), "see the files")]


def targets(i: InputSet) -> list[str]:
    return [a.name for a in i.analytes if not a.is_internal_standard]


class _BuiltFigure(go.Figure):
    """A figure whose plain-dict spec is already complete: hands it over without Plotly's deep copy."""

    def __init__(self, spec: dict):
        super().__init__()
        self._spec = spec

    def to_dict(self) -> dict:
        return self._spec


def plotly_spec(spec: dict, **kwargs) -> None:
    """st.plotly_chart for a finished dict spec (animations carry hundreds of frames).

    Streamlit converts a go.Figure with to_dict(), which deep-copies every frame, and validates a
    plain dict property by property; either costs over a second for the animated views. The specs
    built here come from validated figures plus plain frame data, so both steps are skipped.
    """
    layout = spec.setdefault("layout", {})
    for k in ("paper_bgcolor", "plot_bgcolor"):
        layout.setdefault(k, style.SCREEN)
    layout.setdefault("font", {}).setdefault("color", style.CREAM)
    st.plotly_chart(_BuiltFigure(spec), theme=None, **kwargs)


def chart(fig: go.Figure, **kwargs) -> None:
    """st.plotly_chart on a screen: the room template, and the background and base font colour set on the figure
    itself, because Streamlit fills in any the layout leaves unset from the page theme (the template's are ignored);
    the page's ink would draw button labels and text marks dark on the dark screen."""
    fig.update_layout(paper_bgcolor=fig.layout.paper_bgcolor or style.SCREEN,
                      plot_bgcolor=fig.layout.plot_bgcolor or style.SCREEN,
                      font_color=fig.layout.font.color or style.CREAM)
    st.plotly_chart(fig, theme=None, **kwargs)


# ---------------------------------------------------------------------------
# The through-line: one portion of one compound, followed on every page
# ---------------------------------------------------------------------------

DEFAULT_INJECTION, DEFAULT_COMPOUND = "Column: Cal 8", "Myclobutanil"
STATION_PAGE = {                                   # where each station is explained
    Station.storage: ("views/bench_steps.py", "Bench steps"),
    Station.bench: ("views/bench_steps.py", "Bench steps"),
    Station.injection: ("views/bench_steps.py", "Bench steps"),
    Station.column: ("views/lc_column.py", "Inside the column"),
    Station.source: ("views/ms_ion_path.py", "Ion path"),
    Station.analyser: ("views/ms_quad_filter.py", "Quadrupole"),
    Station.detector: ("views/ms_acquisition.py", "Acquisition"),
}


def injections() -> list[str]:
    """The recipe's injections in order (vessel names 'Column: ...')."""
    return [s.vessel for s in recipe().steps if s.op == "inject"]


def followable() -> list[str]:
    """Target compounds the trace can count: they need a molecular weight."""
    return [a.name for a in inputs().analytes if not a.is_internal_standard and a.molecular_weight]


def follow_controls() -> None:
    """Sidebar 'Follow' pickers, drawn by the entry file so they sit on every page."""
    cols, names = injections(), followable()
    with st.sidebar:
        st.subheader("Follow")
        st.selectbox("Injection", cols, key="follow_injection", format_func=lambda c: c.removeprefix("Column: "),
                     index=cols.index(DEFAULT_INJECTION) if DEFAULT_INJECTION in cols else 0,
                     help="The portion every page follows: this compound in this injection, from its ampoule "
                          "to the reported area. Pages take it as their default and can pick another locally.")
        st.selectbox("Compound", names, key="follow_compound",
                     index=names.index(DEFAULT_COMPOUND) if DEFAULT_COMPOUND in names else 0)
        if input_folder() != EXAMPLES:
            st.caption(f"Inputs: {input_folder()} (lab input set; the Import page switches back)")


def followed() -> tuple[str, str]:
    """(injection, compound) being followed; the defaults when a view runs without the entry file."""
    cols, names = injections(), followable()
    inj = st.session_state.get("follow_injection") or (DEFAULT_INJECTION if DEFAULT_INJECTION in cols else cols[0])
    cmp = st.session_state.get("follow_compound") or (DEFAULT_COMPOUND if DEFAULT_COMPOUND in names else names[0])
    return inj, cmp


def follow_index(options: list, which: str = "compound") -> int:
    """Index of the followed compound (or injection) in a view's own option list, 0 if it is not there."""
    inj, cmp = followed()
    value = cmp if which == "compound" else inj
    return options.index(value) if value in options else 0


def current_settings() -> dict[str, float]:
    """Factor settings from the Follow page's widgets, as factors.apply expects them.

    A number or volume factor is the widget "fv_<key>"; a check factor is "fw_<key>" (on) with its share in
    "fv_<key>". Bench keys carry the step index ("12:pipette") and, for a compound-scoped factor, the compound
    ("12:precipitated@Bifenthrin"); stage keys are bare ("q1_share").
    """
    out: dict[str, float] = {}
    for key, val in st.session_state.items():
        if not isinstance(key, str) or not key.startswith("fv_") or val is None:
            continue
        sk = key[3:]
        f = F.factor_of(sk)
        if f is None or (f.kind == "check" and not st.session_state.get("fw_" + sk)):
            continue
        out[sk] = float(val)
    return out


@st.cache_data(show_spinner=False, max_entries=64)
def _trace(injection: str, compound: str, edits: tuple[tuple[int, float], ...],
           settings: tuple[tuple[str, float], ...]) -> Trace:
    r = recipe()
    cols = injections()
    k = cols.index(injection)
    r_edit = with_actual(r, dict(edits)) if edits else r
    actual = F.apply(inputs(), r_edit, dict(settings), compound, k, cols[k - 1] if k else None)
    return build_trace(inputs(), r, injection, compound, actual=actual, profile=matrix_profile())


def trace_now() -> Trace | None:
    """The followed portion's trace; the actual run carries the Bench steps edits and the Follow page's factors."""
    inj, cmp = followed()
    edits = st.session_state.get("bench_actual", {})
    try:
        return _trace(inj, cmp, tuple(sorted((int(k), float(v)) for k, v in edits.items())),
                      tuple(sorted(current_settings().items())))
    except (RecipeError, ValueError) as e:
        st.warning(f"No trace for {cmp} in {inj.removeprefix('Column: ')}: {e}")
        return None


def _amount(x: float, unit: str) -> str:
    s = f"{x:.3g}" if (x >= 1e5 or 0 < x < 1e-2) else f"{x:.4g}"
    return f"{s} {unit}"


def rail(here: Station | None = None) -> Trace | None:
    """The through-line: the followed portion at every station, this page's station marked.

    Each station shows the amount at its last stop (nominal) and, when the actual differs, the cumulative
    actual / nominal there. The expander lists every stop with ng, molecules or ions, and the share passing.
    """
    t = trace_now()
    if t is None:
        return None
    inj, cmp = t.injection.removeprefix("Column: "), t.compound
    last = {s.station: s for s in t.stops}
    for col, station in zip(st.container(key="rail").columns(len(STATIONS)), STATIONS):
        col = col.container(key="rail-here" if station is here else f"rail-{station.value}")   # style.py lights it
        s = last.get(station)
        head = f"**{station.value.capitalize()}**"
        if s is None:
            col.markdown(f"{head}  \n-")
            continue
        text = f"{head}  \n{_amount(s.nominal, s.unit)}"
        ratio = s.actual / s.nominal if s.nominal > 0 else 1.0
        if abs(ratio - 1.0) > 1e-6:
            text += f"  \n:red[x{ratio:.3g} of expected]"
        col.markdown(text)
        if station is not here and station in STATION_PAGE:
            try:
                col.page_link(STATION_PAGE[station][0], label=STATION_PAGE[station][1])
            except StreamlitAPIException:          # a view run on its own, outside the entry file's navigation
                pass
    with st.expander(f"Every stop for {cmp} in {inj}: ng, molecules or ions, and the share passing"):
        rows = t.rows()
        for row, s in zip(rows, t.stops):
            n = t.molecules(s)
            row["molecules or ions"] = float("nan") if n is None else n   # keep the column numeric
        order = ["station", "stop", "nominal", "actual", "unit", "molecules or ions", "nominal share",
                 "actual share", "loss", "where the rest went", "provenance"]
        numeric = ("nominal", "actual", "molecules or ions", "nominal share", "actual share", "loss")
        st.dataframe(pd.DataFrame(rows)[order], hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4g") for c in numeric})
        st.caption("Nominal: what the procedure and method say. Actual: with the factors set on Follow and the edits "
                   "made on Bench steps. "
                   "Loss: actual / nominal of the share passing that stop; the product over all stops is the "
                   "final ratio. Shares after the inlet are typical values (assumed), except the matrix factor, "
                   "the window and the dwell, which the model computes.")
    return t
