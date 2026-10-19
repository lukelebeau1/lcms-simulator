"""Follow the analyte: the containers it sits in, one after another, from its ampoule to the counted ions.

The main page. Every card is one container, or one instrument stage treated as a container: a cartoon, the
amount inside (ng and molecules, ions past the spray), and the share of the previous container's amount that
got here, with the factors that moved that share off 100%. Factor settings live in this page's widgets and
reach every other page through the rail.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

import cartoons  # noqa: E402
from common import containers, followed, inputs, page, probe_links, recipe, sources_line, trace_now  # noqa: E402

from lcms_sim.factors import BENCH, STAGE, STAGES, Factor, bench_key, is_own_label, istd_note  # noqa: E402
from lcms_sim.quant.ledger import run_recipe  # noqa: E402
from lcms_sim.trace import Stop, Trace  # noqa: E402

page("Follow the analyte",
     "One container after another, from the reference standard to the ions counted. Each card shows what is "
     "in the container, what share of the previous container's amount got here, and the factors that move that "
     "share off 100%. Set a factor and the loss shows here and on every other page's rail.", show_rail=False)

inj, cmp = followed()
i, r, cat = inputs(), recipe(), containers()
t = trace_now()
if t is None:
    st.stop()
LOG_N = run_recipe(r)[1]                                   # the nominal ledger: which vessel holds the IS when
ISTD = next((a.internal_standard for a in i.analytes if a.name == cmp), None)
OWN_LABEL = is_own_label(cmp, ISTD)

COLOUR = "#f2a33a"
STAGE_NAME = dict(STAGES)
STAGE_OF = {"window": "column", "spray": "spray", "matrix": "spray", "drift": "spray", "q1": "q1", "cell": "cell",
            "q3": "q3", "multiplier": "multiplier", "dead_time": "multiplier", "dwell": "counting",
            "interference": "counting", "area": "counting"}
PROBE = {"transfer": [("views/bench_steps.py", "Bench steps: every pipetting step, editable"),
                      ("views/standards_extraction.py", "Standards and extraction")],
         "loop": [("views/bench_steps.py", "Bench steps")],
         "column": [("views/lc_gradient.py", "LC: gradient and retention: why it elutes when it does"),
                    ("views/lc_column.py", "LC: inside the column: the band and the C18 surface")],
         "spray": [("views/ms_ion_path.py", "MS: ion path: the spray close-up")],
         "capillary": [("views/ms_ion_path.py", "MS: ion path")],
         "q1": [("views/ms_quad_filter.py", "MS: how a quadrupole filters"), ("views/ms_ion_path.py", "MS: ion path: Q1 close-up")],
         "cell": [("views/ms_quad_filter.py", "MS: how a quadrupole filters: the collision cell")],
         "q3": [("views/ms_quad_filter.py", "MS: how a quadrupole filters")],
         "multiplier": [("views/ms_molecule_budget.py", "MS: molecules to response")],
         "counting": [("views/ms_acquisition.py", "MS: acquisition: dwell, cycle, points across the peak"),
                      ("views/ms_molecule_budget.py", "MS: molecules to response")]}


@dataclass
class Card:
    key: str
    name: str
    kind: str                      # cartoon kind
    where: str                     # factor `where`: transfer, loop, column, spray, ...
    stops: list[Stop] = field(default_factory=list)
    step_in: int | None = None     # the recipe step that moved the analyte into this container
    spec: object = None


def cards_for(trace: Trace) -> list[Card]:
    out: list[Card] = []
    for s in trace.stops:
        if s.step is not None:
            step = r.steps[s.step]
            if not out or out[-1].key != step.vessel:
                spec = cat.for_vessel(step.vessel)
                inject = step.op == "inject"
                name = f"Injection: {step.vessel.removeprefix('Column: ')}" if inject else step.vessel
                out.append(Card(step.vessel, name, spec.kind if spec else "vial_2ml", "loop" if inject else "transfer",
                                step_in=s.step if step.op in ("transfer", "inject") else None, spec=spec))
            out[-1].stops.append(s)
        elif s.key == "carryover":
            out[-1].stops.append(s)
        else:
            stage = STAGE_OF[s.key]
            if stage == "q1" and (not out or out[-1].key != "capillary"):
                out.append(Card("capillary", "Capillary", "capillary", "capillary"))
            if not out or out[-1].key != stage:
                out.append(Card(stage, STAGE_NAME[stage], stage, stage))
            out[-1].stops.append(s)
    return out


def amount_text(stop: Stop, which: str) -> str:
    x = getattr(stop, which)
    if stop.unit == "ng":
        mass = f"{x / 1000:.4g} ug" if x >= 1000 else f"{x:.4g} ng"
        return f"{mass} ({t.molecules(stop, which):.3g} molecules)"
    if stop.unit == "ions":
        return f"{x:.3g} ions"
    return f"{x:.4g} area units"


def setting_key(card: Card, f: Factor) -> str:
    """Bench factors are keyed by step; a compound-scoped one also by the followed compound, so it sticks to it."""
    if card.where == "transfer" or (card.where == "loop" and f.kind == "volume"):
        return bench_key(card.step_in, f.key, cmp if f.scope == "compound" else None)
    return f.key


def istd_present(card: Card) -> bool:
    """Whether the IS shares the loss: it is in the vessel drawn from at this step, or the portion is past the vial."""
    if ISTD is None:
        return False
    if card.step_in is not None:
        src = LOG_N[card.step_in].source
        return src is not None and ISTD in src.ng
    return True


def factors_for(card: Card) -> list[Factor]:
    if card.where == "transfer":
        return list(BENCH)
    if card.where == "loop":
        return [f for f in BENCH if f.kind == "volume"] + [f for f in STAGE if f.where == "loop"]
    return [f for f in STAGE if f.where == card.where]


def default_for(card: Card, f: Factor) -> float:
    if f.kind == "volume":
        return float(r.steps[card.step_in].volume_ul.nominal)
    return f.default


def state_text(card: Card, f: Factor) -> str:
    """One phrase on the factor's current setting."""
    sk = setting_key(card, f)
    v = st.session_state.get("fv_" + sk)
    d = default_for(card, f)
    if f.kind == "check":
        return f"on, {v if v is not None else f.default:g} {f.unit}" if st.session_state.get("fw_" + sk) else "off"
    v = d if v is None else float(v)
    return f"{v:g} {f.unit}" + ("" if v == d else " (set)")


def controls(card: Card, f: Factor) -> None:
    sk = setting_key(card, f)
    d = default_for(card, f)
    init = {} if ("fv_" + sk) in st.session_state else {"value": d}    # a preset state wins over the default
    if f.kind == "check":
        on = st.checkbox(f.name, key="fw_" + sk)
        st.number_input(f.unit, min_value=f.low, max_value=f.high, step=f.step, key="fv_" + sk, disabled=not on, **init)
    elif f.kind == "volume":
        st.number_input(f.unit, min_value=0.0, max_value=max(10.0 * d, 1.0), step=max(d / 100, 0.1),
                        key="fv_" + sk, help=f"The step says {d:g} uL", **init)
    else:
        st.number_input(f.unit, min_value=f.low, max_value=f.high, step=f.step, key="fv_" + sk, **init)
    who = "**Compound-specific**" if f.scope == "compound" else "**Same for every compound**"
    st.markdown(f"{who}: {f.scope_text.split(': ', 1)[1]} {istd_note(f, ISTD, istd_present(card), OWN_LABEL)}")
    st.markdown(f.theory)
    st.caption(f"Provenance: {f.provenance}")
    sources_line(f.sources)


cards = cards_for(t)
with st.sidebar:
    st.subheader("Containers")
    st.markdown("\n".join(f"- [{c.name}](#c{n})" for n, c in enumerate(cards)))



def pct(x: float) -> str:
    """A share as a percentage: three figures, four near 100 so 99.99% does not round to 100%."""
    return f"{x:.4g}%" if x > 99 else f"{x:.3g}%"


def stop_table(stops: list[Stop]) -> str:
    """One row per stop: what makes it through on one side, what is lost and where on the other."""
    rows = ["| Makes it through | Lost |", "|---|---|"]
    for s in stops:
        fn, fa = s.nominal_fraction, s.actual_fraction
        changed = abs(s.loss - 1) > 1e-6
        if s.unit == "area units":
            through = f"{s.name}: x{fn:.3g} area units per counted ion" + (
                f" :red[(actual x{fa:.3g})]" if changed else "")
            lost = "nothing: a scale, not a share"
        elif fn > 1 + 1e-9:
            through = f"{s.name}: **100%** plus {pct(100 * (fn - 1))} added" + (
                f" :red[(actual plus {pct(100 * (fa - 1))})]" if changed else "")
            lost = "nothing"
        else:
            through = f"{s.name}: **{pct(100 * fn)}**" + (
                f" :red[(actual {pct(100 * fa)}, x{s.loss:.3g})]" if changed else "")
            if fn > 1 - 1e-9 and not changed:
                lost = "nothing"
            else:
                lost = pct(100 * (1 - fn)) + (f" {s.elsewhere}" if s.elsewhere else "")
                if changed:
                    lost += f" :red[(actual {pct(100 * (1 - fa))})]"
        rows.append(f"| {through} | {lost} |")
    return "\n".join(rows)


def lost_text(ratio: float) -> str:
    if abs(ratio - 1) <= 1e-6:
        return "nothing lost to the factors set"
    return (f"{100 * (1 - ratio):.3g}% lost to the factors set" if ratio < 1
            else f"{100 * (ratio - 1):.3g}% gained from the factors set")


to_loop, counted = 1.0, 1.0                       # by design: the dilution to the injection, then the instrument
for s in t.stops:
    if s.unit == "ng" and s.key.startswith("step:"):
        to_loop *= s.nominal_fraction
    elif s.unit != "area units":
        counted *= s.nominal_fraction
# Streamlit pulls every markdown block up by 1rem to cancel its last paragraph's margin; this block is a div with no
# such margin, so it gets one back, or the bar's bottom rule lands on the text's last line. --panel is the page
# colour (style.py), so the cards scroll under the bar instead of showing through it.
st.markdown("""<style>
div[data-testid="stElementContainer"]:has(div.follow-total) {position: sticky; top: 3.75rem; z-index: 99;
  background: var(--panel); padding: 4px 0; border-bottom: 1px solid var(--seam);}
div.follow-total {margin-bottom: 1rem;}
</style>""", unsafe_allow_html=True)
st.markdown(f'<div class="follow-total"><b>{cmp} in {inj.removeprefix("Column: ")}</b>, {len(cards)} containers. '
            f"By design {100 * to_loop:.3g}% of the ampoule's amount is injected and 1 in {1 / counted:,.0f} of the "
            f"molecules injected is counted; actual <b>x{t.ratio:.3g}</b> of expected, {lost_text(t.ratio)}.</div>",
            unsafe_allow_html=True)

cum_share, cum_ratio = 1.0, 1.0
for n, c in enumerate(cards):
    prev = cards[n - 1] if n else None
    last = c.stops[-1] if c.stops else prev.stops[-1]
    for s in c.stops:
        if s.unit != "area units":
            cum_share *= s.nominal_fraction
        cum_ratio *= s.loss
    col_pic, col_amt, col_in = st.columns([1.0, 1.4, 2.6], gap="medium")

    with col_pic:
        n_dots = cartoons.dots_for(t.molecules(last, "actual") or 0.0)
        if c.spec is not None or c.kind == "loop":
            vol = last.volume_ul[1] if last.volume_ul else 0.0
            cap = (c.spec.capacity_ml * 1000) if (c.spec and c.spec.capacity_ml) else max(vol, 1.0)
            caption = c.spec.location if c.spec else ""
            st.markdown(cartoons.vessel(c.kind, vol / cap, n_dots, COLOUR, caption), unsafe_allow_html=True)
        else:
            st.markdown(cartoons.stage(c.kind, n_dots, COLOUR), unsafe_allow_html=True)

    with col_amt:
        st.subheader(c.name, anchor=f"c{n}")
        if c.spec is not None:
            bits = [b for b in (c.spec.location, c.spec.solvent,
                                f"{c.spec.capacity_ml:g} mL" if c.spec.capacity_ml else "") if b]
            st.caption(" · ".join(bits) + (f". {c.spec.note}" if c.spec.note else ""))
        st.markdown(f"**{amount_text(last, 'nominal')}**")
        ratio = last.actual / last.nominal if last.nominal > 0 else 1.0
        if abs(ratio - 1) > 1e-6:
            st.markdown(f":red[actual {amount_text(last, 'actual')}, x{ratio:.3g} of expected]")
        if last.volume_ul and last.unit == "ng":
            v_n, v_a = last.volume_ul
            conc = 1000 * last.nominal / v_n if v_n > 0 else 0.0
            st.caption(f"{v_n:g} uL" + (f" (actual {v_a:g})" if v_a != v_n else "") + f", {conc:.4g} ng/mL")
        tally = f"So far: {100 * cum_share:.3g}% of the ampoule's amount by design"
        tally += (f"; :red[x{cum_ratio:.3g} of expected, {lost_text(cum_ratio)}]" if abs(cum_ratio - 1) > 1e-6
                  else "; as expected")
        st.caption(tally)

    with col_in:
        if prev is None:
            st.markdown("**Start.** The reference standard as received: the certificate's concentration and volume.")
        elif not c.stops:
            st.markdown(f"**100%** of what left {prev.name} passes. No factors modelled here yet: the spray share "
                        "already includes the inlet.")
        else:
            share_n = share_a = loss = 1.0
            for s in c.stops:
                share_n *= s.nominal_fraction
                share_a *= s.actual_fraction
                loss *= s.loss
            text = f"**{100 * share_n:.3g}%** of what was in {prev.name} is here"
            if abs(loss - 1) > 1e-6:
                text += f" :red[(actual {100 * share_a:.3g}%, x{loss:.3g} of expected)]"
            st.markdown(text)
            st.markdown(stop_table(c.stops))
        fs = factors_for(c) if (prev is not None or c.where == "loop") else []
        if fs:
            st.markdown("Loss factors: " + " · ".join(
                f"{f.name.split(':')[0]}{'*' if f.scope == 'compound' else ''} ({state_text(c, f)})" for f in fs)
                + ("  \n\\* compound-specific" if any(f.scope == "compound" for f in fs) else ""))
            with st.expander("Set the factors and read the theory"):
                for tab, f in zip(st.tabs([f.name.split(":")[0] for f in fs]), fs):
                    with tab:
                        controls(c, f)
                probe_links(PROBE.get(c.where, []))
    st.divider()
