"""The trace: one portion of one compound followed from its reference standard to the detector.

Every stop holds the amount present, nominal and actual, and the share of the previous stop that reached it,
so a loss of signal against expectation can be charged to the step where it happened. The stops come from two
layers that already exist and are not changed here:

  bench (quant.ledger): the recipe run twice, nominal and actual. The path is walked back from the injection
      through the vessel that contributed most of the compound at each step: ampoule -> stock -> working
      standard -> tube -> vial -> column. Every step that touched a vessel on the path is a stop, so a solvent
      addition shows as a stop that moves no amount.
  instrument (lc, ms, and the pipeline's expected-detector conventions): the share of the peak inside the
      transition's acquisition window; the ion budget (spray share back-computed from the response factor,
      matrix factor, typical Q1 / collision cell / Q3 / multiplier shares, dwell share); dead time, drift,
      carryover and interference where the run conditions set them; then ions counted -> reported area.

Invariant (tested): the product of every stop's `loss` (actual / nominal of its share, actual / nominal of the
amount for the first stop) equals the last stop's actual / nominal. The nominal area equals the pipeline's
expected area for the same injection and transition.

Units: ng up to the column, ions from the spray on, area units at the end. Molecules and ions convert to ng
through the molecular weight (Trace.molecules, Trace.ng). Provenance per stop: the recipe's (measured) for
bench stops, model for shares the simulator computes, assumed for the typical ion-optics shares and the spray
share.

Not covered yet: crosstalk from other compounds, the internal standard's own trace, and the quantitation
after the area (ratio to the IS, curve, reported concentration). Faults that change the prep or calibration
inputs of a Scenario are not yet mirrored into the recipe; pass an edited recipe (quant.ledger.with_actual)
for bench faults until the ledger feeds the run.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np

from lcms_sim.io.loaders import InputSet
from lcms_sim.io.schemas import AcquisitionMethod, Analyte, PrepRecipe, RecipeStep, Transition, TransitionRole
from lcms_sim.lc.gradient import LCConditions
from lcms_sim.lc.peaks import peak_sigma
from lcms_sim.lc.retention import retention_time
from lcms_sim.ms.acquisition import Schedule, build_schedule
from lcms_sim.pipeline import Kind, RunConditions, default_sequence
from lcms_sim.quant.ledger import StepState, molecules, run_recipe
from lcms_sim.quant.matrix import MatrixProfile, profile_from_traces
from lcms_sim.quant.prep import analyte_vial_conc, on_column_ng, prep_values

Which = Literal["nominal", "actual"]


class Station(str, Enum):
    """Where the portion physically is. Stops of a trace are in this order."""

    storage = "storage"
    bench = "bench"
    injection = "injection"
    column = "column"
    source = "source"
    analyser = "analyser"
    detector = "detector"


STATIONS: tuple[Station, ...] = tuple(Station)

# Typical shares passed by the ion optics after the inlet (name, share, typical range). Assumed; the spray share
# is back-computed so that the whole chain reproduces the response factor (spray_fraction).
TYPICAL_DOWNSTREAM: tuple[tuple[str, float, str], ...] = (
    ("Q1 passes the precursor", 0.3, "0.2-0.6 at unit resolution"),
    ("Collision cell makes the quantifier product", 0.3, "0.1-0.6, depends on CE and the molecule"),
    ("Q3 passes the product", 0.3, "0.2-0.6 at unit resolution"),
    ("Electron multiplier registers the ion", 0.8, "0.5-0.9"),
)
_TYPICAL_STATION = (Station.analyser, Station.analyser, Station.analyser, Station.detector)
STAGE_KEYS = ("q1", "cell", "q3", "multiplier")           # Stop.key of the typical stages, in order
_TYPICAL_ELSEWHERE = ("neutralised on the rods or lost between them",
                      "left as the precursor or became another fragment",
                      "neutralised on the rods or lost between them",
                      "hit the multiplier without starting a cascade")


@dataclass(frozen=True)
class Stop:
    """One place the portion was, with what was there and what share of the previous stop arrived."""

    station: Station
    name: str
    unit: str                                # "ng", "ions" or "area units"
    nominal: float                           # amount present, as the procedure and method say
    actual: float                            # amount present, as it happened
    nominal_fraction: float                  # share of the previous stop that reached this one (1 at the first)
    actual_fraction: float                   # the same, as it happened (actual / nominal at the first stop)
    elsewhere: str                           # where the rest went, read after a share: "64% stays in the vial"
    provenance: str                          # measured, model or assumed
    step: int | None = None                  # recipe step index for bench stops
    volume_ul: tuple[float, float] | None = None   # vessel volume after the step (nominal, actual)
    key: str = ""                            # "step:<k>" for bench stops, else the stage: window, spray, q1, ...

    @property
    def loss(self) -> float:
        """actual / nominal of the share passing here: below 1, this step lost more than expected."""
        return _ratio(self.actual_fraction, self.nominal_fraction)


@dataclass(frozen=True)
class Actual:
    """What actually happened, where it differs from the nominal. A field left None means as nominal."""

    recipe: PrepRecipe | None = None            # the recipe with actual values (quant.ledger.with_actual); same steps
    inputs: InputSet | None = None              # for instance a scenario's inputs (changed analytes, acquisition)
    lc: LCConditions | None = None              # the gradient, void time and dwell time as they are
    run: RunConditions = dataclasses.field(default_factory=RunConditions)
    injection_index: int = 0                    # position in the run, for sensitivity drift
    previous_injection: str | None = None       # recipe injection injected just before, for carryover
    shares: Mapping[str, float] = dataclasses.field(default_factory=dict)   # actual share by stage key:
    # spray, q1, cell, q3, multiplier (the nominal keeps the typical value, so the difference is a loss there)

    @classmethod
    def from_scenario(cls, scenario, recipe: PrepRecipe | None = None, injection_index: int = 0,
                      previous_injection: str | None = None) -> Actual:
        """A fault-library Scenario: its inputs, LC conditions and run conditions."""
        return cls(recipe, scenario.inputs, scenario.lc_conditions(), scenario.run, injection_index,
                   previous_injection)


@dataclass(frozen=True)
class Trace:
    compound: str
    injection: str                           # recipe injection vessel, for instance "Column: Cal 3"
    transition: Transition
    molecular_weight: float
    stops: list[Stop]

    @property
    def nominal_area(self) -> float:
        return self.stops[-1].nominal

    @property
    def actual_area(self) -> float:
        return self.stops[-1].actual

    @property
    def ratio(self) -> float:
        """actual / nominal of the reported area."""
        return _ratio(self.actual_area, self.nominal_area)

    def molecules(self, stop: Stop, which: Which = "nominal") -> float | None:
        """Molecules (ng stops) or ions (ion stops) at a stop. None for the area."""
        x = getattr(stop, which)
        if stop.unit == "ng":
            return molecules(x, self.molecular_weight)
        return x if stop.unit == "ions" else None

    def ng(self, stop: Stop, which: Which = "nominal") -> float | None:
        """Mass at a stop; for ion stops the mass of that many molecules. None for the area."""
        x = getattr(stop, which)
        if stop.unit == "ng":
            return x
        return x / molecules(1.0, self.molecular_weight) if stop.unit == "ions" else None

    def losses(self, tolerance: float = 1e-9) -> list[Stop]:
        """Stops whose share differed from expectation, in order."""
        return [s for s in self.stops if abs(s.loss - 1.0) > tolerance]

    def first_departure(self, tolerance: float = 1e-9) -> Stop | None:
        """The first stop where the actual parted from the nominal."""
        lost = self.losses(tolerance)
        return lost[0] if lost else None

    def rows(self) -> list[dict]:
        """One plain dict per stop, for tables."""
        return [{"station": s.station.value, "stop": s.name, "unit": s.unit, "nominal": s.nominal,
                 "actual": s.actual, "nominal share": s.nominal_fraction, "actual share": s.actual_fraction,
                 "loss": s.loss, "where the rest went": s.elsewhere, "provenance": s.provenance,
                 "step": s.step} for s in self.stops]


def spray_fraction(a: Analyte, acq: AcquisitionMethod,
                   typical: tuple[tuple[str, float, str], ...] = TYPICAL_DOWNSTREAM) -> float:
    """Share of the molecules on column that the spray ionises and the inlet admits, implied by the response factor.

    The response factor gives area per ng; divided by the gain and by molecules per ng it is ions reaching the
    detector per molecule (as if always listening). Dividing out the typical downstream shares leaves what the
    spray and inlet must pass for the chain to reproduce the expected area. Assumed, like the shares it rests on.
    """
    if not a.molecular_weight:
        raise ValueError(f"{a.name} has no molecular weight; molecules cannot be counted")
    lumped = a.response_factor / acq.gain_area_per_ion / molecules(1.0, a.molecular_weight)
    return lumped / math.prod(f for _, f, _ in typical)


def build_trace(inputs: InputSet, recipe: PrepRecipe, injection: str, compound: str,
                transition: Transition | None = None, actual: Actual | None = None,
                profile: MatrixProfile | None = None) -> Trace:
    """Follow `compound` from its reference standard through `injection` (a recipe inject vessel) to the area.

    `transition` defaults to the compound's quantifier. `actual` holds everything that differed from the
    nominal; left out, the actual trace equals the nominal one.
    """
    act = actual or Actual()
    inp_a = act.inputs or inputs
    a_n, a_a = _analyte(inputs, compound), _analyte(inp_a, compound)
    if not a_n.molecular_weight:
        raise ValueError(f"{compound} has no molecular weight; molecules cannot be counted")
    tr_n = transition or _quantifier(inputs, compound)
    tr_a = _same_transition(inp_a, tr_n)
    cond_n = LCConditions.from_method(inputs.system, inputs.lc_method)
    cond_a = act.lc or LCConditions.from_method(inp_a.system, inp_a.lc_method)
    profile = profile or profile_from_traces(inputs.infusion_matrix, inputs.infusion_solvent)
    recipe_a = act.recipe or recipe
    if len(recipe_a.steps) != len(recipe.steps):
        raise ValueError("the actual recipe must have the same steps as the nominal one (quant.ledger.with_actual)")

    vessels_n, log_n = run_recipe(recipe, "nominal")
    vessels_a, log_a = run_recipe(recipe_a, "actual")
    if injection not in vessels_n:
        raise ValueError(f"{injection!r} is not an injection of the recipe")
    stops = _bench_stops(recipe, log_n, log_a, injection, compound)
    kind = _kind(inputs, injection)
    in_matrix = vessels_n[injection].matrix_g > 0
    per_ng = molecules(1.0, a_n.molecular_weight)
    run = act.run

    def push(station, name, unit, nominal, actual_, elsewhere, provenance, key):
        prev = stops[-1]
        pn, pa = prev.nominal, prev.actual
        if prev.unit == "ng" and unit == "ions":           # molecules -> ions: one for one
            pn, pa = pn * per_ng, pa * per_ng
        fn = _ratio(nominal, pn)
        fa = _ratio(actual_, pa) if pa > 0 else fn            # nothing arrived: nothing to charge here
        stops.append(Stop(station, name, unit, nominal, actual_, fn, fa, elsewhere, provenance, key=key))

    # Injection: carryover adds what the previous injection left in the needle, loop and column.
    ng_n, ng_a = stops[-1].nominal, stops[-1].actual
    if run.carryover_fraction and act.previous_injection:
        if act.previous_injection not in vessels_a:
            raise ValueError(f"{act.previous_injection!r} is not an injection of the recipe")
        carry = run.carryover_fraction * vessels_a[act.previous_injection].ng.get(compound, 0.0)
        push(Station.injection, f"Carried over from {act.previous_injection.removeprefix('Column: ')} "
             f"({100 * run.carryover_fraction:g}% of its amount)", "ng", ng_n, ng_a + carry, "", "model", "carryover")
        ng_a += carry

    # Column: the share of the peak inside the transition's acquisition window.
    tR_n = retention_time(a_n.log_kw, a_n.S, cond_n)
    tR_a = retention_time(a_a.log_kw, a_a.S, cond_a)
    sig_n = peak_sigma(a_n.log_kw, a_n.S, tR_n, cond_n, inputs.system.plate_number, inputs.system.extra_column_sigma_s)
    sig_a = peak_sigma(a_a.log_kw, a_a.S, tR_a, cond_a, inp_a.system.plate_number, inp_a.system.extra_column_sigma_s)
    in_n, in_a = _inside_window(tr_n, tR_n, sig_n), _inside_window(tr_a, tR_a, sig_a)
    push(Station.column, f"Eluted inside the window of {_label(tr_n)} "
         f"({tr_n.retention_time_min:g} +- {tr_n.window_min / 2:g} min)", "ng", ng_n * in_n, ng_a * in_a,
         "eluted while this transition was not scheduled", "model", "window")

    # Source: spray and inlet, then matrix suppression, then any drift in sensitivity.
    sp_n = spray_fraction(a_n, inputs.acquisition)
    sp_a = act.shares.get("spray", spray_fraction(a_a, inp_a.acquisition))
    ions_n, ions_a = stops[-1].nominal * per_ng * sp_n, stops[-1].actual * per_ng * sp_a
    push(Station.source, "Ionised in the spray and drawn into the inlet", "ions", ions_n, ions_a,
         "leaves in droplets that never desolvate, as neutrals, or misses the inlet", "assumed", "spray")
    extra = run.sample_matrix_scale if kind is Kind.sample else 1.0
    mf_n = float(profile.factor(tR_n)) if in_matrix else 1.0
    mf_a = float(profile.factor(tR_a, run.mf_scale.get(compound, 1.0) * extra)) if in_matrix else 1.0
    ions_n, ions_a = ions_n * mf_n, ions_a * mf_a
    push(Station.source, "Not suppressed by co-eluting matrix", "ions", ions_n, ions_a,
         "gave its charge to matrix ions in the same droplets", "model" if in_matrix else "model: no matrix in this vial",
         "matrix")
    sens = 1.0 + run.drift_per_injection * act.injection_index
    if sens != 1.0:
        ions_a *= sens
        push(Station.source, f"Instrument sensitivity at injection {act.injection_index} of the run", "ions",
             ions_n, ions_a, "lost to source fouling or tune drift", "model", "drift")

    # Analyser and multiplier: typical shares; a qualifier's cell share carries its relative response.
    rel_n = (tr_n.relative_response or 1.0) if tr_n.role is TransitionRole.qualifier else 1.0
    rel_a = (tr_a.relative_response or 1.0) if tr_a.role is TransitionRole.qualifier else 1.0
    for (name, f, _), station, elsewhere, key in zip(TYPICAL_DOWNSTREAM, _TYPICAL_STATION, _TYPICAL_ELSEWHERE,
                                                      STAGE_KEYS):
        fn, fa = f, act.shares.get(key, f)
        if key == "cell" and tr_n.role is TransitionRole.qualifier:
            name, fn, fa = "Collision cell makes this qualifier's product", fn * rel_n, fa * rel_a
        ions_n, ions_a = ions_n * fn, ions_a * fa
        push(station, name, "ions", ions_n, ions_a, elsewhere, "assumed", key)

    # Detector: dead time (when set), then the dwell: only ions arriving while this transition is measured count.
    tau = inp_a.acquisition.detector_dead_time_s
    if tau:
        kept = _dead_time_share(ions_a, sig_a, tau)
        ions_a *= kept
        push(Station.detector, f"Registered within the dead time ({1e6 * tau:g} us)", "ions", ions_n, ions_a,
             "arrived while the multiplier was still busy", "model", "dead_time")
    sched_n = build_schedule(inputs.transitions, inputs.acquisition, inputs.lc_method.run_time_min)
    sched_a = build_schedule(inp_a.transitions, inp_a.acquisition, inp_a.lc_method.run_time_min)
    d_n, cyc_n = _dwell_at(sched_n, inputs.transitions.index(tr_n), tR_n)
    d_a, cyc_a = _dwell_at(sched_a, inp_a.transitions.index(tr_a), tR_a)
    ions_n, ions_a = ions_n * d_n / cyc_n, ions_a * d_a / cyc_a
    push(Station.detector, f"Counted during the dwell ({d_n:.1f} ms of a {cyc_n:g} ms cycle"
         + (f"; actual {d_a:.1f} of {cyc_a:g}" if (d_a, cyc_a) != (d_n, cyc_n) else "") + ")", "ions",
         ions_n, ions_a, "arrived between dwells, while other transitions were measured", "model", "dwell")
    if kind is Kind.sample and tr_n.role is TransitionRole.quantifier and compound in run.interference_ug_kg:
        nom = prep_values(inputs.prep, "nominal")
        area_i = a_n.response_factor * on_column_ng(analyte_vial_conc(run.interference_ug_kg[compound], 1.0, nom), nom)
        ions_a += area_i / inp_a.acquisition.gain_area_per_ion * d_a / cyc_a
        push(Station.detector, f"Interference counted with the quantifier ({run.interference_ug_kg[compound]:g} ug/kg "
             "equivalent)", "ions", ions_n, ions_a, "", "model", "interference")

    # Reported area: the count scaled to a rate as if always listening, then to area units by the gain.
    gain_n, gain_a = inputs.acquisition.gain_area_per_ion, inp_a.acquisition.gain_area_per_ion
    area_n = ions_n * (cyc_n / d_n if d_n > 0 else 0.0) * gain_n
    area_a = ions_a * (cyc_a / d_a if d_a > 0 else 0.0) * gain_a
    push(Station.detector, f"Area reported (x cycle / dwell, x gain {gain_n:g} area units per ion)", "area units",
         area_n, area_a, "", "assumed", "area")
    return Trace(compound, injection, tr_n, float(a_n.molecular_weight), stops)


# ---------------------------------------------------------------------------
# Bench: the path through the vessels
# ---------------------------------------------------------------------------


def path_steps(recipe: PrepRecipe, log: list[StepState], injection: str, compound: str) -> list[int]:
    """Recipe step indices on the compound's way to `injection`, in order.

    Walks back from the inject step. In each vessel the entry step is the one that brought in most of the
    compound (a stock, a weighed sample, or a transfer from a vessel holding it); every step that touched the
    vessel from that entry until the portion left is on the path.
    """
    steps = recipe.steps
    k_inj = next((k for k, s in enumerate(steps) if s.op == "inject" and s.vessel == injection), None)
    if k_inj is None:
        raise ValueError(f"{injection!r} is not an injection of the recipe")
    path = [k_inj]
    cur, upto = steps[k_inj].source, k_inj
    while cur is not None:
        touched = [k for k in range(upto) if steps[k].vessel == cur]
        best, best_amt = None, 0.0
        for k in touched:
            amt = _brought_in(steps, log, k, compound)
            if amt > best_amt:
                best, best_amt = k, amt
        if best is None:                                   # the compound never entered this vessel
            path = touched + path
            break
        path = [k for k in touched if k >= best] + path
        s = steps[best]
        cur, upto = (s.source if s.op == "transfer" else None), best
    return path


def _brought_in(steps: list[RecipeStep], log: list[StepState], k: int, compound: str) -> float:
    """Amount of the compound step k added to its vessel (nominal run)."""
    s = steps[k]
    if s.op == "stock":
        return log[k].vessel.ng.get(compound, 0.0)
    if s.op == "sample":
        return s.sample_ug_kg.get(compound, 0.0) * float(s.mass_g.nominal)
    if s.op == "transfer":
        before = next((log[j].vessel.ng.get(compound, 0.0) for j in range(k - 1, -1, -1)
                       if steps[j].vessel == s.vessel), 0.0)
        return log[k].vessel.ng.get(compound, 0.0) - before
    return 0.0


def _bench_stops(recipe: PrepRecipe, log_n: list[StepState], log_a: list[StepState], injection: str,
                 compound: str) -> list[Stop]:
    stops: list[Stop] = []
    for k in path_steps(recipe, log_n, injection, compound):
        s, n, a = recipe.steps[k], log_n[k], log_a[k]
        amt_n, amt_a = n.vessel.ng.get(compound, 0.0), a.vessel.ng.get(compound, 0.0)
        if stops:
            fn = _ratio(amt_n, stops[-1].nominal)
            fa = _ratio(amt_a, stops[-1].actual) if stops[-1].actual > 0 else fn
        else:
            fn, fa = 1.0, _ratio(amt_a, amt_n)
        station = (Station.storage if s.op == "stock" else Station.injection if s.op == "inject" else Station.bench)
        stops.append(Stop(station, _describe(s), "ng", amt_n, amt_a, fn, fa, _elsewhere(s), recipe.provenance, k,
                          (n.vessel.volume_ul, a.vessel.volume_ul), f"step:{k}"))
    return stops


def _describe(s: RecipeStep) -> str:
    if s.op == "stock":
        return f"Ampoule: {s.vessel}"
    if s.op == "solvent":
        return f"Solvent into {s.vessel} ({_vol(s)} uL)"
    if s.op == "transfer":
        return f"{_vol(s)} uL of {s.source} into {s.vessel}"
    if s.op == "sample":
        m = s.mass_g
        g = f"{m.nominal:g}" + (f" (actual {m.actual:g})" if m.actual != m.nominal else "")
        return f"{g} g of matrix into {s.vessel}"
    if s.op == "extract":
        return f"Extract {s.vessel}"
    return f"Inject {_vol(s)} uL of {s.source}"


def _vol(s: RecipeStep) -> str:
    v = s.volume_ul
    return f"{v.nominal:g}" + (f" (actual {v.actual:g})" if v.actual != v.nominal else "")


def _elsewhere(s: RecipeStep) -> str:
    if s.op in ("transfer", "inject"):
        return f"stays in {s.source}"
    if s.op == "extract":
        return "left in the matrix, the pellet and the filter"
    return ""


# ---------------------------------------------------------------------------
# Instrument helpers
# ---------------------------------------------------------------------------


def _analyte(inputs: InputSet, compound: str) -> Analyte:
    a = next((x for x in inputs.analytes if x.name == compound), None)
    if a is None:
        raise ValueError(f"{compound!r} is not an analyte of the method")
    return a


def _quantifier(inputs: InputSet, compound: str) -> Transition:
    t = next((x for x in inputs.transitions if x.analyte == compound and x.role is TransitionRole.quantifier), None)
    if t is None:
        raise ValueError(f"{compound!r} has no quantifier transition")
    return t


def _same_transition(inputs: InputSet, tr: Transition) -> Transition:
    """The transition of another input set that measures the same ions (a scenario may add transitions)."""
    return next((x for x in inputs.transitions if (x.analyte, x.precursor_mz, x.product_mz)
                 == (tr.analyte, tr.precursor_mz, tr.product_mz)), tr)


def _label(tr: Transition) -> str:
    return f"{tr.precursor_mz:g} > {tr.product_mz:g} ({tr.role.value})"


def _kind(inputs: InputSet, injection: str) -> Kind:
    """What the run treats this injection as. Not in the run definition: a control (no sample-only effects)."""
    ref = injection.removeprefix("Column: ")
    return next((item.kind for item in default_sequence(inputs) if item.ref == ref), Kind.qc)


def _inside_window(tr: Transition, tR: float, sigma: float) -> float:
    """Share of a Gaussian peak inside the transition's acquisition window (the pipeline's expected-detector rule)."""
    lo, hi = tr.retention_time_min - tr.window_min / 2, tr.retention_time_min + tr.window_min / 2
    inside = _cdf((hi - tR) / sigma) - _cdf((lo - tR) / sigma)
    return inside if inside > 1e-9 else 0.0


def _cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _dwell_at(sched: Schedule, j: int, tR: float) -> tuple[float, float]:
    """Dwell (ms) of transition j in the cycle containing tR (0 if not scheduled), and the cycle time."""
    kc = int(np.clip(np.searchsorted(sched.cycle_starts, tR, side="right") - 1, 0, len(sched.cycle_starts) - 1))
    return (float(sched.dwell_ms[kc]) if sched.active[kc, j] else 0.0), float(sched.cycle_time_ms)


def _dead_time_share(ions: float, sigma_min: float, dead_time_s: float) -> float:
    """Share of a Gaussian peak of `ions` (arriving as if always listening) that a non-paralysable detector registers."""
    if ions <= 0:
        return 1.0
    s = 60.0 * sigma_min
    t = np.linspace(-5 * s, 5 * s, 1001)
    rate = ions / (s * math.sqrt(2 * math.pi)) * np.exp(-0.5 * (t / s) ** 2)
    seen = rate / (1.0 + rate * dead_time_s)
    return float(np.trapezoid(seen, t) / np.trapezoid(rate, t))


def _ratio(x: float, y: float) -> float:
    return x / y if y > 0 else 1.0
