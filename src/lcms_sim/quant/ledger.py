"""Preparation ledger: every vessel's volume and the amount of each compound in it, step by step.

Amounts (ng) are conserved by pipetting, so a transfer of v uL from a vessel holding V uL moves
v / V of every compound in it. Concentrations are derived (ng / uL = ug / mL = ppm in solution),
never stored, so a dilution is just a transfer plus solvent:

    1000 uL of a 10 ng/uL standard holds 10 000 ng. Take 100 uL: 1000 ng. Add 900 uL solvent:
    1000 ng in 1000 uL = 1 ng/uL, a 1:10 dilution.

The recipe is run twice: with each step's nominal values (what the SOP says) and with its actual
values (what happened). Signal follows the actual run; reported results use the nominal one. A
pipetting error is a changed `actual` on one step, and the ledger carries it to every vessel
downstream. A transfer's efficiency is the share of the nominal amount the drawn volume carries:
one value for every compound (the pipette), and on top of it one per named compound (an unmixed,
precipitated or adsorbed component), so a compound-specific loss leaves the other compounds'
amounts untouched. The nominal run reproduces the closed-form prep and calibration values (tested).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from lcms_sim.io.schemas import NominalActual, PrepRecipe, RecipeStep

AVOGADRO = 6.02214076e23
Which = Literal["nominal", "actual"]


@dataclass(frozen=True)
class Vessel:
    name: str
    volume_ul: float = 0.0
    ng: Mapping[str, float] = field(default_factory=dict)
    matrix_g: float = 0.0

    def conc_ng_ml(self, compound: str) -> float:
        """ng/mL (= ng/uL x 1000). Zero for an empty vessel."""
        return 1000.0 * self.ng.get(compound, 0.0) / self.volume_ul if self.volume_ul > 0 else 0.0


@dataclass(frozen=True)
class StepState:
    """The vessel a step changed, and its source if any, right after the step."""

    index: int
    step: RecipeStep
    vessel: Vessel
    source: Vessel | None


class RecipeError(ValueError):
    pass


def _value(na, which: Which) -> float:
    return float(getattr(na, which))


def _add(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, float]:
    out = dict(a)
    for c, x in b.items():
        out[c] = out.get(c, 0.0) + x
    return out


def _efficiency(s: RecipeStep, compound: str, which: Which) -> float:
    """Share of the nominal amount of `compound` the drawn volume carries: the step's uniform efficiency times
    the compound's own, 1 for each when unset."""
    e = _value(s.efficiency, which) if s.efficiency is not None else 1.0
    own = s.efficiency_by_compound.get(compound)
    return e * (_value(own, which) if own is not None else 1.0)


def run_recipe(recipe: PrepRecipe, which: Which = "nominal") -> tuple[dict[str, Vessel], list[StepState]]:
    """Execute every step in order. Returns the final vessels and the state after each step."""
    vessels: dict[str, Vessel] = {}
    log: list[StepState] = []
    for k, s in enumerate(recipe.steps):
        v = vessels.get(s.vessel, Vessel(s.vessel))
        src = None
        if s.op == "stock":
            vol = _value(s.volume_ul, which)
            v = Vessel(s.vessel, vol, {c: x * vol for c, x in s.contents_ng_ul.items()})
        elif s.op == "solvent":
            v = Vessel(v.name, v.volume_ul + _value(s.volume_ul, which), v.ng, v.matrix_g)
        elif s.op in ("transfer", "inject"):
            if s.source not in vessels:
                raise RecipeError(f"step {k + 1}: {s.source!r} does not exist yet")
            src = vessels[s.source]
            vol = _value(s.volume_ul, which)
            if vol > src.volume_ul * (1 + 1e-9):
                raise RecipeError(f"step {k + 1}: taking {vol:g} uL from {src.name!r}, which holds "
                                  f"{src.volume_ul:g} uL")
            f = vol / src.volume_ul if src.volume_ul > 0 else 0.0
            share = {c: f * _efficiency(s, c, which) for c in src.ng}
            moved, matrix = {c: a * share[c] for c, a in src.ng.items()}, src.matrix_g * f
            src = Vessel(src.name, src.volume_ul - vol, {c: a * (1 - share[c]) for c, a in src.ng.items()},
                         src.matrix_g - matrix)
            vessels[src.name] = src
            v = Vessel(v.name, v.volume_ul + vol, _add(v.ng, moved), v.matrix_g + matrix)
        elif s.op == "sample":
            m = _value(s.mass_g, which)
            native = {c: ug_kg * m for c, ug_kg in s.sample_ug_kg.items()}   # ug/kg = ng/g
            v = Vessel(v.name, v.volume_ul, _add(v.ng, native), v.matrix_g + m)
        elif s.op == "extract":
            if which == "actual" and s.recovery:
                v = Vessel(v.name, v.volume_ul, {c: a * s.recovery.get(c, 1.0) for c, a in v.ng.items()}, v.matrix_g)
        vessels[s.vessel] = v
        log.append(StepState(k, s, v, src))
    return vessels, log


def lineage(recipe: PrepRecipe, vessel: str) -> list[int]:
    """Indices of the steps that put anything into `vessel`, directly or through its sources, in order."""
    wanted, keep = {vessel}, set()
    for k in range(len(recipe.steps) - 1, -1, -1):
        s = recipe.steps[k]
        if s.vessel in wanted:
            keep.add(k)
            if s.source:
                wanted.add(s.source)
    return sorted(keep)


def molecules(ng: float, molecular_weight: float) -> float:
    """Number of molecules in `ng` of a compound of molar mass `molecular_weight` (g/mol)."""
    return ng * 1e-9 / molecular_weight * AVOGADRO


def with_efficiency(recipe: PrepRecipe, efficiency: Mapping[int, float | Mapping[str, float]]) -> PrepRecipe:
    """The recipe with the actual transfer efficiency of some transfer or inject steps set.

    step index -> share sets the step's uniform efficiency (every compound); step index -> {compound: share}
    sets the named compounds' own efficiencies and leaves the others alone. The two multiply.
    """
    steps = list(recipe.steps)
    for k, e in efficiency.items():
        s = steps[k]
        if s.op not in ("transfer", "inject"):
            raise RecipeError(f"step {k + 1} ({s.op}) moves nothing; only transfers and injections have an efficiency")
        if isinstance(e, Mapping):
            own = dict(s.efficiency_by_compound)
            for c, v in e.items():
                nominal = own[c].nominal if c in own else 1.0
                own[c] = NominalActual(nominal=nominal, actual=float(v))
            steps[k] = s.model_copy(update={"efficiency_by_compound": own})
        else:
            nominal = s.efficiency.nominal if s.efficiency is not None else 1.0
            steps[k] = s.model_copy(update={"efficiency": NominalActual(nominal=nominal, actual=float(e))})
    return recipe.model_copy(update={"steps": steps})


def with_actual(recipe: PrepRecipe, actual: Mapping[int, float]) -> PrepRecipe:
    """The recipe with the actual value of some steps changed (step index -> uL, or g for a sample)."""
    steps = list(recipe.steps)
    for k, value in actual.items():
        s = steps[k]
        key = "volume_ul" if s.volume_ul is not None else "mass_g" if s.mass_g is not None else None
        if key is None:
            raise RecipeError(f"step {k + 1} ({s.op}) has no volume or mass")
        steps[k] = s.model_copy(update={key: getattr(s, key).model_copy(update={"actual": float(value)})})
    return recipe.model_copy(update={"steps": steps})
