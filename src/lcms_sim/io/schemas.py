"""Typed input schemas for every file the simulator reads.

Units are fixed and appear in field names:
    time            min (chromatography), s or ms (acquisition, as named)
    flow            mL/min
    volume          mL, except injection volume in uL
    mass            g (sample), ng (internal standard amount)
    concentration   ug/kg in the sample, ng/mL in extracts, calibrators and stocks

Every input carries a provenance: measured, fitted, or assumed. The interface
shows it next to each value so the user can see what rests on real data.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Provenance = Literal["measured", "fitted", "assumed"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Nominal / actual pairs
# ---------------------------------------------------------------------------


class NominalActual(_Model):
    """A procedure value: what the SOP says, and what actually happened.

    Signal is generated from `actual`; the reported result is calculated from
    `nominal`. When `actual` is omitted it equals `nominal` (no fault).
    """

    nominal: float
    actual: float | None = None

    @model_validator(mode="after")
    def _default_actual(self) -> NominalActual:
        if self.actual is None:
            object.__setattr__(self, "actual", self.nominal)
        return self

    @property
    def is_faulted(self) -> bool:
        return self.actual != self.nominal


# ---------------------------------------------------------------------------
# LC
# ---------------------------------------------------------------------------


class GradientPoint(_Model):
    time_min: float = Field(ge=0)
    percent_b: float = Field(ge=0, le=100)


class GradientProgram(_Model):
    """Programmed %B at the mixer, linear between points.

    Two points at the same time make a step. Composition before the first point
    equals the first point; after the last point it holds.
    """

    points: list[GradientPoint] = Field(min_length=1)

    @field_validator("points")
    @classmethod
    def _times_non_decreasing(cls, pts: list[GradientPoint]) -> list[GradientPoint]:
        for a, b in zip(pts, pts[1:]):
            if b.time_min < a.time_min:
                raise ValueError(
                    f"gradient times must not decrease ({a.time_min} then {b.time_min})"
                )
        return pts


class SystemConstants(_Model):
    """Column and instrument constants. t0 scales inversely with flow."""

    provenance: Provenance = "assumed"
    column_length_mm: float = Field(gt=0)
    column_id_mm: float = Field(gt=0)
    particle_size_um: float = Field(gt=0)
    t0_min: float = Field(gt=0, description="Void time at reference_flow_ml_min")
    reference_flow_ml_min: float = Field(gt=0)
    dwell_volume_ml: float = Field(ge=0)
    plate_number: float = Field(gt=0)
    extra_column_sigma_s: float = Field(default=0.0, ge=0)

    @property
    def void_volume_ml(self) -> float:
        return self.t0_min * self.reference_flow_ml_min

    def t0_at(self, flow_ml_min: float) -> float:
        return self.void_volume_ml / flow_ml_min


class LCMethod(_Model):
    name: str
    flow_ml_min: float = Field(gt=0)
    gradient: GradientProgram
    run_time_min: float = Field(gt=0)
    ph: float | None = Field(default=None, ge=0, le=14)


class ScoutingGradients(_Model):
    """Gradient programs used for scouting runs, keyed by gradient id."""

    flow_ml_min: float = Field(gt=0)
    gradients: dict[str, GradientProgram] = Field(min_length=1)


class ScoutingRun(_Model):
    """One row of the scouting-runs CSV."""

    analyte: str
    gradient_id: str
    retention_time_min: float = Field(gt=0)
    ph: float | None = Field(default=None, ge=0, le=14, description="Mobile phase pH, for the pH extension")

# ---------------------------------------------------------------------------
# Analytes and acquisition
# ---------------------------------------------------------------------------


class AcidBase(str, Enum):
    neutral = "neutral"
    acid = "acid"
    base = "base"


class Analyte(_Model):
    """One row of the analyte table: retention and response parameters."""

    name: str
    is_internal_standard: bool = False
    internal_standard: str | None = None
    log_kw: float
    S: float = Field(gt=0)
    acid_base: AcidBase = AcidBase.neutral
    pka: float | None = None
    # Optional pH extension: LSS parameters of the fully ionised form. log_kw and S above are
    # then those of the neutral form. Off when absent; the method-pH values are used as-is.
    log_kw_ionised: float | None = None
    S_ionised: float | None = Field(default=None, gt=0)
    response_factor: float = Field(gt=0, description="Quantifier area per ng on column")
    regulated_as: str | None = Field(
        default=None, description="Regulatory name when this is one component of a summed analyte"
    )
    isomer_fraction: float | None = Field(
        default=None, gt=0, le=1,
        description="Share of the regulated total in the calibration standard; levels are entered as the total",
    )
    molecular_weight: float | None = Field(default=None, gt=0)
    provenance: Provenance = "assumed"

    @model_validator(mode="after")
    def _check(self) -> Analyte:
        if self.acid_base is not AcidBase.neutral and self.pka is None:
            raise ValueError(f"{self.name}: pka is required when acid_base is {self.acid_base.value}")
        if (self.log_kw_ionised is None) != (self.S_ionised is None):
            raise ValueError(f"{self.name}: give both log_kw_ionised and S_ionised, or neither")
        if self.log_kw_ionised is not None and self.acid_base is AcidBase.neutral:
            raise ValueError(f"{self.name}: ionised-form parameters need acid_base acid or base")
        if self.is_internal_standard and self.internal_standard:
            raise ValueError(f"{self.name}: an internal standard cannot itself have one")
        return self


class TransitionRole(str, Enum):
    quantifier = "quantifier"
    qualifier = "qualifier"


class Polarity(str, Enum):
    positive = "positive"
    negative = "negative"


class Transition(_Model):
    """One row of the transition table (export of the MRM method)."""

    analyte: str
    precursor_mz: float = Field(gt=0)
    product_mz: float = Field(gt=0)
    role: TransitionRole
    retention_time_min: float = Field(ge=0)
    window_min: float = Field(gt=0)
    polarity: Polarity = Polarity.positive
    adduct: str | None = None
    fragmentor_v: float | None = None
    collision_energy_v: float | None = None
    cell_accel_v: float | None = None
    relative_response: float | None = Field(
        default=None, gt=0, description="Expected area relative to the quantifier (ion ratio); 1 for the quantifier"
    )
    internal_standard: str | None = None


class AcquisitionMethod(_Model):
    """Dynamic MRM acquisition and the detector's counting behaviour.

    Abundance is in area units per minute, so peak area = integral of abundance over time (min).
    `gain_area_per_ion` is the area produced per ion reaching the detector while the transition
    is monitored 100% of the time. With dwell d in a cycle T, the ions actually counted for a
    peak of area A are A / gain x d / T; that number sets the counting precision.
    """

    provenance: Provenance = "assumed"
    cycle_time_ms: float = Field(gt=0)
    inter_scan_delay_ms: float = Field(ge=0)
    polarity_switch_ms: float = Field(default=0.0, ge=0, description="Added per polarity change in a cycle")
    min_points_per_peak: int = Field(default=10, ge=1)
    min_dwell_ms: float = Field(default=1.0, ge=0, description="Dwell below this is flagged")
    gain_area_per_ion: float = Field(gt=0)
    background_rate: float = Field(default=0.0, ge=0, description="Background abundance, area units per min")
    injection_rsd: float = Field(default=0.0, ge=0)
    point_rsd: float = Field(default=0.0, ge=0)
    detector_dead_time_s: float | None = Field(default=None, ge=0)
    q1_fwhm_mz: float = Field(default=0.7, gt=0, description="Q1 pass window, unit resolution")
    q3_fwhm_mz: float = Field(default=0.7, gt=0, description="Q3 pass window, unit resolution")
    crosstalk: bool = Field(default=False, description="Co-eluting compounds leak into transitions they pass")
    crosstalk_yield: float = Field(
        default=1.0, ge=0,
        description="Share of a leaking compound's listed fragment that forms under the receiving transition's "
                    "conditions (1 = as its own relative response); unknown until measured, so assumed")


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------


class InfusionPoint(_Model):
    """One row of a post-column infusion trace CSV."""

    time_min: float = Field(ge=0)
    signal: float = Field(ge=0)


# ---------------------------------------------------------------------------
# Quantitation chain
# ---------------------------------------------------------------------------


class ISTDStage(str, Enum):
    before_extraction = "before_extraction"
    after_extraction = "after_extraction"
    in_vial = "in_vial"


class PrepProcedure(_Model):
    """The extraction SOP, each step as nominal / actual."""

    name: str
    provenance: Provenance = "assumed"
    sample_mass_g: NominalActual
    istd_spike_ul: NominalActual | None = Field(default=None, description="Volume of IS mix added")
    istd_mix_ng_ul: dict[str, float] = Field(
        default_factory=dict, description="Concentration of each IS in the mix, ng/uL (= ug/mL)"
    )
    istd_mix_factor_actual: dict[str, float] = Field(
        default_factory=dict, description="Per-IS actual / nominal content of the mix (mix made wrong); missing = 1"
    )
    istd_stage: ISTDStage | None = None
    extraction_volume_ml: NominalActual
    dilution_factor: NominalActual
    injection_volume_ul: NominalActual
    vial_volume_ml: NominalActual | None = Field(
        default=None, description="Final vial volume; needed only when the internal standard is added in the vial"
    )
    recovery: dict[str, float] = Field(
        default_factory=dict, description="Per-analyte extraction recovery; missing analytes use 1"
    )

    @field_validator("recovery")
    @classmethod
    def _recovery_range(cls, rec: dict[str, float]) -> dict[str, float]:
        for k, v in rec.items():
            if not 0 < v <= 1.5:
                raise ValueError(f"recovery for {k} must be in (0, 1.5], got {v}")
        return rec

    @model_validator(mode="after")
    def _istd_pair(self) -> PrepProcedure:
        if (self.istd_spike_ul is None) != (self.istd_stage is None):
            raise ValueError("give both istd_spike_ul and istd_stage, or neither")
        if self.istd_spike_ul is not None and not self.istd_mix_ng_ul:
            raise ValueError("istd_mix_ng_ul is required when an internal standard is added")
        if self.istd_stage is ISTDStage.in_vial and self.vial_volume_ml is None:
            raise ValueError("vial_volume_ml is required when istd_stage is in_vial")
        return self


class SampleDefinition(_Model):
    id: str
    matrix: str
    true_concentration_ug_kg: dict[str, float] = Field(default_factory=dict)
    add_istd: bool = True
    kind: Literal["blank", "qc", "sample"] | None = None
    spiked_from_calibration_stock: bool = Field(
        default=False,
        description="QC spiked from the same stock as the calibrators: a stock error moves it with the curve",
    )

    @field_validator("true_concentration_ug_kg")
    @classmethod
    def _non_negative(cls, c: dict[str, float]) -> dict[str, float]:
        bad = {k: v for k, v in c.items() if v < 0}
        if bad:
            raise ValueError(f"concentrations must be >= 0: {bad}")
        return c


class CalibrationModel(str, Enum):
    linear = "linear"
    quadratic = "quadratic"


class Weighting(str, Enum):
    none = "none"
    inv_x = "1/x"
    inv_x2 = "1/x2"


class OriginHandling(str, Enum):
    ignore = "ignore"
    include = "include"
    force = "force"


class CalibrationSettings(_Model):
    """Calibration fit settings and how calibrators are made."""

    matrix_matched: bool = False
    use_internal_standard: bool = True
    model: CalibrationModel = CalibrationModel.linear
    weighting: Weighting = Weighting.inv_x
    origin: OriginHandling = OriginHandling.ignore
    preparation: Literal["vial", "procedural"] = Field(
        default="vial",
        description="vial: calibrators are solutions in the vial; procedural: spiked into blank matrix and "
                    "taken through extraction like samples. Levels are always the nominal vial concentration.",
    )
    istd_spike_ul: NominalActual | None = Field(
        default=None, description="IS mix volume added to each calibrator (same mix as samples)"
    )
    stock_concentration_ng_ml: NominalActual
    stock_factor_actual: dict[str, float] = Field(
        default_factory=dict,
        description="Per-analyte actual / labelled content of the stock (degraded or mislabelled analytes); missing = 1",
    )

    @model_validator(mode="after")
    def _istd_given(self) -> CalibrationSettings:
        if self.use_internal_standard and self.istd_spike_ul is None:
            raise ValueError("istd_spike_ul is required when use_internal_standard is true")
        return self


class RecipeStep(_Model):
    """One bench step in the preparation recipe. Volumes in uL, masses in g, stock contents in ng/uL.

    stock:    a new vessel of `volume_ul` holding `contents_ng_ul` of each compound.
    solvent:  add `volume_ul` of clean solvent to `vessel`.
    transfer: move `volume_ul` from `source` to `vessel`, carrying that share of every compound.
    sample:   weigh `mass_g` of matrix into `vessel`, with native `sample_ug_kg` (blank: none).
    extract:  shake and filter `vessel`; the actual run applies `recovery` (nominal assumes complete).
    inject:   draw `volume_ul` from `source` onto the column (`vessel` names the injection).

    `efficiency` (transfer and inject): the share of the nominal amount the drawn volume really carries, 1 when
    the source is homogeneous and nothing is lost on the way. The actual run multiplies the moved amount by it;
    what did not move stays in the source. It is the same for every compound: a pipette or delivery effect.
    `efficiency_by_compound` multiplies on top of it for the compounds named (an unmixed component, a precipitated
    or adsorbed compound); compounds not named carry `efficiency` alone.
    """

    op: Literal["stock", "solvent", "transfer", "sample", "extract", "inject"]
    vessel: str
    source: str | None = None
    volume_ul: NominalActual | None = None
    mass_g: NominalActual | None = None
    efficiency: NominalActual | None = None
    efficiency_by_compound: dict[str, NominalActual] = Field(default_factory=dict)
    contents_ng_ul: dict[str, float] = Field(default_factory=dict)
    sample_ug_kg: dict[str, float] = Field(default_factory=dict)
    recovery: dict[str, float] = Field(default_factory=dict)
    note: str = ""

    @model_validator(mode="after")
    def _check(self) -> RecipeStep:
        need = {"stock": ("volume_ul",), "solvent": ("volume_ul",), "transfer": ("volume_ul", "source"),
                "sample": ("mass_g",), "extract": (), "inject": ("volume_ul", "source")}[self.op]
        missing = [f for f in need if getattr(self, f) is None]
        if missing:
            raise ValueError(f"{self.op} step for {self.vessel!r} needs {', '.join(missing)}")
        return self


class PrepRecipe(_Model):
    """The preparation written out step by step, from reference standards to the injection."""

    name: str
    provenance: Provenance = "assumed"
    steps: list[RecipeStep]


class ContainerSpec(_Model):
    """What a recipe vessel physically is: the thing drawn on the Follow page.

    `vessel` matches a vessel name exactly, `prefix` every vessel starting with it, `contains` every vessel with
    it somewhere in the name. `kind` picks the cartoon.
    """

    vessel: str | None = None
    prefix: str | None = None
    contains: str | None = None
    kind: Literal["ampoule", "vial_2ml", "vial_20ml", "tube", "autosampler_vial", "loop"]
    location: str = ""                   # freezer, fridge, bench, autosampler
    solvent: str = ""
    capacity_ml: float | None = Field(default=None, gt=0)
    supplier: str = ""
    note: str = ""
    provenance: Provenance = "assumed"

    @model_validator(mode="after")
    def _one_match(self) -> ContainerSpec:
        if sum(x is not None for x in (self.vessel, self.prefix, self.contains)) != 1:
            raise ValueError("a container needs exactly one of vessel, prefix or contains")
        return self


class SeparatingCheck(_Model):
    """A bench check whose outcome differs between faults (checks.yaml). A fault not listed under `outcomes`
    is expected to leave the check unchanged."""

    name: str
    do: str
    outcomes: dict[str, str] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list, description="Keys into data/reference/references.yaml")

    def outcome(self, fault: str) -> str:
        return self.outcomes.get(fault, "unchanged")

    def separates(self, a: str, b: str) -> bool:
        return self.outcome(a) != self.outcome(b)


class CheckList(_Model):
    checks: list[SeparatingCheck]


class ContainerCatalogue(_Model):
    containers: list[ContainerSpec]

    def for_vessel(self, name: str) -> ContainerSpec | None:
        """Exact match first, then the longest matching prefix."""
        exact = next((c for c in self.containers if c.vessel == name), None)
        if exact is not None:
            return exact
        hits = [c for c in self.containers if c.prefix is not None and name.startswith(c.prefix)]
        if hits:
            return max(hits, key=lambda c: len(c.prefix))
        return next((c for c in self.containers if c.contains is not None and c.contains in name), None)


class MethodRow(_Model):
    """One line of "where these inputs come from"."""

    input: str
    value: str
    provenance: Provenance


class MethodInfo(_Model):
    """What an input set is and where it came from (method.yaml). The pages name the method and cite its
    sources from here, so no page text is tied to one method."""

    name: str
    short: str = Field(description="How pages refer to the method in a sentence")
    description: str = ""
    sources: list[str] = Field(default_factory=list, description="Keys into data/reference/references.yaml")
    provenance: list[MethodRow] = Field(default_factory=list)
    remarks: dict[str, str] = Field(
        default_factory=dict, description="Method-specific notes keyed by page (for instance bench_steps)")


class CalibrationLevel(_Model):
    """One row of the calibration design (wide CSV: level + one column per analyte)."""

    level: str
    concentrations_ng_ml: dict[str, float]
