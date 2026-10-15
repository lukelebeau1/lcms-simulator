"""Sample preparation: from concentration in the sample to amount on column, nominal and actual.

Order of steps, and where the internal standard (IS) can join:

    weigh (m) -> [IS before extraction] -> extract (recovery R) -> [IS after extraction]
      -> make to volume (V) -> dilute (D) -> [IS in vial, vial volume Vv] -> inject (Vinj)

Steps after the IS joins scale analyte and IS alike and cancel in the area ratio. Signal is
generated from the actual values; the reported result is back-calculated from the nominal ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lcms_sim.io.schemas import ISTDStage, PrepProcedure

Which = Literal["nominal", "actual"]


@dataclass(frozen=True)
class PrepValues:
    mass_g: float
    extraction_ml: float
    dilution: float
    injection_ul: float
    istd_spike_ul: float | None
    istd_stage: ISTDStage | None
    vial_ml: float | None


def prep_values(prep: PrepProcedure, which: Which) -> PrepValues:
    g = (lambda na: None if na is None else getattr(na, which))
    return PrepValues(
        mass_g=g(prep.sample_mass_g), extraction_ml=g(prep.extraction_volume_ml),
        dilution=g(prep.dilution_factor), injection_ul=g(prep.injection_volume_ul),
        istd_spike_ul=g(prep.istd_spike_ul), istd_stage=prep.istd_stage, vial_ml=g(prep.vial_volume_ml),
    )


def analyte_vial_conc(c_ug_kg: float, recovery: float, v: PrepValues) -> float:
    """ng/mL in the vial. ug/kg = ng/g, so ng/g x g / mL."""
    return c_ug_kg * v.mass_g * recovery / (v.extraction_ml * v.dilution)


def istd_vial_conc(v: PrepValues, mix_ng_ul: float, recovery_istd: float = 1.0) -> float | None:
    """ng/mL of one internal standard in the vial, or None when the method has no IS.

    Amount added = spike volume (uL) x that IS's concentration in the mix (ng/uL).
    """
    if v.istd_spike_ul is None:
        return None
    amount = v.istd_spike_ul * mix_ng_ul
    if v.istd_stage is ISTDStage.before_extraction:
        return amount * recovery_istd / (v.extraction_ml * v.dilution)
    if v.istd_stage is ISTDStage.after_extraction:
        return amount / (v.extraction_ml * v.dilution)
    return amount / v.vial_ml


def on_column_ng(vial_conc_ng_ml: float, v: PrepValues) -> float:
    return vial_conc_ng_ml * v.injection_ul / 1000.0


def back_calculate(vial_conc_ng_ml: float, nominal: PrepValues) -> float:
    """Reported ug/kg in the sample from a vial concentration, using the procedure's nominal values."""
    return vial_conc_ng_ml * nominal.extraction_ml * nominal.dilution / nominal.mass_g


@dataclass(frozen=True)
class ChainStage:
    stage: str
    quantity: str
    unit: str
    nominal: float
    actual: float


def walk(c_ug_kg: float, prep: PrepProcedure, recovery: float = 1.0, matrix_factor: float = 1.0) -> list[ChainStage]:
    """One analyte through every stage, nominal beside actual (for the chain walk-through view)."""
    n, a = prep_values(prep, "nominal"), prep_values(prep, "actual")
    stages = [
        ChainStage("Sample", "analyte in sample", "ng", c_ug_kg * n.mass_g, c_ug_kg * a.mass_g),
        ChainStage("Extraction", "analyte in extract", "ng", c_ug_kg * n.mass_g, c_ug_kg * a.mass_g * recovery),
        ChainStage("Make to volume", "extract concentration", "ng/mL",
                   c_ug_kg * n.mass_g / n.extraction_ml, c_ug_kg * a.mass_g * recovery / a.extraction_ml),
        ChainStage("Dilution", "vial concentration", "ng/mL",
                   analyte_vial_conc(c_ug_kg, 1.0, n), analyte_vial_conc(c_ug_kg, recovery, a)),
        ChainStage("Injection", "on column", "ng",
                   on_column_ng(analyte_vial_conc(c_ug_kg, 1.0, n), n),
                   on_column_ng(analyte_vial_conc(c_ug_kg, recovery, a), a)),
        ChainStage("Ionisation", "effective on column (x matrix factor)", "ng",
                   on_column_ng(analyte_vial_conc(c_ug_kg, 1.0, n), n),
                   on_column_ng(analyte_vial_conc(c_ug_kg, recovery, a), a) * matrix_factor),
    ]
    if n.istd_spike_ul is not None:
        stages.insert(1, ChainStage(f"IS addition ({n.istd_stage.value})", "IS mix added", "uL",
                                    n.istd_spike_ul, a.istd_spike_ul))
    return stages
