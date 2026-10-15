"""Load and validate input files.

Every loader validates against its schema and, on failure, raises
`InputValidationError` listing the file, row, and column of each problem.
CSV rows are numbered as they appear in a spreadsheet: the header is row 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import pandas as pd
import yaml
from pydantic import BaseModel, ValidationError

from lcms_sim.io.schemas import (
    AcquisitionMethod,
    Analyte,
    CalibrationLevel,
    CalibrationSettings,
    CheckList,
    ContainerCatalogue,
    InfusionPoint,
    LCMethod,
    MethodInfo,
    PrepProcedure,
    PrepRecipe,
    SampleDefinition,
    ScoutingGradients,
    ScoutingRun,
    SystemConstants,
    Transition,
)

M = TypeVar("M", bound=BaseModel)


@dataclass(frozen=True)
class Problem:
    file: str
    row: int | None
    column: str | None
    message: str

    def __str__(self) -> str:
        where = self.file
        if self.row is not None:
            where += f", row {self.row}"
        if self.column:
            where += f", column '{self.column}'"
        return f"{where}: {self.message}"


@dataclass
class InputValidationError(ValueError):
    problems: list[Problem] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{len(self.problems)} problem(s) in input:\n" + "\n".join(
            f"  - {p}" for p in self.problems
        )


def _loc(err: dict) -> str | None:
    return ".".join(str(x) for x in err["loc"]) or None


# ---------------------------------------------------------------------------
# Generic loaders
# ---------------------------------------------------------------------------


def load_yaml(path: str | Path, model: type[M]) -> M:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        line = getattr(getattr(e, "problem_mark", None), "line", None)
        raise InputValidationError(
            [Problem(path.name, None if line is None else line + 1, None, f"invalid YAML: {e}")]
        ) from e
    try:
        return model.model_validate(raw)
    except ValidationError as e:
        raise InputValidationError(
            [Problem(path.name, None, _loc(err), err["msg"]) for err in e.errors()]
        ) from e


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df.apply(lambda col: col.str.strip())


def _rows_to_models(path: Path, df: pd.DataFrame, model: type[M]) -> list[M]:
    out: list[M] = []
    problems: list[Problem] = []
    for i, rec in enumerate(df.to_dict(orient="records")):
        clean = {k: (None if v == "" else v) for k, v in rec.items()}
        try:
            out.append(model.model_validate(clean))
        except ValidationError as e:
            problems += [Problem(path.name, i + 2, _loc(err), err["msg"]) for err in e.errors()]
    if problems:
        raise InputValidationError(problems)
    return out


def load_csv(path: str | Path, model: type[M]) -> list[M]:
    path = Path(path)
    return _rows_to_models(path, _read_csv(path), model)


# ---------------------------------------------------------------------------
# Typed loaders, one per input file
# ---------------------------------------------------------------------------


def load_system(path: str | Path) -> SystemConstants:
    return load_yaml(path, SystemConstants)


def load_lc_method(path: str | Path) -> LCMethod:
    return load_yaml(path, LCMethod)


def load_acquisition(path: str | Path) -> AcquisitionMethod:
    return load_yaml(path, AcquisitionMethod)


def load_prep(path: str | Path) -> PrepProcedure:
    return load_yaml(path, PrepProcedure)


def load_calibration_settings(path: str | Path) -> CalibrationSettings:
    return load_yaml(path, CalibrationSettings)


def load_prep_recipe(path: str | Path) -> PrepRecipe:
    """Optional file prep_recipe.yaml: the bench steps behind the prep and calibration values."""
    return load_yaml(path, PrepRecipe)


def load_containers(path: str | Path) -> ContainerCatalogue:
    """Optional file containers.yaml: what each recipe vessel physically is."""
    return load_yaml(path, ContainerCatalogue)


def load_method_info(path: str | Path) -> MethodInfo:
    """Optional file method.yaml: the method's name, its sources and where each input comes from."""
    return load_yaml(path, MethodInfo)


def method_info(folder: str | Path) -> MethodInfo:
    """The folder's method.yaml, or a plain description of an input set that has none."""
    folder = Path(folder)
    path = folder / "method.yaml"
    if path.exists():
        return load_method_info(path)
    return MethodInfo(name=folder.name, short=f"the input set in {folder.name}",
                      description="No method.yaml in this folder: sources and provenance are the files' own.")


def load_checks(path: str | Path) -> CheckList:
    """Optional file checks.yaml: the bench checks that separate faults, for the symptom matcher."""
    return load_yaml(path, CheckList)


def load_scouting_gradients(path: str | Path) -> ScoutingGradients:
    return load_yaml(path, ScoutingGradients)


def load_samples(path: str | Path) -> list[SampleDefinition]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("samples"), list):
        raise InputValidationError([Problem(path.name, None, "samples", "expected a list under 'samples'")])
    out, problems = [], []
    for i, item in enumerate(raw["samples"]):
        try:
            out.append(SampleDefinition.model_validate(item))
        except ValidationError as e:
            problems += [Problem(path.name, None, f"samples.{i}.{_loc(err)}", err["msg"]) for err in e.errors()]
    if problems:
        raise InputValidationError(problems)
    return out


def load_analytes(path: str | Path) -> list[Analyte]:
    return load_csv(path, Analyte)


def load_transitions(path: str | Path) -> list[Transition]:
    return load_csv(path, Transition)


def load_scouting_runs(path: str | Path) -> list[ScoutingRun]:
    return load_csv(path, ScoutingRun)


def load_infusion_trace(path: str | Path) -> list[InfusionPoint]:
    return load_csv(path, InfusionPoint)


def load_calibration_design(path: str | Path) -> list[CalibrationLevel]:
    """Wide CSV: a `level` column, then one concentration column (ng/mL) per analyte."""
    path = Path(path)
    df = _read_csv(path)
    if "level" not in df.columns:
        raise InputValidationError([Problem(path.name, 1, "level", "missing required column")])
    analytes = [c for c in df.columns if c != "level"]
    out, problems = [], []
    for i, rec in enumerate(df.to_dict(orient="records")):
        concs: dict[str, float] = {}
        for a in analytes:
            try:
                v = float(rec[a])
                if v < 0:
                    raise ValueError
                concs[a] = v
            except ValueError:
                problems.append(Problem(path.name, i + 2, a, f"expected a number >= 0, got '{rec[a]}'"))
        if rec["level"] == "":
            problems.append(Problem(path.name, i + 2, "level", "level name is empty"))
        out.append(CalibrationLevel(level=rec["level"] or "?", concentrations_ng_ml=concs))
    if problems:
        raise InputValidationError(problems)
    return out


# ---------------------------------------------------------------------------
# A whole input set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputSet:
    """Every input for one method, loaded from a folder with the standard file names."""

    system: SystemConstants
    lc_method: LCMethod
    acquisition: AcquisitionMethod
    analytes: list[Analyte]
    transitions: list[Transition]
    scouting_gradients: ScoutingGradients
    scouting_runs: list[ScoutingRun]
    infusion_matrix: list[InfusionPoint]
    infusion_solvent: list[InfusionPoint]
    prep: PrepProcedure
    calibration: CalibrationSettings
    calibration_design: list[CalibrationLevel]
    samples: list[SampleDefinition]


FILES = {
    "system": ("system.yaml", load_system),
    "lc_method": ("lc_method.yaml", load_lc_method),
    "acquisition": ("acquisition.yaml", load_acquisition),
    "analytes": ("analytes.csv", load_analytes),
    "transitions": ("transitions.csv", load_transitions),
    "scouting_gradients": ("scouting_gradients.yaml", load_scouting_gradients),
    "scouting_runs": ("scouting_runs.csv", load_scouting_runs),
    "infusion_matrix": ("infusion_matrix.csv", load_infusion_trace),
    "infusion_solvent": ("infusion_solvent.csv", load_infusion_trace),
    "prep": ("prep.yaml", load_prep),
    "calibration": ("calibration.yaml", load_calibration_settings),
    "calibration_design": ("calibration_design.csv", load_calibration_design),
    "samples": ("samples.yaml", load_samples),
}


def load_input_set(folder: str | Path) -> InputSet:
    """Load every file, then check references between files. All problems are reported together."""
    folder = Path(folder)
    loaded: dict = {}
    problems: list[Problem] = []
    for key, (name, loader) in FILES.items():
        path = folder / name
        if not path.exists():
            problems.append(Problem(name, None, None, "file not found"))
            continue
        try:
            loaded[key] = loader(path)
        except InputValidationError as e:
            problems += e.problems
    if problems:
        raise InputValidationError(problems)
    inputs = InputSet(**loaded)
    problems = check_references(inputs)
    if problems:
        raise InputValidationError(problems)
    return inputs


def check_references(s: InputSet) -> list[Problem]:
    """Cross-file checks: every name used in one file must be defined in the analyte table."""
    names = {a.name for a in s.analytes}
    istds = {a.name for a in s.analytes if a.is_internal_standard}
    problems: list[Problem] = []

    for a in s.analytes:
        if a.internal_standard and a.internal_standard not in istds:
            problems.append(Problem("analytes.csv", None, "internal_standard",
                                    f"{a.name}: '{a.internal_standard}' is not an internal standard in the table"))

    quant_count: dict[str, int] = {}
    for i, t in enumerate(s.transitions):
        if t.analyte not in names:
            problems.append(Problem("transitions.csv", i + 2, "analyte", f"unknown analyte '{t.analyte}'"))
        if t.role.value == "quantifier":
            quant_count[t.analyte] = quant_count.get(t.analyte, 0) + 1
    for n in names:
        if quant_count.get(n, 0) != 1:
            problems.append(Problem("transitions.csv", None, None,
                                    f"{n}: expected exactly one quantifier, found {quant_count.get(n, 0)}"))

    gids = set(s.scouting_gradients.gradients)
    for i, r in enumerate(s.scouting_runs):
        if r.analyte not in names:
            problems.append(Problem("scouting_runs.csv", i + 2, "analyte", f"unknown analyte '{r.analyte}'"))
        if r.gradient_id not in gids:
            problems.append(Problem("scouting_runs.csv", i + 2, "gradient_id", f"unknown gradient '{r.gradient_id}'"))

    for k in s.prep.recovery:
        if k not in names:
            problems.append(Problem("prep.yaml", None, f"recovery.{k}", "unknown analyte"))

    targets = names - istds
    for lvl in s.calibration_design:
        for k in lvl.concentrations_ng_ml:
            if k not in targets:
                problems.append(Problem("calibration_design.csv", 1, k, "column is not a target analyte"))
        break

    for smp in s.samples:
        for k in smp.true_concentration_ug_kg:
            if k not in targets:
                problems.append(Problem("samples.yaml", None, f"{smp.id}.{k}", "not a target analyte"))

    for name, trace in (("infusion_matrix.csv", s.infusion_matrix), ("infusion_solvent.csv", s.infusion_solvent)):
        times = [p.time_min for p in trace]
        if any(b <= a for a, b in zip(times, times[1:])):
            problems.append(Problem(name, None, "time_min", "times must increase strictly"))
    return problems
