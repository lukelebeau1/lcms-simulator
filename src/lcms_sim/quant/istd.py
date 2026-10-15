"""Internal standard assignment and the calibration response."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from lcms_sim.io.schemas import Analyte


def istd_of(analyte: str, analytes: Sequence[Analyte]) -> str | None:
    for a in analytes:
        if a.name == analyte:
            return a.internal_standard or None
    raise KeyError(analyte)


def response(areas: Mapping[str, float], analyte: str, istd: str | None) -> float:
    """Area, or area ratio to the assigned internal standard."""
    if istd is None:
        return areas[analyte]
    a_is = areas.get(istd, 0.0)
    return areas[analyte] / a_is if a_is > 0 else float("nan")
