"""Phase 0: the example data loads and validates, and bad input is reported by row and column."""

from pathlib import Path

import pytest

from lcms_sim.io.loaders import (
    InputValidationError,
    load_analytes,
    load_calibration_design,
    load_input_set,
    load_prep,
    load_transitions,
)
from lcms_sim.io.schemas import GradientProgram, NominalActual
from lcms_sim.paths import EXAMPLES, REFERENCE


@pytest.fixture(scope="module")
def inputs():
    return load_input_set(EXAMPLES)


def test_example_set_loads(inputs):
    assert len(inputs.analytes) == 73
    assert sum(a.is_internal_standard for a in inputs.analytes) == 9
    assert len(inputs.transitions) == 151
    assert len(inputs.calibration_design) == 8
    assert {s.id for s in inputs.samples} >= {"Matrix blank", "LCS", "ICV", "CCV 1", "LOQ"}


def test_example_set_covers_the_mn_list_but_its_listed_gaps(inputs):
    import pandas as pd

    mn = set(pd.read_csv(REFERENCE / "mn_ocm_cta_v2_analytes.csv").analyte)
    covered = {a.regulated_as for a in inputs.analytes if not a.is_internal_standard}
    gaps = {line[2:].strip() for line in (EXAMPLES / "GAPS.md").read_text(encoding="utf-8").splitlines()
            if line.startswith("- ")}
    assert covered | gaps == mn and not covered & gaps
    assert gaps == {"Azadirachtin", "Chlormequat chloride"}


def test_every_target_has_the_nearest_internal_standard_of_its_polarity(inputs):
    """The example's IS pairing rule (assumed; the application note uses none)."""
    quant = {t.analyte: t for t in inputs.transitions if t.role.value == "quantifier"}
    istds = [a.name for a in inputs.analytes if a.is_internal_standard]
    for a in inputs.analytes:
        if a.is_internal_standard:
            continue
        pol = quant[a.name].polarity
        pool = [i for i in istds if quant[i].polarity == pol] or istds
        rt = quant[a.name].retention_time_min
        best = min(abs(quant[i].retention_time_min - rt) for i in pool)
        assert abs(quant[a.internal_standard].retention_time_min - rt) == pytest.approx(best), a.name


def test_isomer_fractions_sum_to_one(inputs):
    totals = {}
    for a in inputs.analytes:
        if a.isomer_fraction:
            totals[a.regulated_as] = totals.get(a.regulated_as, 0) + a.isomer_fraction
    assert totals == pytest.approx({"Pyrethrins (sum of 6)": 1.0, "Spinosad": 1.0, "Abamectin": 1.0})


def test_model_retention_reproduces_measured(inputs):
    """The LC model places every peak at the retention time its transition table lists (the application note's,
    measured on the note's column and gradient; labelled standards 0.01 min earlier)."""
    from lcms_sim.lc.gradient import LCConditions
    from lcms_sim.lc.retention import retention_time

    target = {t.analyte: t.retention_time_min for t in inputs.transitions}
    cond = LCConditions.from_method(inputs.system, inputs.lc_method)
    for a in inputs.analytes:
        assert retention_time(a.log_kw, a.S, cond) == pytest.approx(target[a.name], rel=1e-4), a.name


def test_quantifier_relative_response_is_one(inputs):
    for t in inputs.transitions:
        if t.role.value == "quantifier":
            assert t.relative_response == 1.0


def test_retention_times_inside_run(inputs):
    run = inputs.lc_method.run_time_min
    for t in inputs.transitions:
        assert 0.95 * inputs.system.t0_at(inputs.lc_method.flow_ml_min) < t.retention_time_min < run


def test_every_analyte_has_scouting_runs_on_each_gradient(inputs):
    gids = set(inputs.scouting_gradients.gradients)
    seen: dict[str, set] = {}
    for r in inputs.scouting_runs:
        seen.setdefault(r.analyte, set()).add(r.gradient_id)
    assert all(seen.get(a.name) == gids for a in inputs.analytes)


def test_matrix_profile_shows_suppression_zones(inputs):
    solvent = {p.time_min: p.signal for p in inputs.infusion_solvent}
    mf = {p.time_min: p.signal / solvent[p.time_min] for p in inputs.infusion_matrix}
    t0 = inputs.system.t0_at(inputs.lc_method.flow_ml_min)
    near_void = min(mf, key=lambda t: abs(t - 1.15 * t0))
    assert mf[near_void] < 0.5   # near the void
    assert 0.9 < mf[5.0] < 1.1   # mid gradient, little effect
    assert mf[8.3] < 0.7         # late zone, where cannabinoids elute near 96-100% methanol


def test_nominal_actual_defaults_to_nominal():
    v = NominalActual(nominal=10.0)
    assert v.actual == 10.0 and not v.is_faulted
    assert NominalActual(nominal=10.0, actual=9.0).is_faulted


def test_gradient_rejects_decreasing_time():
    with pytest.raises(ValueError):
        GradientProgram.model_validate({"points": [{"time_min": 2, "percent_b": 5}, {"time_min": 1, "percent_b": 50}]})


def test_gradient_allows_steps():
    GradientProgram.model_validate({"points": [{"time_min": 1, "percent_b": 95}, {"time_min": 1, "percent_b": 5}]})


# --- error reporting -------------------------------------------------------


def _copy_with(tmp_path: Path, name: str, edit) -> Path:
    text = (EXAMPLES / name).read_text(encoding="utf-8")
    p = tmp_path / name
    p.write_text(edit(text), encoding="utf-8")
    return p


def test_csv_error_names_row_and_column(tmp_path):
    def break_row_3(text):
        lines = text.splitlines()
        cells = lines[2].split(",")
        cells[2] = "-5"  # product_mz
        lines[2] = ",".join(cells)
        return "\n".join(lines) + "\n"

    p = _copy_with(tmp_path, "transitions.csv", break_row_3)
    with pytest.raises(InputValidationError) as e:
        load_transitions(p)
    (prob,) = e.value.problems
    assert prob.row == 3 and prob.column == "product_mz"
    assert "row 3" in str(e.value) and "product_mz" in str(e.value)


def test_csv_error_unknown_role(tmp_path):
    p = _copy_with(tmp_path, "transitions.csv", lambda t: t.replace("qualifier", "qualifer", 1))
    with pytest.raises(InputValidationError) as e:
        load_transitions(p)
    assert e.value.problems[0].column == "role"


def test_analyte_needs_pka_when_ionisable(tmp_path):
    def drop_pka(text):
        lines = text.splitlines()
        hdr = lines[0].split(",")
        i = hdr.index("pka")
        for n, line in enumerate(lines[1:], start=1):
            cells = line.split(",")
            if cells[hdr.index("acid_base")] == "base":
                cells[i] = ""
                lines[n] = ",".join(cells)
                return "\n".join(lines) + "\n", n + 1
        raise AssertionError

    text, row = drop_pka((EXAMPLES / "analytes.csv").read_text(encoding="utf-8"))
    p = tmp_path / "analytes.csv"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(InputValidationError) as e:
        load_analytes(p)
    assert e.value.problems[0].row == row


def test_calibration_design_bad_cell(tmp_path):
    def bad(text):
        lines = text.splitlines()
        cells = lines[3].split(",")
        cells[1] = "abc"
        lines[3] = ",".join(cells)
        return "\n".join(lines) + "\n"

    p = _copy_with(tmp_path, "calibration_design.csv", bad)
    with pytest.raises(InputValidationError) as e:
        load_calibration_design(p)
    hdr = (EXAMPLES / "calibration_design.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert (e.value.problems[0].row, e.value.problems[0].column) == (4, hdr[1])


def test_yaml_error_names_field(tmp_path):
    p = _copy_with(tmp_path, "prep.yaml",
                   lambda t: t.replace("injection_volume_ul:\n  nominal: 10.0", "injection_volume_ul:\n  nominal: ten"))
    with pytest.raises(InputValidationError) as e:
        load_prep(p)
    assert e.value.problems[0].column == "injection_volume_ul.nominal"


def test_cross_file_reference_checked(tmp_path):
    for f in EXAMPLES.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    p = tmp_path / "scouting_runs.csv"
    p.write_text(p.read_text(encoding="utf-8").replace(",g2x,", ",g99,", 1), encoding="utf-8")
    with pytest.raises(InputValidationError) as e:
        load_input_set(tmp_path)
    assert any(pr.column == "gradient_id" and pr.file == "scouting_runs.csv" for pr in e.value.problems)


def test_missing_file_reported(tmp_path):
    with pytest.raises(InputValidationError) as e:
        load_input_set(tmp_path)
    assert len(e.value.problems) == 13


def _analyte(**kw):
    from lcms_sim.io.schemas import Analyte

    base = dict(name="X", log_kw=3.0, S=4.0, response_factor=1e6)
    return Analyte.model_validate({**base, **kw})


def test_ionised_form_fields_optional_and_paired():
    assert _analyte().log_kw_ionised is None
    a = _analyte(acid_base="acid", pka=4.4, log_kw_ionised=1.5, S_ionised=3.5)
    assert a.S_ionised == 3.5
    with pytest.raises(ValueError):
        _analyte(acid_base="acid", pka=4.4, log_kw_ionised=1.5)
    with pytest.raises(ValueError):
        _analyte(log_kw_ionised=1.5, S_ionised=3.5)
