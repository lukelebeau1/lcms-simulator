"""Preparation ledger: amounts through every bench step, against the closed-form prep values."""

from pathlib import Path

import pytest

from lcms_sim.io.loaders import load_input_set, load_prep_recipe
from lcms_sim.io.schemas import PrepRecipe, RecipeStep
from lcms_sim.quant.ledger import RecipeError, lineage, molecules, run_recipe, with_actual
from lcms_sim.quant.prep import analyte_vial_conc, istd_vial_conc, prep_values

EXAMPLES = Path(__file__).resolve().parents[1] / "data" / "examples"


@pytest.fixture(scope="module")
def inputs():
    return load_input_set(EXAMPLES)


@pytest.fixture(scope="module")
def recipe():
    return load_prep_recipe(EXAMPLES / "prep_recipe.yaml")


@pytest.fixture(scope="module")
def nominal(recipe):
    return run_recipe(recipe, "nominal")[0]


def steps(*rows) -> PrepRecipe:
    return PrepRecipe(name="t", steps=[RecipeStep.model_validate(r) for r in rows])


def test_one_to_ten_dilution_by_amounts():
    r = steps({"op": "stock", "vessel": "std", "volume_ul": {"nominal": 1000}, "contents_ng_ul": {"A": 10}},
              {"op": "transfer", "vessel": "dil", "source": "std", "volume_ul": {"nominal": 100}},
              {"op": "solvent", "vessel": "dil", "volume_ul": {"nominal": 900}})
    v, _ = run_recipe(r)
    assert v["std"].ng["A"] == pytest.approx(9000)          # 10 000 ng less the 1000 ng taken
    assert v["dil"].ng["A"] == pytest.approx(1000)
    assert v["dil"].conc_ng_ml("A") == pytest.approx(1000)   # 1 ng/uL = 1 ug/mL


def test_calibrator_vials_match_the_design_levels(inputs, nominal):
    """The serial calibration stocks, 50 uL each into the 500 uL vial, land on the calibration design's levels."""
    frac = {a.name: a.isomer_fraction or 1.0 for a in inputs.analytes}
    for lv in inputs.calibration_design:
        vial = nominal[f"Vial: {lv.level}"]
        for c, level in lv.concentrations_ng_ml.items():
            assert vial.conc_ng_ml(c) / frac[c] == pytest.approx(level, rel=1e-3), (lv.level, c)


def test_qc_and_sample_vials_match_closed_form(inputs, nominal):
    nom = prep_values(inputs.prep, "nominal")
    for s in inputs.samples:
        vial = nominal.get(f"Vial: {s.id}")
        if vial is None:
            continue
        for c, ug_kg in s.true_concentration_ug_kg.items():
            assert vial.conc_ng_ml(c) == pytest.approx(analyte_vial_conc(ug_kg, 1.0, nom), rel=1e-3), (s.id, c)
    assert nominal.get("Vial: PT sample") is not None


def test_internal_standard_in_vial_matches_closed_form(inputs, nominal):
    nom = prep_values(inputs.prep, "nominal")
    for n, mix in inputs.prep.istd_mix_ng_ul.items():
        assert nominal["Vial: Cal 1"].conc_ng_ml(n) == pytest.approx(istd_vial_conc(nom, mix), rel=1e-9), n


def test_amounts_are_conserved(recipe, nominal):
    created: dict[str, float] = {}
    for s in recipe.steps:
        if s.op == "stock":
            for c, x in s.contents_ng_ul.items():
                created[c] = created.get(c, 0.0) + x * s.volume_ul.nominal
        if s.op == "sample":
            for c, x in s.sample_ug_kg.items():
                created[c] = created.get(c, 0.0) + x * s.mass_g.nominal
    for c, total in created.items():
        assert sum(v.ng.get(c, 0.0) for v in nominal.values()) == pytest.approx(total, rel=1e-12), c


def test_pipetting_error_reaches_only_downstream_vessels(recipe, nominal):
    k = next(i for i, s in enumerate(recipe.steps) if s.op == "transfer" and s.vessel == "Cal 5 stock")
    actual, _ = run_recipe(with_actual(recipe, {k: 360.0}), "actual")   # 10% short from Cal 6 stock
    c = "Myclobutanil"
    ratio = {n: actual[f"Vial: Cal {n}"].conc_ng_ml(c) / nominal[f"Vial: Cal {n}"].conc_ng_ml(c) for n in range(1, 9)}
    for n in (6, 7, 8):
        assert ratio[n] == pytest.approx(1.0)
    low = [ratio[n] for n in range(1, 6)]
    assert all(r == pytest.approx(low[0]) for r in low) and low[0] < 0.97   # serial: the same error carries down
    assert actual["Vial: CCV 1"].conc_ng_ml(c) == pytest.approx(nominal["Vial: CCV 1"].conc_ng_ml(c))


def test_recovery_applies_to_the_actual_run_only():
    r = steps({"op": "sample", "vessel": "t", "mass_g": {"nominal": 0.2}, "sample_ug_kg": {"A": 100}},
              {"op": "solvent", "vessel": "t", "volume_ul": {"nominal": 5000}},
              {"op": "extract", "vessel": "t", "recovery": {"A": 0.8}})
    assert run_recipe(r, "nominal")[0]["t"].ng["A"] == pytest.approx(20.0)
    assert run_recipe(r, "actual")[0]["t"].ng["A"] == pytest.approx(16.0)


def test_cannot_take_more_than_the_vessel_holds():
    r = steps({"op": "stock", "vessel": "s", "volume_ul": {"nominal": 100}, "contents_ng_ul": {"A": 1}},
              {"op": "transfer", "vessel": "d", "source": "s", "volume_ul": {"nominal": 150}})
    with pytest.raises(RecipeError):
        run_recipe(r)


def test_lineage_follows_sources(recipe):
    names = {recipe.steps[k].vessel for k in lineage(recipe, "Vial: Cal 1")}
    assert {"Cal 8 stock", "Cal 1 stock", "Tube: Blank flower for calibrators", "IS master mix",
            "Pesticide stock 10 ppm"} <= names
    assert "CCV 1 stock" not in names and "Vial: Cal 2" not in names


def test_molecules_on_column():
    # 12.825 pg of myclobutanil (288.8 g/mol): Cal 1, 10 uL injected.
    assert molecules(0.012825, 288.8) == pytest.approx(2.674e10, rel=1e-3)


def test_transfer_efficiency_moves_less_and_leaves_the_rest_in_the_source():
    from lcms_sim.quant.ledger import with_efficiency

    r = steps({"op": "stock", "vessel": "std", "volume_ul": {"nominal": 1000}, "contents_ng_ul": {"A": 10}},
              {"op": "transfer", "vessel": "dil", "source": "std", "volume_ul": {"nominal": 100}},
              {"op": "solvent", "vessel": "dil", "volume_ul": {"nominal": 900}})
    half = with_efficiency(r, {1: 0.5})                     # the tip found half the nominal concentration
    v, _ = run_recipe(half, "actual")
    assert v["dil"].ng["A"] == pytest.approx(500) and v["std"].ng["A"] == pytest.approx(9500)
    assert v["dil"].volume_ul == pytest.approx(1000) and v["std"].volume_ul == pytest.approx(900)   # volumes as before
    assert run_recipe(half, "nominal")[0]["dil"].ng["A"] == pytest.approx(1000)                     # nominal untouched
    with pytest.raises(RecipeError):
        with_efficiency(r, {2: 0.5})                         # a solvent step moves no compound


def test_a_compound_efficiency_leaves_the_other_compounds_alone():
    from lcms_sim.quant.ledger import with_efficiency

    r = steps({"op": "stock", "vessel": "std", "volume_ul": {"nominal": 1000}, "contents_ng_ul": {"A": 10, "B": 10}},
              {"op": "transfer", "vessel": "dil", "source": "std", "volume_ul": {"nominal": 100}},
              {"op": "solvent", "vessel": "dil", "volume_ul": {"nominal": 900}})
    own = with_efficiency(r, {1: {"A": 0.5}})               # A precipitated in the cold stock; B in solution
    v, _ = run_recipe(own, "actual")
    assert v["dil"].ng == pytest.approx({"A": 500, "B": 1000})
    assert v["std"].ng == pytest.approx({"A": 9500, "B": 9000})                      # A's crystals stay in the vial
    assert run_recipe(own, "nominal")[0]["dil"].ng == pytest.approx({"A": 1000, "B": 1000})
    both = with_efficiency(own, {1: 0.8})                   # and the pipette under-delivers for everything
    v, _ = run_recipe(both, "actual")
    assert v["dil"].ng == pytest.approx({"A": 400, "B": 800})
    assert v["std"].ng == pytest.approx({"A": 9600, "B": 9200})
