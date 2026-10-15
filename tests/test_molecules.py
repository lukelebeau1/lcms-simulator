"""3D molecules and ions of the method: structures, hydrophobicity shares, functional groups, precursor ions."""

import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import molecules as M  # noqa: E402

from lcms_sim.io.loaders import load_input_set  # noqa: E402
from lcms_sim.paths import EXAMPLES  # noqa: E402

INPUTS = load_input_set(EXAMPLES)
QUANT = [t for t in INPUTS.transitions if t.role.value == "quantifier"]


def test_every_analyte_has_a_3d_structure():
    names = {a.name for a in INPUTS.analytes}
    assert names <= set(M.table().index)
    for n in names:
        m = M.load(n)
        assert m.GetNumConformers() == 1 and m.GetConformer().Is3D()
        assert any(a.GetAtomicNum() == 1 for a in m.GetAtoms())          # explicit hydrogens


def test_atom_shares_add_up_to_crippen_logp():
    for n in ("Chlorpyrifos", "Carbaryl", "Avermectin B1a"):
        m = M.load(n)
        assert M.contributions(m).sum() == pytest.approx(Crippen.MolLogP(m), abs=1e-9)
        assert sum(g.logp for g in M.groups(m)) == pytest.approx(Crippen.MolLogP(m), abs=1e-9)


def test_every_atom_in_exactly_one_group():
    for n in ("Myclobutanil", "Azoxystrobin", "Spinosyn A"):
        m = M.load(n)
        seen = [i for g in M.groups(m) for i in g.atoms]
        assert sorted(seen) == list(range(m.GetNumAtoms()))


@pytest.mark.parametrize("name, expected", [
    ("Chlorpyrifos", {"phosphate / thiophosphate ester", "N-heteroaromatic ring (pyridine, pyrimidine...)",
                      "halogen on a ring"}),
    ("Carbaryl", {"carbamate", "aromatic ring (carbon)"}),
    ("Myclobutanil", {"triazole ring", "nitrile"}),
    ("Chlormequat chloride", {"quaternary ammonium (permanent +)"}),
])
def test_functional_groups_found(name, expected):
    assert expected <= {g.name for g in M.groups(M.load(name))}


def test_polar_groups_pull_and_skeleton_holds():
    g = {x.name: x.logp for x in M.groups(M.load("Carbaryl"))}
    assert g["carbamate"] < 0 < g["aromatic ring (carbon)"]


@pytest.mark.parametrize("t", QUANT, ids=lambda t: t.analyte)
def test_every_quantifier_precursor_is_identified_and_the_ion_has_its_mass(t):
    kind, nd, iso = M.precursor_type(t.analyte, t.precursor_mz, t.polarity.value)
    assert kind != "unknown"
    m = M.load(t.analyte)
    ion, added = M.make_ion(m, kind)
    extra = {"[M+H]+": 1, "[M+NH4]+": 5, "[M+Na]+": 1, "[M]+": 0, "[M-H]-": 0, "[M+H-HF]+": 1}[kind]
    assert len(added) == (1 if kind == "[M-H]-" else extra)
    mass = Descriptors.ExactMolWt(m) + nd * M.D_H + iso * M.HEAVY_ISO + \
        {"[M+H]+": M.PROTON, "[M+NH4]+": M.NH4, "[M+Na]+": M.NA, "[M]+": 0.0, "[M-H]-": -M.PROTON,
         "[M+H-HF]+": M.PROTON - M.HF}[kind]
    assert mass == pytest.approx(t.precursor_mz, abs=0.35)


def test_heavy_isotope_peaks_and_negative_mode():
    assert M.precursor_type("Chlorantraniliprole", 483.9)[::2] == ("[M+H]+", 1)
    assert M.precursor_type("Cypermethrin", 435.3)[::2] == ("[M+NH4]+", 1)
    assert M.precursor_type("Fludioxonil", 247.1, "negative")[0] == "[M-H]-"
    assert M.precursor_type("Fludioxonil", 229.0)[0] == "[M+H-HF]+"      # the application note's positive-mode ion


def test_proton_goes_on_the_triazole_for_myclobutanil():
    m = M.load("Myclobutanil")
    site = M._site(m)
    assert m.GetAtomWithIdx(site).GetSymbol() == "N" and m.GetAtomWithIdx(site).GetIsAromatic()
    ion, added = M.make_ion(m, "[M+H]+")
    d = np.linalg.norm(ion.GetConformer().GetPositions()[added[0]] - ion.GetConformer().GetPositions()[site])
    assert d == pytest.approx(1.02, abs=0.01)


def test_viewer_html_is_self_contained():
    html = M.viewer_html(M.load("Chlorpyrifos"))
    assert "3Dmol" in html and "addModel" in html
