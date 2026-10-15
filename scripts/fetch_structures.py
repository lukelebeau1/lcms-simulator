"""3D structures for every analyte and internal standard: PubChem 3D conformer, or RDKit-built where PubChem has none.

Writes data/reference/structures/<name>.sdf and data/reference/structures.csv
(name, cid, smiles, coords: "pubchem 3d" | "rdkit etkdg+mmff", note).
PubChem computes no 3D conformer for very large or flexible molecules (abamectin, spinosyns, azadirachtin);
those are embedded with RDKit (ETKDG, then MMFF94). Internal standards not in PubChem under their own name
fall back to the parent compound, flagged: the deuterium positions are then unknown.

Run:  python scripts/fetch_structures.py
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from lcms_sim.paths import EXAMPLES, ROOT

REF = ROOT / "data" / "reference"
OUT = REF / "structures"
UA = {"User-Agent": "LCMSIM-teaching-simulator/0.1"}
PROP = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{}/{}/property/SMILES/JSON"
SDF3D = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/record/SDF?record_type=3d"
# Method names that PubChem knows differently (spelling in the method data is kept everywhere else).
IS_ALIASES = {"Imazilil-D5": "Imazalil-d5", "Imidicloprid-D4": "Imidacloprid-d4", "Myclobutanil-D4": "Myclobutanil-d4",
              "Permethrins-D5": "Permethrin-d5"}
IS_PARENT = {"Imazilil-D5": "Imazalil", "Imidicloprid-D4": "Imidacloprid", "Myclobutanil-D4": "Myclobutanil",
             "Permethrins-D5": "Permethrins", "Ivermectin-D2": "Ivermectin"}


def get(url: str, as_json: bool = True):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        b = r.read()
    return json.loads(b) if as_json else b.decode("utf-8")


def smiles_for(kind: str, key) -> tuple[int, str] | None:
    try:
        p = get(PROP.format(kind, urllib.parse.quote(str(key))))["PropertyTable"]["Properties"][0]
        return int(p["CID"]), p["SMILES"]
    except Exception:
        return None


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)


def embed(smiles: str) -> str:
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 7                    # fixed: the same structure on every run
    if AllChem.EmbedMolecule(m, params) != 0:
        params.useRandomCoords = True
        AllChem.EmbedMolecule(m, params)
    AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    return Chem.MolToMolBlock(m)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    targets = pd.read_csv(REF / "pesticide_xlogp.csv")[["compound", "cid"]]
    an = pd.read_csv(EXAMPLES / "analytes.csv")
    rows = []
    jobs = [(r.compound, ("cid", int(r.cid)), "") for r in targets.itertuples()]
    for n in an[an.is_internal_standard].name:
        jobs.append((n, ("name", IS_ALIASES.get(n, n)), ""))
    for name, (kind, key), note in jobs:
        hit = smiles_for(kind, key)
        time.sleep(0.25)
        if hit is None and name in IS_PARENT:
            parent = targets.set_index("compound").cid.get(IS_PARENT[name])
            hit = smiles_for("cid", int(parent)) if parent == parent and parent is not None else \
                smiles_for("name", IS_PARENT[name])
            note = "parent structure: deuterium positions unknown"
            time.sleep(0.25)
        if hit is None:
            print("not found:", name)
            continue
        cid, smi = hit
        try:
            block, coords = get(SDF3D.format(cid), as_json=False), "pubchem 3d"
        except Exception:
            block, coords = embed(smi), "rdkit etkdg+mmff"
        time.sleep(0.25)
        (OUT / f"{safe(name)}.sdf").write_text(block, encoding="utf-8")
        rows.append({"name": name, "cid": cid, "smiles": smi, "coords": coords, "note": note,
                     "file": f"structures/{safe(name)}.sdf"})
        print(f"{name:24s} {coords:18s} {note}")
    pd.DataFrame(rows).to_csv(REF / "structures.csv", index=False)
    print(len(rows), "structures written")


if __name__ == "__main__":
    main()
