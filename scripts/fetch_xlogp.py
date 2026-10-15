"""Fetch PubChem XLogP3, formula and monoisotopic mass for each target of the bundled example (and any compound
already in the table).

Writes data/reference/pesticide_xlogp.csv. Lookup by analyte name, then by the OCM list's CAS number.
XLogP3 is PubChem's computed octanol-water log P (an estimate, not a measurement).

Run:  python scripts/fetch_xlogp.py
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

import pandas as pd

from lcms_sim.paths import ROOT

REF = ROOT / "data" / "reference"
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/XLogP,MolecularFormula,MonoisotopicMass/JSON"
# Names PubChem knows under another name (the salt is listed as the cation).
ALIASES = {"Chlormequat chloride": "Chlormequat"}


def lookup(name: str) -> dict | None:
    try:
        with urllib.request.urlopen(PUG.format(urllib.parse.quote(name)), timeout=20) as r:
            return json.load(r)["PropertyTable"]["Properties"][0]
    except Exception:
        return None


def main() -> None:
    an = pd.read_csv(ROOT / "data" / "examples" / "analytes.csv")
    names = list(an.loc[~an.is_internal_standard, "name"])
    old = REF / "pesticide_xlogp.csv"
    if old.exists():
        names += [n for n in pd.read_csv(old).compound if n not in names]
    rows = []
    for name in names:
        hit = lookup(ALIASES.get(name, name))
        time.sleep(0.25)                                   # PubChem asks for <= 5 requests per second
        rows.append({"compound": name, "pubchem_name": ALIASES.get(name, name),
                     "cid": hit.get("CID") if hit else None, "xlogp": hit.get("XLogP") if hit else None,
                     "formula": hit.get("MolecularFormula") if hit else None,
                     "monoisotopic_mass": float(hit["MonoisotopicMass"]) if hit else None})
    out = pd.DataFrame(rows)
    out.to_csv(REF / "pesticide_xlogp.csv", index=False)
    print(out.to_string())
    print("missing xlogp:", list(out.loc[out.xlogp.isna(), "compound"]))


if __name__ == "__main__":
    main()
