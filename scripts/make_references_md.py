"""Write docs/references.md from data/reference/references.yaml and the citing sites.

    .venv/Scripts/python scripts/make_references_md.py

tests/test_references.py fails when the file is stale."""

from lcms_sim.paths import ROOT
from lcms_sim.references import references_markdown

if __name__ == "__main__":
    out = ROOT / "docs" / "references.md"
    out.write_text(references_markdown(), encoding="utf-8")
    print(f"wrote {out}")
