"""Locations of bundled data."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "data" / "examples"
REFERENCE = ROOT / "data" / "reference"
LAB = ROOT / "data" / "lab"
LAB_REFERENCE = LAB / "reference"     # tables parsed from the lab's own documents; never in version control
