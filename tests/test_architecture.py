"""Design rules that hold in every phase."""

import ast
from pathlib import Path

import lcms_sim

SRC = Path(lcms_sim.__file__).parent
UI_MODULES = {"streamlit", "plotly"}


def test_core_library_does_not_import_the_interface():
    for py in SRC.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                assert n.split(".")[0] not in UI_MODULES, f"{py.name} imports {n}"


def test_no_global_random_state():
    """All randomness goes through a generator passed in by the caller."""
    banned = ("np.random.seed", "np.random.normal", "np.random.poisson", "np.random.rand", "import random")
    for py in SRC.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for b in banned:
            assert b not in text, f"{py.name} uses {b}"
