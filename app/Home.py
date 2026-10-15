"""LC-MS/MS simulator: entry point. Follow is the main page; the other views are grouped by the container they probe."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import follow_controls  # noqa: E402

VIEWS = "views/"

nav = st.navigation({
    "": [
        st.Page(VIEWS + "follow.py", title="Follow the analyte", default=True),
        st.Page(VIEWS + "overview.py", title="Overview"),
        st.Page(VIEWS + "references.py", title="References"),
    ],
    "Vials and tubes": [
        st.Page(VIEWS + "standards_extraction.py", title="Standards and extraction"),
        st.Page(VIEWS + "bench_steps.py", title="Bench steps"),
    ],
    "Column": [
        st.Page(VIEWS + "lc_gradient.py", title="LC: gradient and retention"),
        st.Page(VIEWS + "lc_column.py", title="LC: inside the column"),
    ],
    "Spray and capillary": [
        st.Page(VIEWS + "ms_ion_path.py", title="MS: ion path"),
    ],
    "Quadrupoles and cell": [
        st.Page(VIEWS + "ms_quad_filter.py", title="MS: how a quadrupole filters"),
    ],
    "Multiplier and counting": [
        st.Page(VIEWS + "ms_acquisition.py", title="MS: acquisition"),
        st.Page(VIEWS + "ms_molecule_budget.py", title="MS: molecules to response"),
    ],
    "Result": [
        st.Page(VIEWS + "calibration_bench.py", title="Calibration"),
        st.Page(VIEWS + "sample_to_result.py", title="Sample to result"),
    ],
    "Troubleshooting": [
        st.Page(VIEWS + "fault_lab.py", title="Fault lab"),
        st.Page(VIEWS + "matrix_overlay.py", title="Matrix overlay"),
        st.Page(VIEWS + "symptom_matcher.py", title="Symptom matcher"),
    ],
    "Your data": [
        st.Page(VIEWS + "import_data.py", title="Import your data"),
    ],
})
follow_controls()
nav.run()
