"""Overview: what is loaded, and how the views follow a sample from the ampoule to the reported result."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from common import inputs, method, method_provenance, method_sources, page, provenance_table, sources_line  # noqa: E402

page("LC-MS/MS simulator",
     f"A teaching and troubleshooting model of a pesticide and mycotoxin method on an LC triple quadrupole. "
     f"Loaded: {method().name}.")

i = inputs()
n_t = sum(not a.is_internal_standard for a in i.analytes)
n_is = sum(a.is_internal_standard for a in i.analytes)
st.markdown(f"""
**Loaded:** {n_t} target compounds, {n_is} internal standards, {len(i.transitions)} transitions,
{len(i.calibration_design)} calibration levels, {len(i.samples)} samples and QCs.

**Follow** (sidebar) picks one injection and one compound. Every page then shows that portion on the rail
above: the amount at each station from the ampoule to the reported area, nominal, and in red where the actual
parted from it. The pages are grouped by where the portion physically is:

**Storage and bench**
- **Standards and extraction**: calibration and QC levels, the internal standard mix, and the extraction
  and vial volumes, from the bench to the concentration in the vial.
- **Bench steps**: every pipetting step from reference standard to injection, as amounts in each vessel;
  change what was actually pipetted and follow it to the column, down to molecules.

**Injection and column**
- **LC: gradient and retention**: flow, dwell volume, gradient and pH against the method's windows.
- **LC: inside the column**: where every band is along the column at a chosen moment, and the %B around it.

**Source and analyser**
- **MS: ion path**: a cartoon of one transition: spray and desolvation, Q1, collision cell, Q3 and the
  detector, labelled with the method's m/z values and voltages.
- **MS: how a quadrupole filters**: five steps: the saddle, one ion with every knob, the RF cup against the DC
  pull, one ion near the setting, and the collision cell.

**Detector**
- **MS: acquisition**: cycle time, windows and transition count against dwell, points per peak and
  ion ratio precision; step through one cycle transition by transition.
- **MS: molecules to response**: from the molecules injected to the ions counted, stage by stage, and how
  counted ions become the reported area.

**Result**
- **Calibration**: inject the levels through the detector model and compare fit settings.
- **Sample to result**: one sample from balance to reported result, nominal beside actual.

**Troubleshooting**
- **Fault lab**: inject one fault at a chosen magnitude and read every analyte's change and the run's feature vector.
- **Matrix overlay**: the suppression profile against retention time, every analyte beside its internal standard.
- **Symptom matcher**: enter the features you observe and rank the faults that fit, with the checks that separate
  them.

**Your data**
- **Import your data**: any quant export by column mapping, a symptom vector for the matcher, and a lab input set.

**References** lists every source the pages cite, how each was verified, and the claims that remain assumed. Each
page ends its sections with a Sources line; "assumed" there means no adequate source was found.
""")
st.caption(method().description)
sources_line(method_sources(), label="Sources (the loaded method and the regulated list)")
provenance_table(method_provenance(i))
