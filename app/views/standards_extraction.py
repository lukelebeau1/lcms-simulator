"""Pre-analytical: standards, QCs, internal standard and extraction, from the bench to the vial."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import (Station, follow_index, inputs, method_provenance, method_sources, page, provenance_table,  # noqa: E402
                    sources_line, targets)

from lcms_sim.quant.prep import analyte_vial_conc, back_calculate, istd_vial_conc, prep_values  # noqa: E402

page("Standards and extraction",
     "What the method puts in each vial: calibration levels, QC spikes, the internal standard mix and the "
     "extraction volumes. These are the method's nominal values; set actual values in Sample to result.", station=Station.bench)

i = inputs()
names = targets(i)
nom = prep_values(i.prep, "nominal")
cal = i.calibration

with st.sidebar:
    st.subheader("Compound")
    analyte = st.selectbox("Compound", names, index=follow_index(names))
a = next(x for x in i.analytes if x.name == analyte)
frac = a.isomer_fraction or 1.0

# Extraction and vial
st.subheader("Extraction and vial")
extract_ul = (nom.vial_ml * 1000 - nom.istd_spike_ul) if nom.vial_ml and nom.istd_spike_ul else None
c = st.columns(3)
c[0].metric("Sample mass", f"{nom.mass_g:g} g")
c[1].metric("Extraction", f"{nom.extraction_ml:g} mL")
c[2].metric("Injection", f"{nom.injection_ul:g} uL")
c = st.columns(3)
c[0].metric("Extract in vial", f"{extract_ul:g} uL" if extract_ul else "-")
c[1].metric("IS mix in vial", f"{nom.istd_spike_ul:g} uL" if nom.istd_spike_ul else "-")
c[2].metric("Sample to vial", f"{nom.extraction_ml * nom.dilution / nom.mass_g:.2f} mL/g")
st.caption(f"Internal standard added {i.prep.istd_stage.value.replace('_', ' ') if i.prep.istd_stage else 'not used'}. "
           f"Reported ug/kg = vial ng/mL x {nom.extraction_ml * nom.dilution / nom.mass_g:.2f}.")

# Calibration levels
st.subheader("Calibration levels")
st.caption(f"Stock {cal.stock_concentration_ng_ml.nominal:g} ng/mL; "
           f"{'matrix-matched' if cal.matrix_matched else 'solvent'} calibrators, prepared "
           f"{'through the extraction like samples' if cal.preparation == 'procedural' else 'directly in the vial'}. "
           + (f"{analyte} is {100 * frac:g}% of its standard, so its own amount is that share of each level."
              if frac < 1 else ""))
rows = []
for lv in i.calibration_design:
    v = lv.concentrations_ng_ml.get(analyte)
    if v is None:
        continue
    row = {"Level": lv.level, "In vial, as labelled (ng/mL)": v}
    if frac < 1:
        row[f"{analyte} itself (ng/mL)"] = v * frac
    row["Equivalent in sample (ug/kg)"] = back_calculate(v, nom)
    rows.append(row)
st.dataframe(pd.DataFrame(rows).style.format(precision=4), hide_index=True, width="stretch")

# QCs
st.subheader("Quality controls")
qcs = [s for s in i.samples if s.kind == "qc" and analyte in s.true_concentration_ug_kg]
if qcs:
    st.dataframe(pd.DataFrame([{
        "QC": s.id, "Matrix": s.matrix,
        "Spiked (ug/kg)": s.true_concentration_ug_kg[analyte],
        "Expected in vial (ng/mL)": analyte_vial_conc(s.true_concentration_ug_kg[analyte], 1.0, nom),
        "From the calibration stock": s.spiked_from_calibration_stock,
    } for s in qcs]).style.format(precision=4), hide_index=True, width="stretch")
    st.caption("A QC spiked from the calibration stock moves with the curve: a stock error does not show in it.")
else:
    st.caption(f"No QC in the loaded samples contains {analyte}.")

# Internal standards
st.subheader("Internal standard mix")
users: dict[str, list[str]] = {}
for x in i.analytes:
    if x.internal_standard:
        users.setdefault(x.internal_standard, []).append(x.name)
st.dataframe(pd.DataFrame([{
    "Internal standard": n, "In mix (ng/uL)": m, "In vial (ng/mL)": istd_vial_conc(nom, m),
    "Used by": len(users.get(n, [])), "Selected compound's IS": n == a.internal_standard,
} for n, m in i.prep.istd_mix_ng_ul.items()]).style.format(precision=4), hide_index=True, width="stretch")
sources_line((*method_sources(), "sante_11312_2021", "stokvis_2005"),
             label="Sources (every value is the method's; a stock error only shows against a second, independently prepared "
                   "standard; what the internal standard corrects)")

provenance_table(method_provenance(i))
