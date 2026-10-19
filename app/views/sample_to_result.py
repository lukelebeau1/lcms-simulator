"""Post-analytical, sample to result: one sample, every stage, nominal beside actual, to the reported result."""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import (Station, follow_index, inputs, matrix_profile, method_provenance, method_sources, page,  # noqa: E402
                    provenance_table, sources_line, targets)

from lcms_sim.io.schemas import NominalActual, SampleDefinition, TransitionRole  # noqa: E402
from lcms_sim.paths import ROOT  # noqa: E402
from lcms_sim.pipeline import Kind, RunConditions, SequenceItem, run_sequence  # noqa: E402
from lcms_sim.quant.prep import analyte_vial_conc, istd_vial_conc, on_column_ng, prep_values  # noqa: E402
from lcms_sim.rng import make_rng  # noqa: E402

page("Sample to result",
     "Follow one sample from the balance to the certificate number. Signal comes from the actual values; "
     "the reported result is calculated from the nominal ones. Set any actual value to see where a bias enters.")

i = inputs()
names = targets(i)
mn = pd.read_csv(ROOT / "data" / "reference" / "mn_ocm_cta_v2_analytes.csv")
limit = {r.analyte: r.limit_flower * (1000 if r.unit == "ppm" else 1) for r in mn.itertuples()}
p = i.prep

with st.sidebar:
    st.subheader("Sample")
    analyte = st.selectbox("Analyte", names, index=follow_index(names))
    a = next(x for x in i.analytes if x.name == analyte)
    true = st.number_input("True concentration (ug/kg)", 0.0, 100000.0,
                           float(limit.get(a.regulated_as, 100.0) * (a.isomer_fraction or 1.0)), 10.0)
    st.subheader("Actual values (nominal in brackets)")
    mass = st.number_input(f"Sample mass, g ({p.sample_mass_g.nominal})", 0.01, 10.0, p.sample_mass_g.nominal, 0.01)
    vol = st.number_input(f"Extraction volume, mL ({p.extraction_volume_ml.nominal})", 0.5, 50.0,
                          p.extraction_volume_ml.nominal, 0.1)
    spike = st.number_input(f"IS mix added, uL ({p.istd_spike_ul.nominal})", 0.0, 200.0, p.istd_spike_ul.nominal, 1.0)
    inj = st.number_input(f"Injection volume, uL ({p.injection_volume_ul.nominal})", 0.5, 50.0,
                          p.injection_volume_ul.nominal, 0.5)
    rec = st.slider("Extraction recovery in this sample vs calibrators", 0.1, 1.5, 1.0, 0.05)
    mfs = st.slider("Matrix effect in sample vs calibrators (x)", 0.0, 3.0, 1.0, 0.1,
                    help="1 = same matrix as the calibration curve; 2 = twice the suppression")
    stock = st.slider("Calibration stock actual / labelled", 0.25, 4.0, 1.0, 0.05)

istd = a.internal_standard
keep = {analyte, istd}
mini = dataclasses.replace(
    i,
    analytes=[x for x in i.analytes if x.name in keep],
    transitions=[t for t in i.transitions if t.analyte in keep],
    calibration_design=[lv.model_copy(
        update={"concentrations_ng_ml": {analyte: lv.concentrations_ng_ml[analyte]}}) for lv in i.calibration_design],
    samples=[SampleDefinition(id="Sample", matrix="cannabis flower", kind="sample",
                              true_concentration_ug_kg={analyte: true})],
    prep=p.model_copy(update={
        "sample_mass_g": NominalActual(nominal=p.sample_mass_g.nominal, actual=mass),
        "extraction_volume_ml": NominalActual(nominal=p.extraction_volume_ml.nominal, actual=vol),
        "istd_spike_ul": NominalActual(nominal=p.istd_spike_ul.nominal, actual=spike),
        "injection_volume_ul": NominalActual(nominal=p.injection_volume_ul.nominal, actual=inj),
    }),
    calibration=i.calibration.model_copy(update={"stock_concentration_ng_ml": NominalActual(
        nominal=i.calibration.stock_concentration_ng_ml.nominal,
        actual=i.calibration.stock_concentration_ng_ml.nominal * stock)}),
)
seq = [SequenceItem(Kind.calibrator, lv.level) for lv in mini.calibration_design] + [SequenceItem(Kind.sample, "Sample")]
r = run_sequence(mini, make_rng(0), seq, RunConditions(detector="expected", sample_recovery_factor={analyte: rec},
                                                       sample_matrix_scale=mfs), profile=matrix_profile())

n, act = prep_values(mini.prep, "nominal"), prep_values(mini.prep, "actual")
mix = p.istd_mix_ng_ul[istd]
smp = r.record("Sample")
cals = [x for x in r.injections if x.item.kind is Kind.calibrator]
fit = r.fits[analyte]
reported = r.reported_ug_kg["Sample"][analyte]
mf_a, mf_i = smp.matrix_factor[analyte], smp.matrix_factor[istd]
cal_mf_a = cals[0].matrix_factor[analyte]

rows = [
    ("1 Sample", "Analyte weighed in (ng)", true * n.mass_g, true * mass),
    ("2 Extraction", "Analyte in extract (ng)", true * n.mass_g, true * mass * rec),
    ("3 Volume and IS", "Analyte in vial (ng/mL)", analyte_vial_conc(true, 1, n), analyte_vial_conc(true, rec, act)),
    ("3 Volume and IS", f"{istd} in vial (ng/mL)", istd_vial_conc(n, mix), istd_vial_conc(act, mix)),
    ("4 Injection", "Analyte on column (ng)", on_column_ng(analyte_vial_conc(true, 1, n), n),
     on_column_ng(analyte_vial_conc(true, rec, act), act)),
    ("5 Ionisation", "Matrix factor, analyte / IS", cal_mf_a, mf_a),
    ("6 Detection", "Analyte area / IS area", None, smp.areas[analyte] / smp.areas[istd]),
    ("7 Calibration", "Slope (response per concentration ratio)", None, fit.slope),
    ("8 Back-calculation", "Reported (ug/kg)", true, reported),
]
# Formatted as text: st.dataframe draws a missing number as "None" whatever the Styler's na_rep says.
num = lambda v: "-" if v is None else f"{v:.5g}"  # noqa: E731
df = pd.DataFrame([(s, q, num(e), num(a)) for s, q, e, a in rows],
                  columns=["Stage", "Quantity", "Nominal / expected", "Actual"])
st.dataframe(df, hide_index=True, width="stretch")

bias = reported / true if true else float("nan")
c1, c2, c3 = st.columns(3)
c1.metric("Reported", f"{reported:.4g} ug/kg")
c2.metric("Reported / true", f"{bias:.3f}")
c3.metric("IS area, sample / calibrators", f"{smp.areas[istd] / (sum(x.areas[istd] for x in cals) / len(cals)):.2f}")

st.subheader("Where the bias comes from")
parts = {
    "Recovery (analyte; calibrators were extracted too)": rec,
    "Matrix: analyte vs its IS in the sample, relative to calibrators": (mf_a / mf_i) / (cal_mf_a / cals[0].matrix_factor[istd]),
    "Calibration stock (labelled / actual)": 1 / stock,
    "IS added (nominal / actual)": p.istd_spike_ul.nominal / spike if spike else float("inf"),
    "Steps before the IS (mass x nominal volume / nominal mass x volume)":
        (mass / p.sample_mass_g.nominal) * (p.extraction_volume_ml.nominal / vol),
    "Injection volume (cancels: after the IS)": 1.0,
}
prod = 1.0
for v in parts.values():
    prod *= v
st.dataframe(pd.DataFrame({"Factor": list(parts), "Value": [f"{v:.3f}" for v in parts.values()]}),
             hide_index=True, width="stretch")
st.caption(f"Product of factors {prod:.3f}; simulated chain {bias:.3f}. They agree because the fit is linear "
           f"and noise-free here. Internal standard for {analyte}: {istd}.")
sources_line(("stokvis_2005", "matuszewski_2003", *method_sources()),
             label="Sources (what an internal standard cancels; the matrix factor normalised to the IS; the "
                   "calibration and the IS pairs from the method; action limits; the factorisation itself is the chain's "
                   "own test oracle)")
provenance_table(method_provenance(i))
