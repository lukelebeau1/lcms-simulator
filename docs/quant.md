# Quantitation chain

## Stages

    weigh m -> [IS before extraction] -> extract, recovery R -> [IS after extraction]
      -> make to volume V -> dilute D -> [IS in vial, volume Vv] -> inject Vinj
      -> ionise, matrix factor MF -> detect -> calibrate -> back-calculate

Every prep value is a nominal / actual pair (`prep.yaml`). Signal comes from the actual values
and the reported result from the nominal ones, so any stage where they differ is a fault.

| Quantity | Sample | Calibrator |
|---|---|---|
| Analyte in the vial (ng/mL) | C x m x R / (V x D) | level x stock actual/nominal x per-analyte stock factor |
| IS in the vial (ng/mL) | before extraction: I x R_i / (V x D); after: I / (V x D); in vial: I / Vv | IS concentration (actual) |
| On column (ng) | vial x Vinj (actual) | vial x Vinj (nominal) |
| Matrix factor | profile at its retention time, if the sample has a matrix | profile if matrix-matched, else 1 |

## Calibration and back-calculation

- Without IS: y = area and x = vial concentration.
- With IS: y = area ratio and x = concentration ratio, using the nominal IS concentration in
  the calibrators.
- Fit: linear or quadratic; weighting none, 1/x or 1/x² [almeida_2002, gu_2014]; origin ignore, include or force.
  Outputs are coefficients, weighted R², and back-calculated accuracy per level. The example's acceptance is
  R² >= 0.99 [agilent_5994_1734en] and every level within +-20% [sante_11312_2021]; a lab input set carries its
  own (acceptance.yaml). The Calibration view injects each level through the
  detector model (counting noise, proportional noise, optional dead time [knoll_2010]) and compares two fits.
- For a sample, x is read from the inverse of the fit. With an IS, the vial concentration is x
  times the nominal IS vial concentration. Reported ug/kg = vial concentration x V x D / m,
  all nominal.

## Bias relation (test oracle)

Matrix factors are the response in matrix over the response in solvent [matuszewski_2003], mapped against
retention time by post-column infusion [bonfiglio_1999]; the suppression arises in the droplets [king_2000].

For a linear fit with zero intercept and no noise:

    C_reported / C_true = (R_a MF_a,s) / (R_i MF_i,s) x (MF_i,c / MF_a,c) x (C_std,nominal / C_std,actual)
                          x (I_c / I_s) x (f_actual / f_nominal)

Here f is the product of the prep factors that act on the analyte before the IS joins. Without
an IS, drop the i and I terms and let f cover every step, injection volume included.
`tests/test_quant.py` checks each case to 0.1%:
- stock stronger than labelled
- one analyte degraded in the stock
- IS over-spiked in samples, or in calibrators
- recovery with a matched IS, an unmatched IS, a late IS, and no IS
- dilution error before and after the IS joins
- mass error
- low injection volume, with and without IS
- matrix suppression with a co-eluting IS, an IS eluting elsewhere, matrix-matched calibration, and no correction

## Internal standards and calibrators

A labelled analogue of the analyte corrects what a surrogate internal standard cannot, because it co-elutes and
shares the chemistry [stokvis_2005, matuszewski_2003]. The Sample to result view factors one sample's reported / true
ratio into recovery, the analyte's matrix factor over its IS's relative to the calibrators [matuszewski_2003], the
stock, the IS volume, the steps before the IS, and the injection volume, which cancels once the IS is in the vial
[stokvis_2005]; the product equals the simulated chain because the fit is linear and noise-free (the bias relation).

- Each IS has its own concentration in the mix (`prep.istd_mix_ng_ul`, ng/uL). The amount added
  is the spike volume x that concentration. A mix made wrong for one IS is `istd_mix_factor_actual`.
- `calibration.preparation: vial` (the example, as in the application note): calibrators are made in blank
  flower extract after the extraction, so they carry the matrix but not the recovery. `procedural`: spiked into
  blank matrix and extracted like samples, so they carry the recovery too. Either way the levels are the
  nominal vial concentrations and the calibrators get the same IS addition as samples.
- Summed analytes: each component's level is entered as the total, the calibrator holds
  total x isomer fraction, and the reported component is multiplied by the fraction.
  `RunResult.totals()` adds components into regulated analytes.

## Sequence

`run_sequence` simulates injections in order. Carryover adds a fraction of the previous
injection's on-column amount. Drift multiplies sensitivity by (1 + drift x injection index).
Drift moves raw areas, but area ratios hold while signal stays well above the limit of quantitation.
`detector="expected"` gives noise-free areas, used for oracle tests and for the matcher's sweeps.

## Assumptions

- Calibrators are vial solutions injected at the nominal injection volume, and are not extracted.
  The app note's matrix-matched calibrators are spiked after extraction, as here.
- An internal standard's recovery is read from `prep.recovery` under the IS name, default 1.
- Labelled internal standards have the same response factor as their analyte. The example's nine are assumed,
  because the app note uses none: labelled analogues bracketing the run from daminozide to permethrin, 10 ng/mL
  in the vial, each target paired with the nearest in retention time and polarity. Their transitions are the
  parent's quantifier shifted by the label, on the product only where the label stays on it (label positions
  read from the PubChem structures; the reason is recorded per standard in scripts/make_example_data.py).

## Preparation ledger (`ledger.py`, `data/examples/prep_recipe.yaml`)

The method's preparation as bench steps (stock, solvent, transfer, sample, extract, inject), each
volume or mass a NominalActual. `run_recipe` tracks every vessel's volume and the amount (ng) of
each compound; a transfer of v from a vessel holding V moves v / V of every amount, so
concentrations are derived and dilutions need no formula. Run with nominal values it reproduces
the closed-form values: the calibration design's vial levels, the QC and
sample vial concentrations of `prep.py`, and the IS in the vial (tested). Amounts are conserved
(tested). Run with actual values, a single changed step carries to every vessel made from it and
no other (tested). `molecules` converts ng to molecules (Avogadro, molecular weight). The recipe
is generated by `scripts/make_example_data.py`: the application note's extraction (1 g, 15 mL acetonitrile, C18
SPE, two 5 mL rinses, made to 25 mL) and final dilution (50 uL in 500 uL), with assumed stock dilutions, QC
spikes and the IS solution added as 50 uL of the 450 uL diluent. Not yet used by
`pipeline.run_sequence`, which still takes the closed-form prep and calibration values.
