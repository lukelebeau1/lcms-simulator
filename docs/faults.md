# Fault library and signature features

A fault is a named change to one stage's actual value, with a size. It never changes the
pipeline: `LIBRARY[<key>].apply(scenario, magnitude)` returns a new Scenario, and the ordinary
simulation shows the consequence. Every fault in `faults/library.py` carries its stage, its
expected signature, a magnitude range for sweeping, its `sources` (keys into `data/reference/references.yaml`;
docs/references.md), and its scope: uniform when every compound's result
moves by the same share, compound when the bias differs between compounds, with what it varies with.
The scope is of the effect on the results, not the cause (docs/factors.md, "Scope"). Every uniform fault is
tested to leave the sample bias spread across analytes at zero with the noise-free detector.

| Key | Fault | Magnitude | Scope: varies with | Signature |
|---|---|---|---|---|
| stock_strong | Calibration stock stronger than labelled | actual / labelled, 1.05-4 | uniform | Uniform low bias on samples; curve, IS and same-stock QCs look normal |
| stock_degraded | Stock degraded, some analytes | remaining fraction | compound: stability in the stock | High bias on those analytes only |
| wrong_weighting | Unweighted fit | on/off | uniform (level, not compound) | Low-level accuracy fails, high levels pass |
| isomer_share | Isomer share in the standard differs from the stated one | actual / stated | compound: the summed analyte's components | One component biased; same-stock QCs pass |
| istd_overspiked | IS over-spiked in samples | actual / nominal | uniform | Uniform low bias; IS areas higher in samples |
| istd_late | IS added after extraction | on/off | compound: extraction recovery | Recovery not corrected; bias tracks recovery |
| istd_window_miss | IS window misses its peak | RT error (min) | compound: the analytes paired to that IS | IS area ~0; paired analytes unreportable |
| low_recovery | Samples extract worse than the calibration matrix | typical recovery | compound: extraction chemistry | Analyte-specific low bias; QCs pass |
| volume_error | Extraction volume error | actual / nominal | uniform | Uniform bias on samples and QCs |
| low_injection | Low injection volume | actual / nominal | uniform | Raw and IS areas drop; ratio results unchanged |
| matrix_suppression | Stronger suppression in samples | scale vs calibration matrix | compound: retention time, IS co-elution | Bias depends on RT; worst where the IS elutes elsewhere |
| interference | Interference on a quantifier | ug/kg equivalent | compound: one transition | Ion ratios fail in samples only; low levels read high |
| saturation | Detector dead time | s | compound: ion rate (strong responders, high levels) | Curve bends at the top; high samples read low |
| drift | Sensitivity drift | change per injection | uniform | IS areas trend with order; ratio results hold |
| carryover | Carryover | fraction | compound: the previous injection's level | Blanks positive after high injections |
| too_many_transitions | Extra concurrent transitions | count | uniform (precision) | Precision worsens at low level; mean unchanged |
| ph_error | Mobile phase pH wrong | pH units | compound: pKa | Ionisable analytes shift (ochratoxin A); others do not |
| organic_error | Mobile phase %B wrong | fraction B | compound: S, window edge | Every analyte shifts |
| dwell_volume | Dwell volume change | actual / nominal | compound: retention time, window width | Shift across analytes; large shifts clip peaks at window edges |

## Features (`faults/signatures.py`)

`extract(result, inputs)` returns the same numbers for any run: the sample bias and its spread
across analytes, the correlation of bias with retention time, and QC recovery. It also returns:
- the ratio of LOQ to CCV recovery
- IS area in samples over calibrators, missing IS, and the IS trend with injection order
- the retention shift and its spread
- the fraction of ion ratios outside +-30% in samples and in calibrators
- positive blanks, calibration accuracy at the lowest and highest levels, and unreportable results

A lab observation can supply any subset of these. The symptom matcher compares them.
The uniform-or-compound split is read from `sample_bias_spread`, `rt_shift_spread` and `bias_rt_corr`; the
other correlates a compound-scoped fault follows (log P, the ionisable class, m/z, level) are not features yet,
so the matcher cannot yet tell adsorption from suppression from a pH error by the bias pattern alone.
`FEATURE_MEANING` carries the one-line meaning of each feature for the views.

## Symptom matcher (`faults/matcher.py`)

The sweep table. Every fault's magnitude is swept on its own grid (`Fault.sweep`, 9 points, geometric where the
range spans more than a factor of five, one point for an on/off fault) and the run simulated with the noisy
detector at three seeds; the feature vector is the mean over the seeds. The baseline is simulated the same way,
and its spread over the seeds is kept as the replicate noise. The table is built once by
`scripts/make_fault_sweeps.py` (a few minutes) and bundled as `data/examples/fault_sweeps.csv`;
`tests/test_matcher.py` checks it matches the current library grid and recomputes one cell.

Tolerances. Each feature has a floor that matters at the bench (5% on a bias ratio, 0.05 min on a retention
shift, 0.3 on a correlation, 2% on a calibration accuracy, 0.05 to 0.1 on a fraction) and the tolerance is the
larger of that floor and twice the replicate noise, so a noisy feature (the bias-against-retention correlation,
the IS trend with order, positive blanks) counts for less.

Scoring. For an observation (any subset of the features), each fault's feature vector is interpolated between
its sweep points and the distance at magnitude m is the root mean square of (simulated - observed) / tolerance
over the features supplied; the best fit is the magnitude with the smallest distance. A feature is "matched"
when its difference is within one tolerance. "No fault in the library" is a candidate too, at the baseline. The
ranking is by best distance; candidates in the leading three closer than the tie margin (0.25 tolerance units)
are reported as not told apart, and for every leading pair the matcher names the bench checks whose outcome
differs between them (`data/examples/checks.yaml`: name, what to do, the outcome per fault, "unchanged" for a
fault not listed). The matcher reports consistency, not probability; a fault outside the library cannot be found.

Reference scenario (the plan's acceptance test): a proficiency test at 30% of assigned, uniform across
analytes, dilution ruled out. Supplied as sample_bias 0.30, spread 0, no correlation with retention, with the
extraction volume error ruled out, the matcher ranks the calibration stock and the IS over-spike first, both
consistent, and rules matrix suppression out (its bias cannot reach 30% at any size in the library's range, so
it sits more than five tolerance units from the leaders); the separating checks it names are the IS areas in
samples against calibrators and a QC from an independent stock. On the bundled data every fault ranks first
on its own default-size signature when every feature is supplied; the faults whose signatures are subtle at
that size (stock degraded for three analytes, a pH error, extra transitions, an interference, an isomer share)
tie with several others and the table names the check for each.

Checks (`checks.yaml`, provenance assumed, for the lab to review; each cites its `sources`): IS areas in samples against calibrators; a QC
from an independent stock; post-column infusion of a sample extract; spike before and after extraction;
re-inject the same vial; dilute the extract and re-inject; retention times against the method table; blanks
after the highest standard; reprocess with 1/x weighting; longer dwell with fewer concurrent transitions.
Every library fault appears under at least one check.

Confusion table. `confusion(table, checks)` takes each fault at its sweep point nearest its default magnitude,
supplies every feature, and records the fault's own rank, the faults within the tie margin of it, and the check
that separates each. It is the documented output the plan asks for: where the results alone cannot tell faults
apart and a bench experiment is required. The symptom matcher page shows it in an expander; the table below was
produced by `scripts/make_fault_sweeps.py` on the bundled data.

| Fault | Size | Rank | Ties within the margin | Separated by |
|---|---|---|---|---|
| stock_strong | 1.42 | 1 | none | |
| stock_degraded | 0.575 | 1 | 11: isomer_share, too_many_transitions, baseline, organic_error, drift, interference, matrix_suppression, ph_error, volume_error, saturation, istd_window_miss | IS areas, samples against calibrators; QC from an independent stock |
| istd_overspiked | 1.54 | 1 | none | |
| istd_late | 1 | 1 | none | |
| istd_window_miss | 1.92 | 1 | none | |
| low_recovery | 0.625 | 1 | none | |
| matrix_suppression | 2.1 | 1 | none | |
| volume_error | 1.53 | 1 | none | |
| low_injection | 0.481 | 1 | 1: drift | IS areas, samples against calibrators |
| ph_error | 0.875 | 1 | 12: too_many_transitions, stock_degraded, istd_window_miss, saturation, volume_error, dwell_volume, drift, organic_error, isomer_share, baseline, interference, matrix_suppression | Dilute the extract and re-inject; IS areas, samples against calibrators; QC from an independent stock; Re-inject the same vial; Retention times against the method table |
| organic_error | 0.025 | 1 | none | |
| dwell_volume | 2.25 | 1 | none | |
| too_many_transitions | 152 | 1 | 10: ph_error, saturation, isomer_share, stock_degraded, matrix_suppression, volume_error, interference, drift, baseline, organic_error | IS areas, samples against calibrators; QC from an independent stock; Re-inject the same vial |
| wrong_weighting | 1 | 1 | 1: saturation | Re-inject the same vial |
| saturation | 1.78e-07 | 1 | none | |
| interference | 50.1 | 1 | 11: stock_degraded, isomer_share, baseline, organic_error, drift, too_many_transitions, matrix_suppression, ph_error, volume_error, saturation, istd_window_miss | Dilute the extract and re-inject; IS areas, samples against calibrators; QC from an independent stock; Re-inject the same vial |
| carryover | 0.00532 | 1 | none | |
| drift | -0.025 | 1 | 1: low_injection | IS areas, samples against calibrators |
| isomer_share | 0.75 | 1 | 11: stock_degraded, saturation, too_many_transitions, ph_error, drift, baseline, organic_error, volume_error, interference, matrix_suppression, istd_window_miss | IS areas, samples against calibrators; QC from an independent stock |

### Views

- Fault lab: pick a fault and a magnitude (the sweep grid, 25 points), noise-free or noisy detector with the
  seed, and an injection to compare. Every analyte before and after: retention, area, ion ratio, reported result
  and bias, a bias-against-retention plot, and the run's feature vector before and after with the tolerance and
  whether each feature moved. A button sends the after-vector to the matcher.
- Matrix overlay: the matrix factor against retention time from the bundled synthetic profile or an uploaded
  pair of infusion traces [bonfiglio_1999, matuszewski_2003] (CSV with time_min and signal), smoothing and a
  suppression scale; every analyte drawn
  where it elutes with its internal standard, and a table of the share of the suppression the ratio does not
  remove (analyte factor over IS factor), worst first.
- Symptom matcher: tick the features known and enter their values (presets: the reference scenario, a late
  shift, positive blanks, or the fault lab's last run), rule out faults already checked, set the tie margin; the
  ranked table, a bar chart of each leading candidate's differences in tolerance units, the separating checks,
  and the confusion table.

## Assumptions

- pH fault: when no ionised-form retention is measured, the ionised form is taken as 1.5 log
  units less retained, with the same S.
- Analyte-specific recovery uses a fixed pseudo-random spread per analyte name, so runs are reproducible.
- Noise-free mode (`detector="expected"`) includes window clipping (Gaussian share inside the window)
  but not saturation or counting noise. Those need the simulated detector.

## Note on the example method's QCs

The example's LCS, ICV and CCV are spiked from the calibration stock
(`spiked_from_calibration_stock: true`). A stock error therefore moves them with the curve, so
they pass while independent samples, such as a PT, are biased. Only a second-source check
standard would catch it.
