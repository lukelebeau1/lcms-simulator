# Inputs: files, schemas, units

All inputs live in one folder with fixed file names. `load_input_set(folder)` loads every
file, validates each against its schema in `lcms_sim/io/schemas.py`, then checks names
across files. Every problem found is reported together, with file, row (header = row 1)
and column.

| File | Schema | Contents |
|---|---|---|
| `system.yaml` | `SystemConstants` | Column dimensions, t0 at a reference flow, dwell volume, plate number, extra-column sigma |
| `lc_method.yaml` | `LCMethod` | Gradient program (%B at the mixer), flow, run time, pH |
| `acquisition.yaml` | `AcquisitionMethod` | Cycle time, inter-scan delay, points-per-peak threshold, gain, background, noise RSDs, dead time |
| `analytes.csv` | `Analyte` | log kw, S, acid/base and pKa, response factor, ion ratio, assigned internal standard |
| `transitions.csv` | `Transition` | MRM table: precursor, product, quantifier/qualifier, retention time, window |
| `scouting_gradients.yaml` | `ScoutingGradients` | Programs for the scouting runs, keyed by id |
| `scouting_runs.csv` | `ScoutingRun` | Measured retention per analyte per scouting gradient |
| `infusion_matrix.csv`, `infusion_solvent.csv` | `InfusionPoint` | Post-column infusion traces |
| `prep.yaml` | `PrepProcedure` | Mass, internal standard amount and stage, volumes, dilution, injection, recoveries |
| `calibration.yaml` | `CalibrationSettings` | Solvent or matrix-matched, fit model, weighting, origin, IS and stock concentrations |
| `calibration_design.csv` | `CalibrationLevel` | Wide: `level`, then one ng/mL column per target |
| `samples.yaml` | `SampleDefinition` | Sample id, matrix, true concentration per analyte |

## Units

| Quantity | Unit |
|---|---|
| Chromatographic time | min |
| Acquisition time | ms or s, as named |
| Flow | mL/min |
| Volume | mL; injection volume uL |
| Sample mass | g |
| Internal standard spike | ng |
| Concentration in sample | ug/kg |
| Concentration in extract, calibrator, stock | ng/mL |
| Response factor | quantifier area per ng on column |

## Nominal and actual

Each prep and calibration value is a `NominalActual` pair. `actual` defaults to `nominal`.
Signal is generated from `actual`, and the reported result is calculated from `nominal`.

## Assumptions

- Retention parameters in `analytes.csv` apply at the method pH. Modelling pH shifts needs a
  second set from scouting at another pH (Phase 1 extension, off by default).
- The gradient is linear between points. Two points at one time make a step.

## Importing your own data (`lcms_sim/io/importer.py`, page "Import your data")

There is no universal parser, because the inputs come from three places. Instrument exports have a few
layouts; the procedure has none; the fitted parameters are derived, not imported.

A quant batch export is read by column mapping. `read_table` tells MassHunter's wide layout (two header
rows, "<compound> Results" over blocks of RT, Final Conc., Accuracy, Area) from a long one (one row per
injection and compound, as Sciex, Thermo and Waters export, or any hand-made table; comma or tab separated)
and makes both a long frame. In the wide layout, "Qualifier (a -> b) Results" blocks belong to the block
before them (the first one's transition and ratio go on that row as Qualifier and Qualifier Ratio), and a
target's "<IS> (ISTD) Results" block is kept once per injection as a compound of its own, since the IS
repeats under every target it serves. A name injected twice at different acquisition times (a reagent blank
repeated later in the run) becomes two injections, the later one suffixed " (2)". Checked on a real MassHunter
flower export with areas (707 columns): 39 injections, 66 targets and 14 internal standards. `guess_mapping` names the columns from header synonyms (injection, type, level,
acquired, compound, retention time, area, concentration, accuracy, expected concentration); the page lets the
user correct it. `to_long` applies the mapping into fixed columns; the type labels are read as blank,
calibrator, QC or sample from a vendor list, then from the injection names, and the page's kind table
overrides. Without a level column the calibrators' expected concentrations rank the levels. Compound names
match the method's through `normalise_name` (case, punctuation, known misspellings).

From the long table: `summarise` gives one row per compound (retention median and spread over the found
calibrator and QC peaks, calibration accuracy range and points outside 80-120%, QC accuracy mean and RSD,
blanks, the lowest calibrator's concentration), and `observe` gives the symptom vector the matcher takes,
for every feature the export supports, with a note per feature saying how it was read or why it could not
be: sample bias needs expected values for the samples (an accuracy column, or an expected concentration);
the IS features need an area column and IS names matching the method's; the LOQ/CCV ratio needs QCs named
or levelled LOQ and CCV; blanks are compared with the lowest calibrator in the export's own units, one unit
for every row assumed; the ion ratio features are not computed from an export yet, although the first
qualifier's ratio is kept when the export carries it. Tested as a round trip: a simulated run written in MassHunter's shape and again as a long Sciex-like
table imports to the symptom vector `faults/signatures.py` computes from the run itself.

A transition table is read the same way (`TRANSITION_ROLES`: compound, precursor, product, retention time,
window, collision energy, fragmentor, cell accelerator, polarity, role, IS flag, dwell) into `Transition`
models by `transitions_from_table`; every schema problem is listed with its row. With no role column the
first product of each compound is its quantifier. Qualifier ion ratios are left empty until measured. The
window defaults to a stated value when the table has none. Round-trips the bundled method.

A lab input set (`make_lab_inputs`, data/lab/inputs/) is every file the loader reads, copied from the bundled
example as a template unless already there, the imported transitions written over transitions.csv, and a
README naming each file's origin and listing every schema field with its description. `validate_inputs`
runs the loader and returns the problem list. The page can switch the app to that folder (session state
`input_folder`; `common.inputs()` reads it) and back. The fitted parameters in analytes.csv stay the
example's until a lab's own runs are fitted: that is the bridge that waits on an area export.

Everything the importer writes goes under data/lab/<batch>/ (long.csv, summary.csv, observation.json,
transitions.csv), which git ignores.

## Example data

`scripts/make_example_data.py` builds `data/examples/` for the Minnesota OCM cannabis list
(`data/reference/mn_ocm_cta_v2_analytes.csv`). Transitions, fragmentor and CE come from Agilent
5994-1734EN Appendix A (`data/reference/agilent_5994-1734en_transitions.csv`, parsed by
`scripts/parse_agilent_appnote.py`). The app note covers 57 of the 59 listed analytes;
Azadirachtin and Chlormequat chloride are listed in `data/examples/GAPS.md`. Summed analytes
(Abamectin, Spinosad, Pyrethrins) appear as their components, with `regulated_as` naming the sum.
Column, gradient, retention times and sample preparation are the note's; `method.yaml` lists where every input
comes from, and the pages read it.

Internal standards, calibration and QC levels, isomer fractions, S, response factors, ion ratios, the
detector gain, the matrix profile and the samples are assumed; see the script docstring and `docs/lc.md`.
A lab builds its own set the same way (`Method`, `write_set`) from its own documents into data/lab/, which
git ignores; the Import page's input-set picker loads any folder there.
