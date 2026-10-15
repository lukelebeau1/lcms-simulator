# MS acquisition and signal model

## What happens to a peak between the column and the area

1. **Scheduling (dynamic MRM).** Each transition is active inside its retention window. The
   cycle time T is fixed; the n(t) active transitions share it:
   `t_dwell = (T - polarity overhead) / n(t) - t_delay`. More concurrent transitions means
   shorter dwell, not a longer cycle [agilent_5990_3595]. Points across a peak = 4 sigma / T, flagged below 10:
   a chosen threshold, between the 15-point convention [waters_720001168], the 7 points found sufficient for
   narrow peaks [zeng_2023] and a vendor's 6-8 at half height [sciex_scheduled_mrm].
2. **Counting.** While a transition is monitored, ions arrive at a rate set by the peak.
   During a dwell d the detector counts c ~ Poisson(rate x d) [knoll_2010]. Reported abundance is c / d,
   scaled by the gain, so the expected area does not depend on dwell.
3. **Precision.** The ions counted over a peak are N = A / gain x d / T. Area RSD ~ 1/sqrt(N)
   from counting alone. Shorter dwell, fewer ions, worse precision at the same mean. That is
   the "too many concurrent transitions" fault: noisy, not biased.
4. **Proportional noise.** Two multiplicative terms with user-set RSD. The per-injection term
   is shared by every transition, so it cancels in ion ratios and in internal standard ratios.
   The per-point term does not cancel.
5. **Background.** A constant abundance adds counts, and so noise, everywhere. It is
   estimated from the window points outside the integration range and subtracted.
6. **Saturation (optional).** Non-paralysable dead time: observed rate = true / (1 + true x tau) [knoll_2010].
   High standards and samples read low and the curve bends at the top.
7. **Integration.** Trapezoid over the sampled points from centre - 4 sigma to centre + 4 sigma + 6 tau.
   Points outside the acquisition window do not exist, so a peak that drifts toward the edge
   of its window loses area (`clipped`).
8. **Ion ratio.** Qualifier area / quantifier area. Its counting limit is
   RSD ~ sqrt(1/N_quant + 1/N_qual), and it holds when background is negligible. The default
   identification tolerance is +-30% relative [sante_11312_2021].

## Units

| Quantity | Unit |
|---|---|
| Abundance | area units per min |
| Area | area units (abundance x min) |
| Gain | area per ion arriving at the detector over the peak at 100% duty |
| Dwell, cycle, delay, polarity switch | ms |
| Dead time | s |

## Estimating the gain from your data

Inject one standard at a low level five or more times. Take RSD from replicate areas,
mean area A, and the method's dwell d and cycle T at that retention time.
Then N = 1 / RSD^2 and gain = A x d / (T x N). The estimate is an upper bound on N if other
noise contributes; use the lowest level whose RSD is clearly counting-limited.

## Assumptions

- Each sample point is the abundance at the middle of its dwell, not the average over it.
- Within a cycle, positive transitions run before negative ones, with two polarity switches
  when both run. The Ultivo switching time is a user setting, default 0.
- Abundance and area units are internally consistent but are not MassHunter's units. Match
  them through the gain and the response factors.
- Transmission through Q1 is a hook that defaults to 1. A separate Q1 simulator exists.

## Tests

`tests/test_ms.py` checks the dwell formula exactly. It checks that areas are unbiased with
RSD within 10% of 1/sqrt(N), and that ion ratio RSD is within 10% of the counting limit from
100 to 100,000 ions. It also covers injection noise cancelling in ratios, background
subtraction, window clipping, saturation, and identical output for the same seed.

## Inside one cycle

`acquisition.cycle_layout(schedule, acq, t)` reads back, for the cycle containing t, each active
transition's slot (start, dwell, inter-scan delay) and the polarity-switch intervals [agilent_5990_3595]. It is the
same timing the signal model samples at (slot centre = sample time; tested). The Acquisition
view draws the cycle as a strip and steps through it one transition at a time with a slider:
the Q1 > Q3 pair being passed, the ions of that transition counted in its dwell, and for every
compound eluting at that moment the share counted and the ions counted over its peak. Arrival
rates use the expected (noise-free) peak at the chosen calibrator.

## Ion path view (illustration, not a model)

"MS: ion path" draws one transition as a cartoon [yost_enke_1978, paul_steinwedel_1953]: droplets shrinking as solvent evaporates,
ions released, Q1 passing only the precursor m/z (other ions oscillate out to the rods), the
collision cell breaking precursors into this compound's products in proportion to the method's
relative responses, Q3 passing only the chosen product, and hits on the electron multiplier.
The motion is drawn, not computed: ESI and quadrupole ion optics stay out of scope (build plan).
What is real on it: the transition's m/z values, fragmentor and collision voltages, polarity,
the dwell at that retention time, and the compounds the LC model has co-eluting. A stage table
says which simulator term stands for each stage (response and matrix factor, relative
response, counting noise and gain).

### Spray close-up (`app/spray_cartoon.py`)

A zoomed cartoon of electrospray under the ion path: the Taylor cone [taylor_1964] and where its excess charge
comes from (electrolysis at the needle [blades_1991, van_berkel_1995]), droplets holding solvent and analyte molecules with charge
at the surface, solvent leaving as vapour that the counter-flowing drying gas sweeps away [yamashita_fenn_1984, agilent_5990_3494], Rayleigh
fission [rayleigh_1882] (offspring about 2% of the mass, 15% of the charge [kebarle_verkerk_2009, gomez_tang_1994]), released analyte and solvent/additive
ions [iribarne_thomson_1976, dole_1968], and droplets that reach the inlet still wet [page_2007, page_2009]. Sliders for %B (defaults to the composition
eluting with the compound), drying gas temperature and flow, nebuliser pressure and spray voltage
steer it through real solvent properties (vapour pressure, surface tension of water and methanol [crc_handbook_97])
and two textbook rules: the d-squared evaporation law [langmuir_1918] and the Rayleigh charge limit [rayleigh_1882]. The scalings for
gas, pressure and voltage are monotonic in the known direction but not calibrated. It does not feed
the simulated signal.

## Molecules to response view

"MS: molecules to response" starts from the molecules injected (bench-step ledger, with any edits
from Bench steps) and multiplies stage fractions down to ions counted, then scales the count back
up to the reported area: counted ions x cycle / dwell x gain [knoll_2010]. With the default fractions it
reproduces the simulator's expected area, response factor x ng x matrix factor x relative
response (= `expected_counted_ions` x cycle / dwell x gain). Stages from the model: matrix factor
at the retention time [matuszewski_2003, bonfiglio_1999], relative response for qualifiers, dwell / cycle. The
simulator only knows the product response factor / gain for the rest; the page splits it with typical orders of
magnitude for Q1, collision cell, Q3 and detector (assumed, editable; the ranges from [dawson_1995,
hoffmann_stroobant_2007, douglas_french_1992]) and gives ionisation and inlet sampling the remainder
[page_2007, kebarle_verkerk_2009]. The stage table's Source column names the sources (`references.STAGE_SOURCES`).
Absolute counts stay illustrative until a lab's area export sets the response factors and gain.

### Q1 close-up (`app/ion_stream.py`, ion path view)

The ions entering Q1 at a chosen moment, from a chosen injection: every compound eluting at its
precursor m/z, with a rate from the bench-step ledger amount, the LC peak at that time, the matrix
profile and the model's response factor / gain divided by the typical downstream fractions (the
same split as the molecules-to-response view), plus an assumed background (solvent/additive
clusters falling off with m/z, matrix ions near m/z 380 where the matrix profile suppresses).
Q1 is an assumed Gaussian pass window, 0.7 m/z FWHM (the example method's unit resolution [agilent_5994_1734en]), 0.3 peak transmission. The view steps
through the dMRM cycle at that moment (cycle time, or a fixed dwell with the cycle it implies and
the points across the peak that leaves), shows the spectrum with the window, ion paths through
the rods (drawn, not computed) and per-dwell counts: ions arriving, passing, who they are, and the
compound's ions lost while Q1 is elsewhere. The spray close-up draws its analyte molecules from the
same population in the same colours. Not part of the signal model.

## Crosstalk (`crosstalk.py`)

A transition j counts any ion that Q1 passes at its precursor setting and Q3 passes at its product
setting [hoffmann_stroobant_2007, dawson_1995]. A co-eluting compound k contributes

    factor(j <- k) = sum over k's transitions t:  r_t x T1(P_j - P_t) x T3(p_j - p_t) x yield

of its quantifier peak, at k's own retention and shape. T are Gaussian pass windows of FWHM
q1_fwhm_mz and q3_fwhm_mz (0.7, the method's unit resolution [agilent_5994_1734en]; the Gaussian shape is
assumed); r_t is k's relative response from the
method (placeholders); k's fragments are only those its own transitions list. `yield` is the share of
k's fragment that really forms under j's conditions (collision energy): unknown until k is injected
alone, so `crosstalk` is off by default and `crosstalk_yield` (default 1) is set by the user. The
noise-free path adds the part of k's peak inside j's acquisition window and integration range; the
simulated path adds k's peak to j's signal before counting, so integration takes what falls inside.
Tests: leak both ways for a shared precursor with products 0.1 apart, none across polarity or 2 m/z,
no strong pair in the example (bifenazate and prallethrin share 301.1 but elute 0.75 min apart), a labelled
standard moved onto paclobutrazol's ions found both ways, the acquired area adds the leak, and that pair's
paclobutrazol qualifier ratio rises most at the lowest calibrator, linearly in the yield.
The MS: acquisition view lists the pairs and shows ion ratios across the run for a chosen yield,
with the yield at which the lowest calibrator's ratio leaves the +-30% tolerance [sante_11312_2021]. The check
(inject the leaking compound alone and read the receiving transition) is the bioanalytical guideline's selectivity
test for the internal standard channel [ich_m10_2022] applied to a shared product ion.

### Stability diagram (`app/quad_stability.py`, Q1 close-up)

Mathieu stability parameters a = 8zeU/(m r0^2 Omega^2), q = 4zeV/(m r0^2 Omega^2) [dawson_1995]. The first stability
region for both rod pairs lies under min(-a0(q), b1(q)) (Abramowitz and Stegun 20.2.25 series [abramowitz_stegun_1964]; apex
q = 0.706, a = 0.237), checked against a direct Floquet calculation (tests/test_quad_stability.py).
Fixed U, V put all ions on a = (2U/V) q with q ~ 1/m; the stretch inside the region is the ideal
window, narrowing to zero at the apex (U/V = 0.1678). The view sets U/V from a chosen width and shows
the arriving ions on the line, the RF amplitude (assumed 4 mm rod gap, 1 MHz) and the fraction of V
that moves the window 0.1 m/z. Real edges are sloped (finite RF cycles in the rods); illustration
only, not part of the signal model.

### Inside the rods (Q1 close-up)

One U and one V per quadrupole, applied as +(U + V cos wt) to one rod pair and -(U + V cos wt) to the
other [paul_steinwedel_1953]. The view animates an end-on cross-section (rod signs flip each RF cycle) with three ions:
the setting, one lighter and one heavier by a chosen m/z. Their sideways paths are computed from the
Mathieu equation for the current U/V (`quad_stability.paths`, x plane a, q; y plane -a, -q; start 1%
of the gap off axis): the lighter ion is lost in the + pair's plane, the heavier in the - pair's
(tested). Static side plots show both planes over the whole flight.

### The saddle (Q1 close-up)

The potential between the rods, (U + V cos wt)(x^2 - y^2) / r0^2, drawn as a 3D saddle that flips with
the RF (Play), with a switch for DC + RF (Q1, Q3), DC only and RF only (collision cell). Beside it,
each ion's averaged restoring strength per plane as an equivalent static curvature, computed exactly
from one RF cycle (`quad_stability.effective_a`): a bowl beta^2 when stable, a hill -mu^2 when it grows
steadily (the DC wins), and "driven out by the RF" when the growth is locked to the RF (trace < -2).
The textbook pseudopotential (bowl q V / 4, a_eff = q^2 / 2 [dehmelt_1967, gerlich_1992]) is its small-q limit (tested); at Q1's
q ~ 0.7 it overestimates the bowl by ~5% and would wrongly keep ions near the window edge, so the
exact value is used. DC only: bowl along x, hill along y for every m/z (Earnshaw: no static bowl).

### How a quadrupole filters (`app/views/ms_quad_filter.py`)

A teaching page in five steps: the saddle built from the rods' own pushes (line charges, a simplification of the
hyperbolic field round rods approximate near the axis [paul_steinwedel_1953, dawson_1995, paul_1990]; no static field
holds a charge in every direction, Earnshaw's theorem [gerlich_1992, paul_1990]), one ion with every knob, the RF cup
against the DC pull (exact stability edges in volts [dawson_1995, abramowitz_stegun_1964]; the default DC gives the
example method's 0.7 m/z unit resolution [agilent_5994_1734en]), one ion near the Q1 setting, and the collision cell
(next section).

Step 2 (one ion, every knob): ion mass, charge, RF amplitude V, RF frequency f, rod gap r0, DC U and starting
offset. They enter only through a = 8zeU / (m r0^2 W^2) and q = 4zeV / (m r0^2 W^2), W = 2 pi f
(`quad_stability.mathieu_aq`; only m/z matters, q goes as 1/f^2 and 1/r0^2, tested) [paul_steinwedel_1953,
dawson_1995]. The path is the Mathieu equation in both planes; the verdict comes from the one-cycle Floquet trace per
plane [abramowitz_stegun_1964] (bowl: trapped; trace < -2: shaken out; trace > 2: the DC wins), plus whether the swing
from the chosen start reaches a rod. The slow swing (the cup's period, the secular motion at beta W/2) is 2 / beta RF
cycles [dawson_1995, dehmelt_1967, gerlich_1992] (`slow_swing_cycles`; ~2 sqrt 2 / q at small q, tested). "Once per RF
cycle" samples the same phase of each cycle, which removes the jiggle and leaves the slow swing. Ion at rest at
the start, no fringing fields. Illustration; not used by the signal model.

### The collision cell (`app/collision_cell.py`, step 5 of the quadrupole page)

Ions through an RF-only quadrupole (U = 0) filled with N2 [douglas_french_1992, dawson_1995]. Radial motion: the Q1 field with U = 0; axial: a steady
push across the cell. Collisions: elastic hard spheres with N2 at 300 K, isotropic in the centre-of-mass frame, at
rate n sigma g (g relative speed; null-collision sampling [skullerud_1968]). Mean energy kept per collision with gas
at rest is 1 - 2mM/(m+M)^2 (tested); the centre-of-mass share M/(m+M) is the most a collision can turn into internal
energy [hoffmann_stroobant_2007]. Fragmentation is a change of mass at a chosen collision number, velocity kept, so q
jumps by precursor/product (kinetic energy release neglected [hoffmann_stroobant_2007]); which bond breaks and when
are inputs, not predictions (out of scope). One ion is shown in
full; 200 ions give the shares reaching Q3, lost on rods (precursor or fragment), unbroken, scattered back or
stalled, and the time in the cell.

Tested: energy and momentum conserved per collision; no gas, slow ions held at q 0.85 and lost at 0.95; gas cools
a 20 eV ion by > 100x; a fragment past the cutoff is lost and one below it passes; same seed, same ions.

Found while building it: with gas the cutoff is soft. At 60 V, m/z 49 from 184 (q 0.75) reaches Q3 30% of the
time, m/z 95 at 120 V (q 0.77) 84%, m/z 143 at 200 V (q 0.86) 27%, against 100% for products at q 0.26-0.51.
Collisions at awkward RF phases push near-edge fragments out, light ones most.

Assumed, not the Ultivo's: a quadrupole cell (Agilent does not publish the Vortex cell's geometry; Agilent's
Cyclone ion guide is a hexapole, and a hexapole or octopole cell would have a softer cutoff [gerlich_1992]), r0 4 mm,
1 MHz, 8 cm, the pressure (default 5 mTorr), the cell RF (default 60 V) and the cross-section (150 A^2 at 200 Da,
scaled as mass^(2/3); the order of magnitude of measured small-molecule cross-sections in nitrogen, caffeine 194 Da
about 140 A^2 [picache_2019]). Illustration; not used by the signal model. Feeding a small-fragment cell loss into the
signal (Daminozide 161.1 > 44.2 is a quantifier; Acephate 184 > 49 a qualifier) would need the real cell RF.

## Flower matrix ion map (first draft, not wired into the signal)

529 compounds reported in Cannabis sativa (LOTUS via Wikidata, scripts/fetch_cannabis_compounds.py), 499 with
PubChem XLogP3 and mass, placed in time by `lc.analogy` (the fit's leave-one-out error as the band). Ions:
[M+H]+, [M+Na]+, [M+NH4]+ and 13C M+1, M+2. Abundance tier per class is a guess (cannabinoids 1, terpenoids 0.1,
others 0.01, hydrocarbons 0.001). The placement and scoring scripts were written against a lab's own method and
matrix blanks; they, their inputs and their results are not part of this repository.

### Q3 filter from library spectra (scripts/fetch_matrix_spectra.py)

MassBank Europe, positive MS2, matched on the InChIKey skeleton: 125 of 425 flower skeletons have spectra
(2147 records), but only 2 of 102 cannabinoids (both THC; MassBank has no CBD, CBN or CBG). Compounds without a
spectrum would use a library prior (share of intensity at the product m/z across all spectra with a heavier
precursor); Na adducts are taken not to fragment. None of the 30 THC [M+H]+ spectra has any peak at 197.4-198.4
(THC's fragments: 193, 123, 91, 77...).

GNPS core library (GNPS-LIBRARY.json, 16,177 spectra; scripts/extract_gnps_spectra.py streams it): 419 positive
MS2 spectra for 68 flower skeletons, adding cannabinol and D8-THC (still no CBD, CBG or the acids). Cannabinol and
D8-THC (one spectrum each) have one peak at 197.4-198.4, at 0.9% of the base peak.

Open: a check against matrix-blank traces (baseline level per transition) or a precursor-ion scan from public
data, before the map can feed the signal.
