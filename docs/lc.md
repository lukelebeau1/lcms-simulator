# LC model

## Equations

| Quantity | Equation | Code |
|---|---|---|
| Retention factor | log10 k = log10 kw - S phi [snyder_dolan_2007] | `retention.k_lss` |
| Inlet composition | phi_inlet(t) = phi_programmed(t - tD), tD = VD / F [snyder_dolan_2007, dolan_2006_dwell] | `gradient.LCConditions.phi_inlet` |
| Retention time | integral_0^(tR - t0) dt / (t0 k(phi_inlet(t))) = 1 [snyder_dolan_2007] | `retention.retention_time` |
| Width | sigma_t = G t0 (1 + ke) / sqrt(N), ke = k(phi_inlet(tR - t0)); extra-column sigma in quadrature [snyder_kirkland_dolan_2010, snyder_dolan_2007] | `peaks.peak_sigma` |
| pH (optional) | acid: k = (k_HA + k_A 10^(pH-pKa)) / (1 + 10^(pH-pKa)); base mirrored [horvath_1977, snyder_kirkland_dolan_2010] | `retention.k_ph` |
| Shape | Gaussian, or exponentially modified Gaussian (mean = centre + tau) | `peaks.gaussian`, `peaks.emg` |

## Solver

The inlet composition is piecewise linear, so on each segment the integrand
1 / (t0 k) = 10^(S phi0 - log kw) / t0 x 10^(S m (t - ts)) is a single exponential with an
exact antiderivative. `retention_time` sums segments and inverts inside the crossing segment, with log1p: a
hold whose slope comes out as 1e-16 in floating point made log(1 + u) quantise tR in steps of about 0.2 min
(found on the application note's 30% B hold; tested).
This is exact for multi-segment gradients, holds and steps. It differs from the plan's
"cumulative integration and root finding" only in having no grid error.

`retention_time_numeric` is the plan's grid method and works for any k(phi), including the
pH extension. Tests check that the two agree within 0.01% on the app note gradient. The
exact solver also integrates back to 1 under adaptive quadrature, to 1e-8.

## Fitting

- `fit_lss`: log kw and S from two or more gradients, minimising relative residuals.
  Residuals are reported, and a fit is flagged when any residual exceeds 2%.
- `fit_log_kw`: log kw from one retention time with S fixed. It is used for literature
  retention times, where only one gradient is known.
- `estimate_S`: S ~ 0.25 MW^0.5 [snyder_dolan_2007]. An assumption.

## Inside the column (`column.py`)

A band at column fraction x moves at dx/dt = 1 / (t0 (1 + k)), with k at the composition around
it, phi(x, t) = phi_inlet(t - x t0). With tau = t - x t0 (the inlet time of the mobile phase the
band sits in), dx/dtau = 1 / (t0 k(phi_inlet(tau))), so x(tau) is the retention integral read
part way and t = tau + x t0. `band_position` inverts that with the same exact per-segment
antiderivative as `retention_time`, so a band reaches the outlet exactly at tR (tested), and
under isocratic conditions x = t / (t0 (1 + k)) (tested). Band width along the column is
sigma_x = sqrt(x / N) from the plate height; gradient compression and extra-column broadening
are left out. `band_trajectory` gives the same positions for a whole time grid at once (exact x(tau) on a fine
inlet-time grid, interpolated; matches `band_position` to 1e-4, tested). The view "LC: inside the
column" uses it to play the run with Play / Pause (nothing advances until Play is pressed). The
sidebar "Jump to time" slider shows any moment; Play runs on from there. The frames are carried in
the Play button rather than registered on the figure, because Streamlit's fullscreen redraw drops
registered frames (the layout survives).

## Assumptions

- The LSS model is linear in phi. The error grows away from the fitted conditions.
- Gradient and flow are the application note's [agilent_5994_1734en] (A 5 mM ammonium formate / 0.1% formic
  acid in water, B 0.1% formic acid in methanol; 30% B held to 1 min, 75% at 2, 96% at 8, 100% from 9 to 9.5,
  back to 30% at 9.51; 0.5 mL/min, 13 min). log kw is fitted to the note's retention times on that gradient,
  with S from molecular weight. One gradient cannot separate log kw from S, so predictions for other gradients
  carry the S assumption. A lab input set fits its own the same way (scripts/make_example_data.py).
- t0 comes from column geometry (100 x 3.0 mm) with total porosity 0.5: 0.71 min at 0.5 mL/min.
  Daminozide, the first peak, elutes at 1.10 min, consistent. Dwell volume is assumed to be 0.6 mL
  (tD 1.2 min), so the first group elutes during the 30% B hold and the dwell delay.
- Temperature, column overload and injection-solvent effects are not modelled.
- The Snyder G expression is unverified and off by default; calling it raises a warning.

## Retention by analogy (`analogy.py`), for matrix compounds only

Provenance: assumed. Places compounds with no measured RT (the flower matrix) by analogy to the method's
measured pesticide RTs (the example's, from the application note: 60 neutral compounds). Analytes keep their
measured RTs.

    tR = t_lo + (t_hi - t_lo) / (1 + exp(-(x - x0) / w)),   x = XLogP3 + b log10(mass / 300)

XLogP3 from PubChem (scripts/fetch_xlogp.py -> data/reference/pesticide_xlogp.csv). The upper plateau is capped
at the time the end of the 100% B hold reaches the column outlet (9.5 + tD + t0 = 11.4 min); the pesticides
alone do not reach it and an uncapped fit runs away. Fit: leave-one-out RMS error 0.58 min, median 0.34 min on
the 13 min run (tested); this is the band given to every placement. Bases and the acid (ochratoxin) are
excluded from the fit.

Check against literature (not fitted): the main cannabinoids land late, as expected, but their order within the
class is not reproduced (rank correlation ~0.06 against the published acidic C18 order CBDV, CBDA, CBGA, CBG,
CBD, THCV, CBN, THC, D8-THC, CBC, THCA): the class spans less XLogP3 than the fit's error,
and acids partly charged at pH 3 are treated as neutral. Use it for "what is in this 2 min region", not for
order within a class.

## Molecules in 3D (`app/molecules.py`, "Why it's retained" on LC: inside the column)

Illustration, not a model. Structures for all 66 targets and 14 internal standards (scripts/fetch_structures.py ->
data/reference/structures/): PubChem 3D conformers, or RDKit ETKDG + MMFF where PubChem has none (abamectin,
spinosyns, azadirachtin, ivermectin-D2, imazalil-D5). Counter-ions dropped (PubChem's imazalil-D5 is a sulfate
salt). Myclobutanil-D4 is not in PubChem: drawn as myclobutanil, deuterium positions unknown.

Colour: Wildman-Crippen atomic log P contributions (RDKit) [wildman_crippen_1999], which sum to the Crippen log P (tested). Grouped by
functional group (SMARTS, first match wins; every atom in one group, tested): positive shares hold the molecule
on C18, negative shares pull it into the mobile phase (a reading of the partitioning picture [dorsey_dill_1989,
rafferty_2007]). Crippen log P differs from the XLogP3 [cheng_2007] used for retention by analogy; both are shown.

Ion: the precursor type is read from the quantifier (precursor m/z minus the neutral's mass): [M+H]+, [M+NH4]+,
[M+Na]+, [M]+ (chlormequat) or [M-H]- (negative mode: fludioxonil, fipronil). All 80 identified (tested). Five
methods select the M+2 isotope peak of Cl/Br-rich molecules (chlorantraniliprole, chlorfenapyr, cypermethrin,
cyfluthrin, naled, and fipronil in negative mode). The proton or adduct is placed at the most likely site by a
ranking (basic N, azole N, pyridine N, carbonyl or P=O/P=S O, ether O): a sketch of where the charge sits, not a
computed gas-phase structure. Viewer: py3Dmol (3Dmol.js loaded from its CDN, so the browser needs internet), on
the app's dark background, wheel zoom slowed to ~3-5% per notch (3Dmol's own is ~20%), every atom other than C and H
labelled by default, shares within +-0.12 of zero drawn neutral (most C and H carry a small positive share). Beside it
the conventional 2D skeletal diagram (RDKit, dark style).

## Down at the C18 surface (`app/c18_scene.py`, LC: inside the column)

Illustration driven by the retention model. One analyte molecule (its 3D structure, coloured by Crippen share)
among bonded C18 chains on silica, with water and methanol. Real: the share of time held in the C18 layer,
k/(1+k) [snyder_kirkland_dolan_2010, dorsey_dill_1989], with k = k_lss(phi) at the %B around the band (column.band_trajectory, phi_in_column), and the methanol
mole fraction for that %B (molar volumes 40.7 and 18.05 mL/mol [pubchem_methanol], no mixing contraction). Drawn: a disordered
brush (~1 chain per 50 A^2, tilt ~25 deg, occasional gauche kinks: values inside the simulated ranges, which depend
on the grafting density [lippa_2005, rafferty_2008]), the hop timing (two-state Markov chain
with a fixed cycle: mean free stretch max(1, 10/(1+k)) frames, mean held stretch k times that, so frequent
short holds at moderate k and brief releases at high k, held share k/(1+k) either way), poses, flow speed, and
methanol (not water) soaking into the chain tops [rafferty_2011]. Held, the molecule is partitioned into the chain layer along
its polar-to-oily axis [rafferty_2007] (Crippen-weighted centroids of negative and positive shares): oily end among the chains,
polar end at the chain tops. An earlier version pointed the single most hydrophobic spot at the chains, which
let one aryl Cl set the pose. Eight copies can be shown: at any frame about k/(1+k) of them are held, the time
average seen as an average over molecules. Time is compressed: real sorption events are far
shorter than a frame (an order-of-magnitude statement with no measurement behind it: listed as assumed in
docs/references.md). Two modes: through the run (each frame a representative moment, band position and local
%B from the model; the molecule disappears once the band elutes) or one moment (fixed %B). Tests: held share =
k/(1+k), frequent holds at moderate k, copies average to k/(1+k), methanol count = mole fraction, polar
end up and inside the layer when held, no analyte after elution.

Two compounds in one window (build_multi): the window follows the first compound's band and shows the solvent
there; each compound holds and hops by its own k at its own band. The second keeps round(copies x overlap) of
its molecules in the window, overlap = exp(-d^2 / 2(sA^2 + sB^2)) from the two band centres (d) and widths
(plate broadening plus a floor at the injected plug: 10 uL over the open cross-section, porosity 0.50 from t0,
2.8 mm, no on-column focusing assumed; without the floor the model gives a zero-width band at the head and any
gap reads as separation). Leavers are carried out downstream (band ahead) or left upstream (band behind). With
myclobutanil followed: acephate leaves at once, carbaryl at 2.2 min, metalaxyl and tebuconazole ~3.6 min,
paclobutrazol only at 6.5 min (they elute 0.05 min apart).

Camera (c18_scene.html): 3Dmol's own mouse handling is off (nomouse) and the page sets the view through
setView, whose array is [minus the view centre, camera z, quaternion taking model axes to screen axes]. The
view rotates and zooms but never pans. Rotation modes: turntable (drag sideways spins about the surface normal
through the view centre, up and down tilts the camera between 5 and 88 degrees above the surface, so the surface
stays down), spin only (one plane), free (trackball; switching back re-levels to the nearest turntable pose).
Zoom: the wheel within 0.7 to 4x of the fitted view (whole surface), or "close on <compound>": the camera
centres on that compound's first drawn molecule and follows it from frame to frame (40% of the gap per frame),
framing from the molecule's size (surface fit radius / (2.2 x farthest atom), clamped 1.5 to 6x), wheel off;
while no molecule of that compound is in the window (its band has moved away) the camera eases back to the
whole surface. Per frame the page sets the view state 3Dmol's setView writes and lets render draw once; setView
itself (which draws) serves drags, the wheel and the selects.
Scene carries per-frame centres and per-species extents for this (tested: the centre is the copy's centroid and
the extent bounds its atoms).

