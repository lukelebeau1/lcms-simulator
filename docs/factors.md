# Loss factors and the Follow page (`src/lcms_sim/factors.py`, `app/views/follow.py`)

The simulator's main page is the chain of containers the followed compound sits in, from its reference
standard to the ions counted. Each card is one container (or one instrument stage treated as a container):
a cartoon (`app/cartoons.py`, illustration), the amount inside in ng and molecules (ions past the spray), and
the share of the previous container's amount that got here, with the factors that move that share off 100%.
Each card lists its stops as two sides: what makes it through (the stop and its share, bold) and what is lost and where it went (the complement: 70% neutralised on the rods), with the actual figures in red when a factor is set; a stop that passes everything says nothing is lost, a gain says what was added, and the area stop is a scale, not a share. Clicking into a card's expander gives one tab per factor: its control, its theory, its provenance, and links
to the pages that go deeper. A sticky bar at the top carries the totals (share of the ampoule's amount
reaching the counted ions by design, and actual against expected), and every card states the running
tally so far. The injection card is named "Injection: <item>" although the recipe's vessel is "Column: <item>":
it is the 10 uL drawn from the vial; the Column card is where the window share is counted.

## A container

A recipe vessel plus what it physically is (`data/examples/containers.yaml`: kind, location, solvent, capacity,
supplier, provenance). Instrument stages are fixed in `factors.STAGES`: injection loop, column, spray,
capillary, Q1, collision cell, Q3, multiplier, counting. The Follow page groups the trace's stops by vessel for
the bench and by stage key for the instrument (`Stop.key`).

## A factor

One object everywhere: key, name, where it acts, kind (check, number, or the step's actual volume), default,
range, theory, provenance, and `sources`: keys into `data/reference/references.yaml` for the literature behind the
theory (docs/references.md lists every reference, its verification and every citing site; a factor with no source
stays labelled assumed). Bench factors act per transfer or injection step; their setting key carries the step
index ("12:unmixed"). A transfer's efficiency is the product of its factor multipliers (unmixed: the share of
the nominal concentration at the tip, below or above 1; precipitated: 1 minus the share still out of
solution; adsorbed: 1 minus the percent lost); the pipette factor is the step's actual volume. `RecipeStep.
efficiency` carries it into the ledger, which moves that share of the nominal amount and leaves the rest in
the source, so amounts stay conserved and an unmixed draw shows as ng that never left the vial.

Stage factors in this slice are the ones the model already computes or the trace already carries: carryover
(loop); dwell volume, %B offset, pH offset and plate number (column, through the fault library's LC faults and
the window share); spray share multiplier, matrix suppression scale and drift (spray); Q1, cell, Q3 and
multiplier shares (absolute, against the typical values); dead time (multiplier); extra concurrent transitions
(counting, through the dMRM schedule). The capillary has no factor yet: the spray share includes the inlet.

`factors.apply(inputs, recipe, settings, compound, injection_index, previous_injection)` turns a settings
mapping into the trace's `Actual`: an edited recipe (volumes and efficiencies), a fault-library Scenario for
the column and acquisition changes, run conditions, and share overrides. `common.current_settings` reads the
Follow page's widgets into that mapping, so every page's rail shows the same actual run.

## Scope: analyte-independent or analyte-dependent

The troubleshooter's first split, and the fault lab's first axis. Every factor (and every fault in
`faults/library.py`) carries `scope`:

- `uniform`: the same share for every compound. A volume, a stock, a drift, a multiplier setting. The result
  looks like a dilution or a gain change, every compound moves together, and an internal standard that was
  present cancels it in the ratio.
- `compound`: the share depends on the compound, and `varies_with` names what it follows: solubility
  (precipitated), surface affinity (adsorbed), the component's own solvent (unmixed), pKa (pH), S (%B),
  retention time (dwell volume, matrix suppression), m/z (Q1, Q3), the breakdown curve (cell), ion rate (dead
  time), the previous injection's level (carryover). Some compounds move and others do not, and the pattern
  across compounds is the diagnosis.

The classification is of the effect on the per-compound result, not of the cause. A dwell volume change
shifts every retention time by the same amount, but the area loss lands on early eluters and narrow windows
first, so it is compound-scoped. A plate loss broadens every peak by the same factor, so it is uniform, with
the window clipping noted in its theory.

Compound-scoped bench factors are set per compound: the setting key is `"<step>:<factor>@<compound>"`
(`bench_key(step, factor, compound)`), and `bench_efficiencies` turns the settings into the ledger's two
efficiencies, the step's uniform share (`RecipeStep.efficiency`, every compound) and the per-compound shares
(`RecipeStep.efficiency_by_compound`), which multiply. The ledger charges a compound's own share to that
compound only; every other compound in the same draw is untouched, in the destination and in the source. A
compound-scoped key with no compound named is charged to the compound being followed. Stage factors are keyed
without a compound: they are conditions of the instrument, and the model derives each compound's loss from them
(the matrix scale is applied to every analyte and the infusion profile sets each one's suppression).

### Whether the internal standard sees it

The third thing each factor card states. `istd_note(factor, istd, present, own_label)`: with no internal
standard the loss goes straight to the result; where the IS is not yet in the vessel drawn from, the same; where
it is present, a uniform loss cancels in the ratio, and a compound-scoped loss cancels only when the IS is the
compound's own labelled analogue (`is_own_label`, by name: Myclobutanil-D4 for Myclobutanil) and so shares its
chemistry; a surrogate IS does not. The Follow page finds whether the IS is present from the nominal ledger:
in the source vessel at a bench step (the example adds the IS in the vial), always on the instrument.

The Follow page marks compound-specific factors with an asterisk in each card's factor list and states the
scope and the IS note at the top of every factor tab.

## What the tests pin down

A factor set on one transfer is charged to that stop only and the rest of the stock is untouched; what the tip
did not pick up stays in the source vial (conservation). A pipette setting halves the amount moved, for every
compound. A compound-scoped factor on a draw from a mix changes that compound alone: every other compound's
amount in the destination and the source equals the nominal run, and the setting stays with its compound when
another one is followed. Every factor has a scope, and only a compound-scoped one says what it varies with.
Every uniform fault in the library leaves the sample bias spread across analytes at zero (noise-free detector).
Stage factors land on their own stage, and the product of the losses is the final ratio. Column factors change
only the window share. Extra concurrent transitions count fewer ions but report the same area. Default settings
change nothing. Every recipe vessel resolves to a container.

## Not yet

Class rules as defaults (which compounds precipitate, adsorb or layer: the scope and `varies_with` name the
property, the rule from XLogP3, pKa and class is not written), a storage card with time, temperature and
mixed-standard ageing, the sandwich injection, capillary transmission, on-column adsorption and breakdown, and
column temperature. The fault lab's features correlate bias with retention time only; log P, the ionisable
class and m/z are the next correlates, so the matcher can tell adsorption from suppression from a pH error.
The MS probe pages still stand on their own; the plan is to fold their visuals into the cards' expanders.
