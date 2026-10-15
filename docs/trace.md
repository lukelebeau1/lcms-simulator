# The trace (`src/lcms_sim/trace.py`)

One portion of one compound followed from its reference standard to the reported area, as a list of stops.
Each stop holds the amount present, nominal (what the procedure and method say) and actual (what happened),
the share of the previous stop that reached it, where the rest went, and the provenance of that share. The
purpose is accounting: a loss of signal against expectation is charged to the stop where it happened.

## Stops and stations

Stations, in the order the stops come: storage, bench, injection, column, source, analyser, detector.

| Stop | Unit | Share | Basis |
|---|---|---|---|
| Ampoule (stock step) | ng | 1 (actual / nominal of the amount) | recipe |
| Transfer, solvent, sample, extract, inject steps on the path | ng | amount here / amount at the previous stop | recipe (ledger) |
| Carried over from the previous injection | ng | adds carryover fraction x its amount (actual only) | run conditions |
| Eluted inside the acquisition window | ng | Gaussian share of the peak inside the transition's acquisition window | model: retention and width at the conditions |
| Ionised in the spray and drawn into the inlet | ions | spray_fraction (below) | assumed |
| Not suppressed by co-eluting matrix | ions | matrix factor at the retention time, scaled by the run conditions for samples | model |
| Instrument sensitivity at injection k | ions | 1 + drift x k (actual only) | run conditions |
| Q1, collision cell, Q3, multiplier | ions | typical shares 0.3, 0.3 (x relative response for a qualifier), 0.3, 0.8 | assumed |
| Registered within the dead time | ions | non-paralysable detector integrated over the Gaussian peak (actual, when set) | model |
| Counted during the dwell | ions | dwell / cycle in the cycle containing the apex, 0 if not scheduled | model: dMRM schedule |
| Interference counted with the quantifier | ions | adds the interference's ions (samples, quantifier, actual only) | run conditions |
| Area reported | area units | x cycle / dwell, x gain | gain is a placeholder |

Units change where the thing changes: ng while it is a quantity of substance, ions from the spray on, area
units at the end. `Trace.molecules` and `Trace.ng` convert between them through the molecular weight.

## The bench path

The recipe is run twice (quant.ledger, nominal and actual, same steps). The path is walked back from the inject
step: in each vessel the entry step is the one that brought in most of the compound (a stock, a weighed sample,
or a transfer from a vessel holding it), and every step that touched the vessel from that entry until the
portion left is on the path. A solvent addition is therefore a stop that moves no amount, and a sample tube
spiked on top of a native amount shows the smaller route as a step whose share is above 1. A short draw of one
component also leaves the stock with less volume, so every later draw from it moves a slightly larger share of
everything else: the trace shows that as a small gain at the first transfer out of the stock (tested).

## Spray share

Nothing measured separates the spray, inlet, quadrupoles and multiplier. The response factor (area per ng) is
the only constraint on the product, so the spray share is what is left after dividing the response factor by
the gain, by molecules per ng, and by the typical downstream shares. It is assumed, like the shares it rests
on, and the trace labels it so. Replicate areas at low level fix the gain (counting scatter), a lab's area export
fixes the response factor; the split stays assumed until a tune report gives transmission figures.

## Invariant and oracle

The product of every stop's `loss` (actual / nominal of its share; for the first stop actual / nominal of the
amount) equals the last stop's actual / nominal, by construction and tested. The nominal area equals the
pipeline's expected-detector area for the same injection and transition (quantifier and qualifier tested), so
the trace is a view of the pipeline, not a second model. The dead-time stop is the one place the actual trace
goes beyond the expected detector: the pipeline applies dead time only in the simulated detector.

## Keys and share overrides

Every stop has a `key`: "step:<k>" for bench stops, else the stage (carryover, window, spray, matrix, drift,
q1, cell, q3, multiplier, dead_time, dwell, interference, area). The Follow page groups stops by it.
`Actual.shares` overrides the actual share of the spray, Q1, cell, Q3 or multiplier stage while the nominal
keeps the typical value, so the difference is charged there. `factors.apply` builds an `Actual` from the
Follow page's settings (docs/factors.md).

## Not covered yet

Crosstalk from other compounds; the internal standard's own trace and the ratio; the curve and the reported
concentration; a homogeneity state for vessels (an unmixed component drawn from the top or bottom). Faults
that change a Scenario's prep or calibration inputs are not yet mirrored into the recipe: pass an edited
recipe (quant.ledger.with_actual) for bench faults until the ledger feeds the run.
