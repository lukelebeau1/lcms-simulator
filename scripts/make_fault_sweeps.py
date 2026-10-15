"""Build the symptom matcher's sweep table: every library fault at every sweep magnitude, the feature vector
averaged over the sweep seeds with the noisy detector, plus the baseline's mean and replicate spread.

    .venv/Scripts/python scripts/make_fault_sweeps.py          -> data/examples/fault_sweeps.csv

Takes a few minutes (faults x magnitudes x seeds simulations). Rerun after any change to the fault library,
the feature set, the example data or the pipeline; tests/test_matcher.py checks the table is current.
Prints the confusion table at the end (docs/faults.md carries a copy).
"""

import sys
import time

from lcms_sim.faults.matcher import SWEEP_FILE, confusion, default_checks, sweep_table
from lcms_sim.io.loaders import load_input_set
from lcms_sim.paths import EXAMPLES


def main() -> None:
    inputs = load_input_set(EXAMPLES)
    t0 = time.time()

    def progress(key: str, m: float) -> None:
        print(f"  {key} @ {m:g}  ({time.time() - t0:.0f} s)", flush=True)

    table = sweep_table(inputs, progress=progress)
    table.to_csv(SWEEP_FILE, index=False)
    print(f"wrote {SWEEP_FILE} ({len(table)} rows) in {time.time() - t0:.0f} s")
    conf = confusion(table, default_checks())
    print("\nConfusion table (each fault near its default magnitude, all features supplied):\n")
    print(conf.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
