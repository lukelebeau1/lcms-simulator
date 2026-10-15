"""The single source of randomness. Callers create one generator and pass it down.

The seed is always an explicit integer so the interface can show it and the user can fix it.
"""

from __future__ import annotations

import numpy as np


def new_seed() -> int:
    """A fresh seed for an unseeded run. Record it so the run can be reproduced."""
    return int(np.random.SeedSequence().entropy % (2**32))


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
