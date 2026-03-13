"""Parse Bengtsson-format .txt files and augment with synthetic 2D geometry."""

import numpy as np
from typing import Optional
from .item import Item
from .instance import Instance
from .. import config


def load_instance(
    filepath: str,
    bin_W: float = config.BIN_W,
    bin_H: float = config.BIN_H,
    bin_delta: Optional[float] = None,
    seed: int = 42,
) -> Instance:
    """
    Parse a BPPC dataset file and return an Instance.

    File format:
        N   C
        i   w   [a1  a2  ...  ak]
        ...

    Synthetic 2D dimensions (w_i, h_i) are generated from a fixed seed.
    """
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]

    header = lines[0].split()
    N = int(header[0])
    C = int(header[1])   # bin weight capacity

    weights = []
    conflict_sets = []

    for line in lines[1:N + 1]:
        parts = list(map(int, line.split()))
        # parts[0] = item id (1-indexed), parts[1] = weight, parts[2:] = conflicts
        weights.append(parts[1])
        conflict_sets.append(set(parts[2:]))

    # Synthetic geometry: uniform [10, 50]
    rng = np.random.default_rng(seed)
    widths  = rng.integers(10, 51, size=N).astype(float)
    heights = rng.integers(10, 51, size=N).astype(float)

    items = [
        Item(id=i + 1, width=widths[i], height=heights[i],
             weight=weights[i], can_rotate=True)
        for i in range(N)
    ]

    # Default max barycenter displacement: half of bin diagonal
    if bin_delta is None:
        bin_delta = 0.1 * min(bin_W, bin_H)

    return Instance(
        items=items,
        bin_W=bin_W,
        bin_H=bin_H,
        bin_Q=float(C),
        bin_delta=bin_delta,
        conflicts=conflict_sets,
    )
