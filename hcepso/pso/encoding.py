"""
Encoding/decoding between real-valued particle positions and packing sequences.

Particle position X is a real-valued array of length n.
Decoding:
  - rank |X| values to get a permutation (item order)
  - sign(X[i]) < 0 → item is rotated 90°
"""

import numpy as np
from typing import List, Tuple
from ..instance.item import Item


def encode_random(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Initialize a random particle position.
    Values are drawn from Uniform[-n, n], excluding 0.
    The permutation implied by |X| ranks is uniformly random.
    """
    X = rng.uniform(-n, n, size=n)
    # Avoid zeros (sign undefined)
    X[X == 0] = 1e-6
    return X


def decode(X: np.ndarray, items: List[Item]) -> List[Tuple[Item, bool]]:
    """
    Convert a real-valued particle position to an ordered packing sequence.

    Returns list of (item, rotated) in placement order.
    Rank by |X| ascending → determines item order.
    sign(X[i]) < 0 → item i is placed rotated.
    """
    n = len(X)
    abs_X = np.abs(X)
    order = np.argsort(abs_X)           # ascending rank of |X|
    sequence = []
    for idx in order:
        item = items[idx]
        rotated = X[idx] < 0
        sequence.append((item, rotated))
    return sequence
