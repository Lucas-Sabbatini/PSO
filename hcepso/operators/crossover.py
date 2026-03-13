"""Elitist 2-point crossover operator."""

import numpy as np


def elitist_crossover(
    pBest: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    2-point crossover between pBest (elite parent) and parent2.

    The segment [c1:c2] is taken from pBest.
    Remaining positions are filled with values from parent2 in order,
    skipping indices whose |value| rank is already in the offspring.

    Works on real-valued arrays; preserves sign (rotation flag).
    """
    n = len(pBest)
    c1, c2 = sorted(rng.integers(0, n, size=2))
    if c1 == c2:
        c2 = min(c2 + 1, n)

    offspring = np.zeros(n)
    offspring[c1:c2] = pBest[c1:c2]

    # Track which positions (by index) are already filled
    filled_indices = set(range(c1, c2))

    # Fill remaining positions with parent2 values in their original order
    parent2_iter = (v for i, v in enumerate(parent2) if i not in filled_indices)
    for i in range(n):
        if i not in filled_indices:
            offspring[i] = next(parent2_iter)

    offspring[offspring == 0] = 1e-6
    return offspring
