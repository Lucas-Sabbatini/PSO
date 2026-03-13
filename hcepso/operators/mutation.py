"""Mutation operators: swap and sign-flip."""

import numpy as np


def mutation_swap(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Swap two randomly selected positions."""
    X = X.copy()
    i, j = rng.choice(len(X), size=2, replace=False)
    X[i], X[j] = X[j], X[i]
    return X


def mutation_flip_sign(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flip sign of a random position (toggles 90° rotation)."""
    X = X.copy()
    i = rng.integers(0, len(X))
    X[i] = -X[i]
    return X
