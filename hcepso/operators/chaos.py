"""Chaotic local search via Logistic Map (μ=4)."""

import numpy as np
from typing import Callable
from .. import config


def logistic_map(x: float, mu: float = config.MU) -> float:
    return mu * x * (1 - x)


def chaotic_search(
    X: np.ndarray,
    fitness_fn: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    n_steps: int = config.CHAOS_STEPS,
    mu: float = config.MU,
) -> np.ndarray:
    """
    Generate a chaotic sequence around X using the Logistic Map.
    Maps the chaotic variable into the neighborhood of X and returns
    the best solution found.
    """
    n = len(X)
    best_X = X.copy()
    best_f = fitness_fn(X)

    # Initialize chaotic variable in (0, 1) — avoid 0, 0.25, 0.5, 0.75, 1
    z = rng.uniform(0.01, 0.99)
    while z in (0.25, 0.5, 0.75):
        z = rng.uniform(0.01, 0.99)

    # Scale range: ±amplitude around each dimension
    amplitude = np.abs(X).mean() * 0.5 + 1e-3

    for _ in range(n_steps):
        z = logistic_map(z, mu)
        # Map z ∈ (0,1) → perturbation ∈ (-amplitude, +amplitude)
        perturbation = (2 * z - 1) * amplitude
        candidate = X + perturbation
        candidate[candidate == 0] = 1e-6

        f = fitness_fn(candidate)
        if f < best_f:
            best_f = f
            best_X = candidate.copy()

    return best_X
