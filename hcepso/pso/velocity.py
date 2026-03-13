"""PSO velocity and position update equations."""

import numpy as np


def update_velocity(
    V: np.ndarray,
    X: np.ndarray,
    pBest: np.ndarray,
    gBest: np.ndarray,
    omega: float,
    c1: float,
    c2: float,
    rng: np.random.Generator,
) -> np.ndarray:
    r1 = rng.random(len(X))
    r2 = rng.random(len(X))
    return omega * V + c1 * r1 * (pBest - X) + c2 * r2 * (gBest - X)


def update_position(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    new_X = X + V
    # Prevent zeros (undefined sign → rotation flag)
    new_X[new_X == 0] = 1e-6
    return new_X


def decay_omega(k: int, k_max: int, omega_ini: float, omega_fin: float) -> float:
    return omega_ini - (omega_ini - omega_fin) / k_max * k
