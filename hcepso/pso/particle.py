"""Particle dataclass."""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Particle:
    X: np.ndarray           # current position
    V: np.ndarray           # current velocity
    pBest: np.ndarray       # personal best position
    pBest_fitness: float    # personal best fitness value
