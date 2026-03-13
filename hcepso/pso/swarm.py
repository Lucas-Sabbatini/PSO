"""Main HCEPSO optimization loop."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from ..instance.instance import Instance
from .. import config
from .particle import Particle
from .encoding import encode_random, decode
from .velocity import update_velocity, update_position, decay_omega
from ..packing.heuristic import pack_items
from ..packing.fitness import compute_fitness
from ..operators.crossover import elitist_crossover
from ..operators.mutation import mutation_swap, mutation_flip_sign
from ..operators.chaos import chaotic_search


@dataclass
class Solution:
    gBest: np.ndarray
    fitness: float
    n_bins: int
    history: List[float]                              # best fitness per iteration
    particle_snapshots: Optional[List[np.ndarray]] = field(default=None)  # (iter, pop, n)
    gbest_snapshots: Optional[List[np.ndarray]] = field(default=None)     # (iter, n)
    bin_snapshots: Optional[List] = field(default=None)                   # iter -> List[Bin]


def _make_fitness_fn(instance: Instance):
    """Return a callable that evaluates a particle position."""
    def fn(X: np.ndarray) -> float:
        sequence = decode(X, instance.items)
        bins = pack_items(sequence, instance.bin_W, instance.bin_H, instance.conflicts)
        return compute_fitness(
            bins, instance.bin_W, instance.bin_H,
            instance.bin_Q, instance.bin_delta,
            instance.conflicts,
        )
    return fn


def run(
    instance: Instance,
    population: int = config.POPULATION,
    max_iter: int = config.MAX_ITER,
    c1: float = config.C1,
    c2: float = config.C2,
    omega_ini: float = config.OMEGA_INI,
    omega_fin: float = config.OMEGA_FIN,
    p_mutation: float = config.P_MUTATION,
    seed: Optional[int] = None,
    verbose: bool = False,
    record: bool = False,
) -> Solution:
    rng = np.random.default_rng(seed)
    n = instance.n
    fitness_fn = _make_fitness_fn(instance)

    # 1. Initialize particles
    particles: List[Particle] = []
    for _ in range(population):
        X = encode_random(n, rng)
        V = rng.uniform(-1.0, 1.0, size=n)
        f = fitness_fn(X)
        particles.append(Particle(X=X, V=V, pBest=X.copy(), pBest_fitness=f))

    # 2. Global best
    gBest_idx = int(np.argmin([p.pBest_fitness for p in particles]))
    gBest = particles[gBest_idx].pBest.copy()
    gBest_fitness = particles[gBest_idx].pBest_fitness

    history = [gBest_fitness]

    part_snaps: List[np.ndarray] = []
    gbest_snaps: List[np.ndarray] = []
    bin_snaps: List = []

    def _snapshot():
        part_snaps.append(np.vstack([p.X for p in particles]))
        gbest_snaps.append(gBest.copy())
        seq = decode(gBest, instance.items)
        bin_snaps.append(pack_items(seq, instance.bin_W, instance.bin_H, instance.conflicts))

    if record:
        _snapshot()

    # 3. Main loop
    for k in range(1, max_iter + 1):
        omega = decay_omega(k, max_iter, omega_ini, omega_fin)

        for p in particles:
            # Velocity & position update
            p.V = update_velocity(p.V, p.X, p.pBest, gBest, omega, c1, c2, rng)
            p.X = update_position(p.X, p.V)

            # Elitist crossover: gBest × current particle
            offspring = elitist_crossover(gBest, p.X, rng)
            f_offspring = fitness_fn(offspring)
            if f_offspring < fitness_fn(p.X):
                p.X = offspring

            # Mutation
            if rng.random() < p_mutation:
                if rng.random() < 0.5:
                    p.X = mutation_swap(p.X, rng)
                else:
                    p.X = mutation_flip_sign(p.X, rng)

            # Chaotic local search
            p.X = chaotic_search(p.X, fitness_fn, rng)

            # Evaluate and update personal best
            f = fitness_fn(p.X)
            if f < p.pBest_fitness:
                p.pBest = p.X.copy()
                p.pBest_fitness = f

        # Update global best
        for p in particles:
            if p.pBest_fitness < gBest_fitness:
                gBest = p.pBest.copy()
                gBest_fitness = p.pBest_fitness

        history.append(gBest_fitness)
        if record:
            _snapshot()
        if verbose:
            particle_fitnesses = [p.pBest_fitness for p in particles]
            avg_f = float(np.mean(particle_fitnesses))
            worst_f = float(np.max(particle_fitnesses))
            print(
                f"  iter {k:4d}/{max_iter} | ω={omega:.4f} | "
                f"gBest={gBest_fitness:.2f}  avg={avg_f:.2f}  worst={worst_f:.2f}"
            )

    # Count bins in best solution
    sequence = decode(gBest, instance.items)
    bins = pack_items(sequence, instance.bin_W, instance.bin_H, instance.conflicts)
    n_bins = len(bins)

    return Solution(
        gBest=gBest,
        fitness=gBest_fitness,
        n_bins=n_bins,
        history=history,
        particle_snapshots=part_snaps if record else None,
        gbest_snapshots=gbest_snaps if record else None,
        bin_snapshots=bin_snaps if record else None,
    )
