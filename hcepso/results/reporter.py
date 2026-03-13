"""Collect, aggregate, and print results."""

import csv
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class RunResult:
    dataset: str
    beta_idx: int
    run: int
    fitness: float
    n_bins: int


def aggregate(results: List[RunResult], dataset: str, beta_idx: int):
    runs = [r for r in results if r.dataset == dataset and r.beta_idx == beta_idx]
    if not runs:
        return None
    fitnesses = [r.fitness for r in runs]
    bins_list = [r.n_bins for r in runs]
    return {
        "dataset": dataset,
        "beta_idx": beta_idx,
        "best_fitness": min(fitnesses),
        "avg_fitness": float(np.mean(fitnesses)),
        "std_fitness": float(np.std(fitnesses)),
        "avg_bins": float(np.mean(bins_list)),
        "best_bins": min(bins_list),
    }


def print_table(results: List[RunResult]):
    datasets = sorted(set(r.dataset for r in results))
    beta_idxs = sorted(set(r.beta_idx for r in results))

    header = f"{'Dataset':>12} {'β-idx':>6} {'Best F':>10} {'Avg F':>10} {'Std F':>10} {'Avg Bins':>9} {'Best Bins':>9}"
    print(header)
    print("-" * len(header))

    for ds in datasets:
        for bi in beta_idxs:
            agg = aggregate(results, ds, bi)
            if agg:
                print(
                    f"{agg['dataset']:>12} {agg['beta_idx']:>6} "
                    f"{agg['best_fitness']:>10.2f} {agg['avg_fitness']:>10.2f} "
                    f"{agg['std_fitness']:>10.2f} {agg['avg_bins']:>9.1f} "
                    f"{agg['best_bins']:>9}"
                )


def save_csv(results: List[RunResult], filepath: str):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "beta_idx", "run", "fitness", "n_bins"])
        for r in results:
            writer.writerow([r.dataset, r.beta_idx, r.run, r.fitness, r.n_bins])
    print(f"Results saved to {filepath}")
