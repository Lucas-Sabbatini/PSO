"""
Plot convergence curve(s) for HCEPSO runs.

Usage:
    python plot/convergence.py [--dataset-id ID] [--beta-idx IDX]
                               [--runs N] [--iters N] [--pop N]
                               [--output FILE]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from hcepso.instance.loader import load_instance
from hcepso.pso.swarm import run
from hcepso import config


DATASET_DIR = Path(__file__).parent.parent / "BPPC" / "Istanze"


def parse_args():
    p = argparse.ArgumentParser(description="Convergence plot for HCEPSO")
    p.add_argument("--dataset-dir", default=str(DATASET_DIR))
    p.add_argument("--dataset-id", type=int, default=1)
    p.add_argument("--beta-idx", type=int, default=0)
    p.add_argument("--runs", type=int, default=config.N_RUNS)
    p.add_argument("--iters", type=int, default=config.MAX_ITER)
    p.add_argument("--pop", type=int, default=config.POPULATION)
    p.add_argument("--output", default=None,
                   help="Save plot to file instead of displaying it")
    return p.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    ds_id = args.dataset_id
    beta_idx = args.beta_idx

    histories = []

    for run_id in range(1, args.runs + 1):
        fname = f"BPPC_{ds_id}_{beta_idx}_{run_id}.txt"
        fpath = dataset_dir / fname
        if not fpath.exists():
            print(f"[skip] {fname} not found")
            continue

        print(f"Run {run_id}/{args.runs} ...", end=" ", flush=True)
        instance = load_instance(str(fpath), seed=run_id)
        sol = run(
            instance,
            population=args.pop,
            max_iter=args.iters,
            seed=run_id * 1000 + beta_idx,
        )
        histories.append(sol.history)
        print(f"fitness={sol.fitness:.2f}")

    if not histories:
        print("No results to plot.")
        return

    iters = np.arange(len(histories[0]))

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, h in enumerate(histories):
        ax.plot(iters, h, alpha=0.35, linewidth=1, label=f"Run {i + 1}")

    if len(histories) > 1:
        mean_h = np.mean(histories, axis=0)
        ax.plot(iters, mean_h, color="black", linewidth=2, label="Mean")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    ax.set_title(f"Convergence – Dataset {ds_id}, β-idx {beta_idx}")
    ax.legend(fontsize="small", ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Plot saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
