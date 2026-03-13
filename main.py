"""
Entry point: run HCEPSO experiments on the BPPC dataset.

Usage:
    python main.py [--dataset-dir DIR] [--dataset-id ID] [--beta-idx IDX]
                   [--runs N] [--iters N] [--verbose] [--output FILE]
"""

import argparse
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from hcepso.instance.loader import load_instance
from hcepso.pso.swarm import run
from hcepso.results.reporter import RunResult, print_table, save_csv
from hcepso import config


DATASET_DIR = Path(__file__).parent / "BPPC" / "Istanze"


def parse_args():
    p = argparse.ArgumentParser(description="HCEPSO for 2D-BPPCL")
    p.add_argument("--dataset-dir", default=str(DATASET_DIR))
    p.add_argument("--dataset-id", type=int, default=1,
                   help="Dataset index (1–8); 1 = Bengtsson 120 items")
    p.add_argument("--beta-idx", type=int, default=None,
                   help="Conflict density index (0–9). None = all.")
    p.add_argument("--runs", type=int, default=config.N_RUNS)
    p.add_argument("--iters", type=int, default=config.MAX_ITER)
    p.add_argument("--pop", type=int, default=config.POPULATION)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", default=None,
                   help="CSV output path (optional)")
    return p.parse_args()


def run_experiment(args) -> list[RunResult]:
    dataset_dir = Path(args.dataset_dir)
    ds_id = args.dataset_id
    beta_range = range(10) if args.beta_idx is None else [args.beta_idx]

    results: list[RunResult] = []

    for beta_idx in beta_range:
        beta_results = []
        for run_id in range(1, args.runs + 1):
            fname = f"BPPC_{ds_id}_{beta_idx}_{run_id}.txt"
            fpath = dataset_dir / fname
            if not fpath.exists():
                print(f"  [skip] {fname} not found")
                continue

            instance = load_instance(str(fpath), seed=run_id)

            if args.verbose:
                print(f"\nDataset {ds_id}, β-idx={beta_idx}, run {run_id}/{args.runs}")

            sol = run(
                instance,
                population=args.pop,
                max_iter=args.iters,
                seed=run_id * 1000 + beta_idx,
                verbose=args.verbose,
            )

            r = RunResult(
                dataset=str(ds_id),
                beta_idx=beta_idx,
                run=run_id,
                fitness=sol.fitness,
                n_bins=sol.n_bins,
            )
            results.append(r)
            beta_results.append(r)

            print(
                f"  DS={ds_id} β-idx={beta_idx} run={run_id:2d} | "
                f"fitness={sol.fitness:.2f}  bins={sol.n_bins}"
            )

    return results


def main():
    args = parse_args()
    print(f"Running HCEPSO | dataset={args.dataset_id} | runs={args.runs} | iters={args.iters}")
    print()

    results = run_experiment(args)

    if results:
        print("\n=== Summary ===")
        print_table(results)

        if args.output:
            save_csv(results, args.output)


if __name__ == "__main__":
    main()
