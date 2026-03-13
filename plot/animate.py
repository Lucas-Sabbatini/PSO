"""
Two animations for a single HCEPSO run:
  1. Particle positions projected to 2D (PCA) converging over iterations.
  2. Best-known bin packing layout evolving over iterations.

Usage:
    python plot/animate.py [--dataset-id ID] [--beta-idx IDX] [--run RUN]
                           [--iters N] [--pop N] [--step N] [--interval MS]
                           [--output-particles FILE] [--output-bins FILE]

Outputs are saved as GIF when --output-* is provided, otherwise displayed.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import numpy as np

from hcepso.instance.loader import load_instance
from hcepso.pso.swarm import run
from hcepso import config


DATASET_DIR = Path(__file__).parent.parent / "BPPC" / "Istanze"


def parse_args():
    p = argparse.ArgumentParser(description="HCEPSO animations")
    p.add_argument("--dataset-dir", default=str(DATASET_DIR))
    p.add_argument("--dataset-id", type=int, default=1)
    p.add_argument("--beta-idx", type=int, default=0)
    p.add_argument("--run", type=int, default=1, help="Run ID (used as seed)")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--pop", type=int, default=config.POPULATION)
    p.add_argument("--step", type=int, default=1,
                   help="Show every N-th iteration frame")
    p.add_argument("--interval", type=int, default=300,
                   help="Milliseconds between frames")
    p.add_argument("--output-particles", default=None,
                   help="Save particle animation to this file (e.g. particles.gif)")
    p.add_argument("--output-bins", default=None,
                   help="Save bins animation to this file (e.g. bins.gif)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# PCA (via NumPy SVD) – no sklearn dependency
# ---------------------------------------------------------------------------

def pca2d(snapshots: list[np.ndarray]):
    """
    Project n-dimensional particle positions to 2D using PCA.

    snapshots: list of (population, n) arrays, one per recorded iteration.
    Returns:
        projections: list of (population, 2) arrays
        gbest_proj : not computed here – caller must handle separately
        explained  : variance explained by PC1, PC2 (tuple of floats)
    """
    # Fit PCA on the initial snapshot so that the initial spread defines the
    # principal directions – this guarantees particles are visibly apart in frame 0.
    init_snap = snapshots[0]              # (pop, n)
    mu = init_snap.mean(axis=0)
    _, S, Vt = np.linalg.svd(init_snap - mu, full_matrices=False)
    pc = Vt[:2]                           # (2, n)
    var = S ** 2
    explained = var[:2] / var.sum() * 100

    projections = [(snap - mu) @ pc.T for snap in snapshots]
    return projections, mu, pc, explained


# ---------------------------------------------------------------------------
# Animation 1: particles in PCA space
# ---------------------------------------------------------------------------

def make_particle_animation(sol, args):
    snapshots = sol.particle_snapshots
    gbest_snaps = sol.gbest_snapshots
    history = sol.history

    projections, mu, pc, explained = pca2d(snapshots)

    # Also project gBest positions
    gbest_proj = np.vstack([(g - mu) @ pc.T for g in gbest_snaps])  # (T, 2)

    frames = list(range(0, len(projections), args.step))

    all_pts = np.vstack(projections)
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    pad_x = (x_max - x_min) * 0.08 or 0.5
    pad_y = (y_max - y_min) * 0.08 or 0.5

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% var)")
    ax.grid(True, linestyle="--", alpha=0.3)

    scat = ax.scatter([], [], s=60, color="steelblue", zorder=3, label="Particles")
    gbest_scat = ax.scatter([], [], s=150, marker="*", color="crimson",
                            zorder=5, label="gBest")
    ax.legend(loc="upper right")
    title = ax.set_title("")

    def init():
        scat.set_offsets(np.empty((0, 2)))
        gbest_scat.set_offsets(np.empty((0, 2)))
        return scat, gbest_scat, title

    def update(frame_idx):
        i = frames[frame_idx]
        pts = projections[i]
        scat.set_offsets(pts)
        gbest_scat.set_offsets(gbest_proj[i].reshape(1, 2))
        title.set_text(
            f"Particles – iter {i}/{len(projections) - 1} | "
            f"gBest fitness = {history[i]:.2f}"
        )
        return scat, gbest_scat, title

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(frames), interval=args.interval, blit=True)
    return fig, anim


# ---------------------------------------------------------------------------
# Animation 2: bin packing layout
# ---------------------------------------------------------------------------

def make_bins_animation(sol, instance, args):
    bin_snaps = sol.bin_snapshots
    history = sol.history

    frames = list(range(0, len(bin_snaps), args.step))

    max_bins = max(len(b) for b in bin_snaps)
    ncols = min(max_bins, 5)
    nrows = math.ceil(max_bins / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.8, nrows * 2.8 + 0.6),
                             squeeze=False)
    fig.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
    sup = fig.suptitle("", fontsize=10)

    # Stable color map: one colour per item ID
    n_items = instance.n
    cmap = plt.get_cmap("tab20", n_items)

    def draw_frame(frame_idx):
        i = frames[frame_idx]
        bins = bin_snaps[i]

        for r in range(nrows):
            for c in range(ncols):
                ax = axes[r][c]
                ax.cla()
                bin_idx = r * ncols + c
                ax.set_aspect("equal")
                ax.set_xlim(0, instance.bin_W)
                ax.set_ylim(0, instance.bin_H)
                ax.tick_params(left=False, bottom=False,
                               labelleft=False, labelbottom=False)

                if bin_idx < len(bins):
                    ax.set_title(f"Bin {bin_idx + 1}", fontsize=7, pad=2)
                    # Bin border
                    ax.add_patch(mpatches.Rectangle(
                        (0, 0), instance.bin_W, instance.bin_H,
                        linewidth=1.2, edgecolor="black", facecolor="whitesmoke"
                    ))
                    for pi in bins[bin_idx].placed:
                        color = cmap((pi.item.id - 1) % n_items)
                        ax.add_patch(mpatches.Rectangle(
                            (pi.x, pi.y), pi.w, pi.h,
                            linewidth=0.5, edgecolor="black", facecolor=color, alpha=0.85
                        ))
                        ax.text(
                            pi.x + pi.w / 2, pi.y + pi.h / 2,
                            str(pi.item.id), ha="center", va="center",
                            fontsize=5, color="black"
                        )
                else:
                    ax.set_visible(False)

        sup.set_text(
            f"Bin packing – iter {i}/{len(bin_snaps) - 1} | "
            f"bins = {len(bins)} | gBest fitness = {history[i]:.2f}"
        )

    def init():
        draw_frame(0)
        return []

    def update(frame_idx):
        draw_frame(frame_idx)
        return []

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(frames), interval=args.interval, blit=False)
    return fig, anim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    fname = f"BPPC_{args.dataset_id}_{args.beta_idx}_{args.run}.txt"
    fpath = dataset_dir / fname

    if not fpath.exists():
        print(f"File not found: {fpath}")
        sys.exit(1)

    print(f"Loading {fname} ...")
    instance = load_instance(str(fpath), seed=args.run)

    print(f"Running PSO ({args.iters} iters, pop={args.pop}) with recording ...")
    sol = run(
        instance,
        population=args.pop,
        max_iter=args.iters,
        seed=args.run * 1000 + args.beta_idx,
        record=True,
    )
    print(f"Done. fitness={sol.fitness:.2f}  bins={sol.n_bins}")

    print("Building particle animation ...")
    fig_p, anim_p = make_particle_animation(sol, args)

    print("Building bins animation ...")
    fig_b, anim_b = make_bins_animation(sol, instance, args)

    if args.output_particles:
        print(f"Saving particle animation to {args.output_particles} ...")
        anim_p.save(args.output_particles, writer="pillow")
        print("Saved.")
    if args.output_bins:
        print(f"Saving bins animation to {args.output_bins} ...")
        anim_b.save(args.output_bins, writer="pillow")
        print("Saved.")

    if not args.output_particles and not args.output_bins:
        plt.show()
    elif not args.output_particles:
        plt.figure(fig_p.number)
        plt.show()
    elif not args.output_bins:
        plt.figure(fig_b.number)
        plt.show()


if __name__ == "__main__":
    main()
