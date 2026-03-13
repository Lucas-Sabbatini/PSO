"""Fitness function: f = Σ_k [W·H - ΣE_ik + α·PC_k + λ·w_k + δ·b_k]"""

import math
from typing import List, Set
from .heuristic import Bin
from .. import config


def compute_fitness(
    bins: List[Bin],
    W: float,
    H: float,
    Q: float,
    delta_max: float,
    conflicts: List[Set[int]],
    alpha: float = config.ALPHA,
    lambda_: float = config.LAMBDA_,
    delta: float = config.DELTA,
) -> float:
    """
    f = Σ_k [ W·H - used_area_k + α·PC_k + λ·w_k + δ·b_k ]

    PC_k  = number of conflicting pairs in bin k
    w_k   = max(0, total_weight_k - Q)
    b_k   = max(0, displacement_k - delta_max)
    """
    total = 0.0
    bin_area = W * H

    for bin_ in bins:
        # Unused area term
        used = bin_.used_area()
        total += bin_area - used

        # Conflict penalty: count conflicting pairs in this bin
        pc = 0
        item_ids = list(bin_.item_ids)
        for i in range(len(item_ids)):
            for j in range(i + 1, len(item_ids)):
                a, b = item_ids[i], item_ids[j]
                if b in conflicts[a - 1]:
                    pc += 1
        total += alpha * pc

        # Weight violation penalty
        w_viol = max(0.0, bin_.total_weight() - Q)
        total += lambda_ * w_viol

        # Center-of-mass displacement penalty
        cx, cy = bin_.center_of_mass()
        disp = math.sqrt((cx - W / 2) ** 2 + (cy - H / 2) ** 2)
        b_viol = max(0.0, disp - delta_max)
        total += delta * b_viol

    return total
