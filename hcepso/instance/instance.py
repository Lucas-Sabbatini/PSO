from dataclasses import dataclass, field
from typing import List, Set, Tuple
from .item import Item


@dataclass
class Instance:
    items: List[Item]
    bin_W: float
    bin_H: float
    bin_Q: float          # weight capacity
    bin_delta: float      # max center-of-mass displacement
    # conflicts[i] = set of item ids that conflict with item i (1-indexed)
    conflicts: List[Set[int]] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.items)

    def conflict_pair(self, id_a: int, id_b: int) -> bool:
        """Return True if items with given ids conflict."""
        return id_b in self.conflicts[id_a - 1]
