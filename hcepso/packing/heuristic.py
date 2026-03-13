"""Bottom-Left Fill (BLF) 2D bin-packing heuristic with conflict awareness."""

from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional
from ..instance.item import Item


@dataclass
class PlacedItem:
    item: Item
    x: float
    y: float
    rotated: bool

    @property
    def w(self) -> float:
        return self.item.height if self.rotated else self.item.width

    @property
    def h(self) -> float:
        return self.item.width if self.rotated else self.item.height

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass
class Bin:
    W: float
    H: float
    placed: List[PlacedItem] = field(default_factory=list)
    item_ids: Set[int] = field(default_factory=set)

    def total_weight(self) -> float:
        return sum(p.item.weight for p in self.placed)

    def used_area(self) -> float:
        return sum(p.area for p in self.placed)

    def center_of_mass(self) -> Tuple[float, float]:
        if not self.placed:
            return self.W / 2, self.H / 2
        total_area = sum(p.area for p in self.placed)
        if total_area == 0:
            return self.W / 2, self.H / 2
        cx = sum(p.cx * p.area for p in self.placed) / total_area
        cy = sum(p.cy * p.area for p in self.placed) / total_area
        return cx, cy


def _overlaps(p: PlacedItem, x: float, y: float, w: float, h: float) -> bool:
    """True if the rectangle (x,y,w,h) overlaps with PlacedItem p."""
    return not (x >= p.x2 or x + w <= p.x or y >= p.y2 or y + h <= p.y)


def _find_blf_position(
    bin_: Bin, w: float, h: float
) -> Optional[Tuple[float, float]]:
    """
    Find the Bottom-Left-most position for a rectangle of size (w, h) in bin_.
    Candidate x-coordinates: 0 and all right edges of placed items.
    Candidate y-coordinates: 0 and all top edges of placed items.
    """
    xs = [0.0] + [p.x2 for p in bin_.placed]
    ys = [0.0] + [p.y2 for p in bin_.placed]

    best = None
    for y in sorted(set(ys)):
        if y + h > bin_.H:
            continue
        for x in sorted(set(xs)):
            if x + w > bin_.W:
                continue
            if all(not _overlaps(p, x, y, w, h) for p in bin_.placed):
                if best is None or (y, x) < best:
                    best = (y, x)
    if best is None:
        return None
    return best[1], best[0]  # (x, y)


def pack_items(
    sequence: List[Tuple[Item, bool]],  # (item, rotated)
    bin_W: float,
    bin_H: float,
    conflicts: List[Set[int]],
) -> List[Bin]:
    """
    Bottom-Left Fill heuristic.

    sequence: ordered list of (item, rotated) tuples from decoded particle.
    Returns list of Bin objects with placed items.
    """
    bins: List[Bin] = []

    for item, rotated in sequence:
        w = item.height if rotated else item.width
        h = item.width if rotated else item.height

        placed = False
        for bin_ in bins:
            # Conflict check: item must not conflict with any item already in this bin
            item_conflicts = conflicts[item.id - 1]
            if item_conflicts & bin_.item_ids:
                continue  # conflict — skip this bin

            pos = _find_blf_position(bin_, w, h)
            if pos is not None:
                x, y = pos
                bin_.placed.append(PlacedItem(item=item, x=x, y=y, rotated=rotated))
                bin_.item_ids.add(item.id)
                placed = True
                break

        if not placed:
            new_bin = Bin(W=bin_W, H=bin_H)
            pos = _find_blf_position(new_bin, w, h)
            if pos is None:
                # Item is larger than bin — place at (0,0) as fallback
                pos = (0.0, 0.0)
            x, y = pos
            new_bin.placed.append(PlacedItem(item=item, x=x, y=y, rotated=rotated))
            new_bin.item_ids.add(item.id)
            bins.append(new_bin)

    return bins
