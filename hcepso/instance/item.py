from dataclasses import dataclass


@dataclass
class Item:
    id: int
    width: float
    height: float
    weight: float
    can_rotate: bool = True
