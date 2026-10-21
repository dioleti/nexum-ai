from dataclasses import dataclass
from enum import Enum


class RegionType(Enum):
    TABLE = "table"
    TEXT = "text"
    IMAGE = "image"
    FLOWCHART = "flowchart"


@dataclass(frozen=False, slots=True)
class Region:
    type: RegionType
    box: tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)
    score: float
    text: str | None = None