from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Detection:
    """
    Represents a single detected object.
    """

    label: str
    score: float
    bbox: Tuple[int, int, int, int]   # (xmin, ymin, xmax, ymax)

    mask: Optional[object] = None
    depth: Optional[float] = None
    track_id: Optional[int] = None
    caption: Optional[str] = None