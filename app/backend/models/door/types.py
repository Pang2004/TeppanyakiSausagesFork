"""Public result types for Door fault inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

DoorLabel = Literal["Normal", "Abnormal resistance"]
DoorOperation = Literal["Open", "Close"]


@dataclass(frozen=True)
class DoorPrediction:
    """One detected operation and its resistance classification."""

    start_time: str
    end_time: str
    prediction: DoorLabel
    confidence: float
    operation: DoorOperation
    boundary_reason: str
    quality_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for app integration."""

        return asdict(self)
