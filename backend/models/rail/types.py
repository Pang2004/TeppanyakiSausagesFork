"""Public result types for Rail corrugation inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

RailLabel = Literal["Normal", "Side I", "Side II"]


@dataclass(frozen=True)
class RailPrediction:
    """Prediction and display diagnostics for one Rail recording."""

    file_id: str
    prediction: RailLabel
    scores: dict[str, float]
    side_energy: dict[str, dict[str, float]]
    dominant_frequency: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for the app adapter."""

        return asdict(self)
