"""Public result types for ACV inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ACVPrediction:
    """Ranked cars and display diagnostics for one ACV workbook."""

    file_id: str
    ranked_cars: tuple[str, ...]
    car_scores: dict[str, float]
    diagnostics: dict[str, dict[str, float | None]]
    schema: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for the app adapter."""

        values = asdict(self)
        values["ranked_cars"] = list(self.ranked_cars)
        return values
