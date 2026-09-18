"""Public result type for SHM inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SHMPrediction:
    """Predicted fatigue damage and interpretable cycle diagnostics."""

    file_id: str
    prediction: float
    cycle_count: float
    equivalent_stress_amplitude: float
    maximum_cycle_range: float
    estimated_percentage_error: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for app integration."""

        return asdict(self)
