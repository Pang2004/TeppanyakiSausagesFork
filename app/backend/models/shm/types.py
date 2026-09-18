"""Public result type for SHM inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SHMPrediction:
    """Predicted fatigue damage and interpretable cycle diagnostics.

    ``estimated_percentage_error`` is retained for API compatibility. It is the
    historical outer-validation 95th percentile of absolute percentage error,
    expressed as a fraction, identical for every file. It is not a per-file
    uncertainty estimate or a prediction interval with guaranteed coverage.
    """

    file_id: str
    prediction: float
    cycle_count: float
    equivalent_stress_amplitude: float
    maximum_cycle_range: float
    estimated_percentage_error: float
    error_indicator_kind: str = "historical_validation_p95_absolute_percentage_error"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for app integration."""

        return asdict(self)
