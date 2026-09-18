"""Headerless stress loading and physics-informed SHM feature extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rainflow
from scipy import stats


class SHMInputError(ValueError):
    """Raised when an SHM stress file is malformed."""


@dataclass(frozen=True)
class SHMFeatureConfig:
    """Versioned fatigue feature configuration stored in the model artifact."""

    schema_version: str = "shm-features-v1"
    block_count: int = 128
    candidate_exponents: tuple[int, ...] = (3, 4, 5, 6, 7)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> SHMFeatureConfig:
        return cls(
            schema_version=str(values["schema_version"]),
            block_count=int(values["block_count"]),
            candidate_exponents=tuple(
                int(value) for value in values["candidate_exponents"]
            ),
        )


def load_stress(path: str | Path) -> np.ndarray:
    """Load one headerless, one-column stress history without dropping its first sample."""

    source = Path(path)
    try:
        values = np.loadtxt(source, delimiter=",", dtype=np.float64, ndmin=1)
    except (OSError, ValueError) as exc:
        raise SHMInputError(
            f"Could not read numeric SHM stress data from {source}: {exc}"
        ) from exc
    if values.ndim != 1:
        raise SHMInputError(
            f"{source.name} must contain exactly one headerless numeric column."
        )
    if len(values) < 3:
        raise SHMInputError(
            f"{source.name} must contain at least three stress samples."
        )
    if not np.isfinite(values).all():
        raise SHMInputError(
            f"{source.name} contains missing or non-finite stress values."
        )
    return values


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probabilities: tuple[float, ...]
) -> np.ndarray:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    targets = np.asarray(probabilities) * cumulative[-1]
    return ordered_values[np.searchsorted(cumulative, targets, side="left")]


def miner_proxy(ranges: np.ndarray, counts: np.ndarray, exponent: int) -> float:
    """Return the uncalibrated Miner/S-N damage sum for one exponent."""

    amplitudes = ranges / 2.0
    return float(np.sum(counts * np.power(amplitudes, exponent)))


def extract_features(
    path: str | Path, config: SHMFeatureConfig | None = None
) -> dict[str, float | str]:
    """Extract deterministic distribution, dynamics, and rainflow features."""

    config = config or SHMFeatureConfig()
    source = Path(path)
    values = load_stress(source)
    centered = values - np.mean(values)
    differences = np.diff(values)

    extracted_cycles = np.asarray(
        list(rainflow.extract_cycles(values)), dtype=np.float64
    )
    if extracted_cycles.size == 0:
        ranges = np.zeros(1, dtype=np.float64)
        means = np.zeros(1, dtype=np.float64)
        counts = np.zeros(1, dtype=np.float64)
    else:
        ranges = extracted_cycles[:, 0]
        means = extracted_cycles[:, 1]
        counts = extracted_cycles[:, 2]
    total_cycles = float(np.sum(counts))

    features: dict[str, float | str] = {"file_id": source.name}
    features.update(
        {
            "signal__n_samples": float(len(values)),
            "signal__mean": float(np.mean(values)),
            "signal__std": float(np.std(values)),
            "signal__rms": float(np.sqrt(np.mean(np.square(values)))),
            "signal__min": float(np.min(values)),
            "signal__max": float(np.max(values)),
            "signal__range": float(np.ptp(values)),
            "signal__skewness": float(stats.skew(values, bias=False)),
            "signal__kurtosis": float(stats.kurtosis(values, fisher=True, bias=False)),
            "difference__std": float(np.std(differences)),
            "difference__rms": float(np.sqrt(np.mean(np.square(differences)))),
            "difference__mean_abs": float(np.mean(np.abs(differences))),
            "difference__max_abs": float(np.max(np.abs(differences))),
            "signal__zero_crossing_rate": float(
                np.mean(np.signbit(centered[1:]) != np.signbit(centered[:-1]))
            ),
        }
    )
    for name, value in zip(
        (
            "q001",
            "q005",
            "q01",
            "q05",
            "q10",
            "q25",
            "median",
            "q75",
            "q90",
            "q95",
            "q99",
            "q995",
            "q999",
        ),
        np.quantile(
            values,
            (
                0.001,
                0.005,
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
                0.995,
                0.999,
            ),
        ),
        strict=True,
    ):
        features[f"signal__{name}"] = float(value)
    for name, value in zip(
        ("q50", "q75", "q90", "q95", "q99", "q995", "q999", "q9999"),
        np.quantile(
            np.abs(centered), (0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999, 0.9999)
        ),
        strict=True,
    ):
        features[f"absolute_centered__{name}"] = float(value)

    block_count = min(config.block_count, len(values))
    usable = (len(values) // block_count) * block_count
    block_ranges = np.ptp(values[:usable].reshape(block_count, -1), axis=1)
    for name, value in zip(
        (
            "mean",
            "std",
            "min",
            "q10",
            "q25",
            "median",
            "q75",
            "q90",
            "q95",
            "q99",
            "max",
        ),
        (
            np.mean(block_ranges),
            np.std(block_ranges),
            np.min(block_ranges),
            *np.quantile(block_ranges, (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)),
            np.max(block_ranges),
        ),
        strict=True,
    ):
        features[f"block_range__{name}"] = float(value)

    features["rainflow__cycle_count"] = total_cycles
    features["rainflow__maximum_range"] = float(np.max(ranges))
    if total_cycles > 0:
        features["rainflow__mean_range"] = float(np.average(ranges, weights=counts))
        features["rainflow__std_range"] = float(
            np.sqrt(
                np.average(
                    np.square(ranges - np.average(ranges, weights=counts)),
                    weights=counts,
                )
            )
        )
        features["rainflow__mean_cycle_stress"] = float(
            np.average(means, weights=counts)
        )
        cycle_quantiles = _weighted_quantile(
            ranges, counts, (0.50, 0.75, 0.90, 0.95, 0.99, 0.999)
        )
    else:
        features["rainflow__mean_range"] = 0.0
        features["rainflow__std_range"] = 0.0
        features["rainflow__mean_cycle_stress"] = 0.0
        cycle_quantiles = np.zeros(6)
    for name, value in zip(
        ("q50", "q75", "q90", "q95", "q99", "q999"), cycle_quantiles, strict=True
    ):
        features[f"rainflow__range_{name}"] = float(value)
    for exponent in config.candidate_exponents:
        proxy = miner_proxy(ranges, counts, exponent)
        features[f"rainflow__miner_proxy_m{exponent}"] = proxy
        features[f"rainflow__equivalent_amplitude_m{exponent}"] = float(
            np.power(proxy / total_cycles, 1.0 / exponent) if total_cycles > 0 else 0.0
        )

    numeric = np.asarray(
        [value for name, value in features.items() if name != "file_id"],
        dtype=np.float64,
    )
    if not np.isfinite(numeric).all():
        raise SHMInputError(
            f"Feature extraction produced non-finite values for {source.name}."
        )
    return features


def residual_feature_names(columns: list[str]) -> list[str]:
    """Return the compact feature set allowed into the residual correction model."""

    excluded = {"file_id", "signal__n_samples"}
    return [
        name for name in columns if name not in excluded and "miner_proxy" not in name
    ]
