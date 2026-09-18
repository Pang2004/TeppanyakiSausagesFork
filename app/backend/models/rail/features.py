"""Side-aware feature extraction for Rail corrugation recordings."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats

SAMPLE_RATE_HZ = 10_000
EXPECTED_SAMPLES = 10_000
EXPECTED_COLUMNS = 129
SPEED_COLUMN = "Rotating speed"
EPSILON = 1e-12

_SENSOR_PATTERN = re.compile(
    r"^(Vibration|Shock) of bearing in position ([1-8]) of car ([1-8])$"
)


class RailInputError(ValueError):
    """Raised when an input file does not match the official Rail schema."""


@dataclass(frozen=True)
class FeatureConfig:
    """Versioned signal-processing configuration stored with the model."""

    schema_version: str = "rail-features-v1"
    sample_rate_hz: int = SAMPLE_RATE_HZ
    expected_samples: int = EXPECTED_SAMPLES
    welch_nperseg: int = 2048
    frequency_bands: tuple[tuple[int, int], ...] = (
        (0, 50),
        (50, 100),
        (100, 250),
        (250, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 5000),
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> FeatureConfig:
        bands = tuple(
            tuple(int(bound) for bound in band) for band in values["frequency_bands"]
        )
        return cls(
            schema_version=str(values["schema_version"]),
            sample_rate_hz=int(values["sample_rate_hz"]),
            expected_samples=int(values["expected_samples"]),
            welch_nperseg=int(values["welch_nperseg"]),
            frequency_bands=bands,
        )


@dataclass(frozen=True)
class SensorColumn:
    """Physical meaning of one input sensor column."""

    name: str
    index: int
    signal_type: str
    position: int
    car: int
    side: str


def parse_sensor_layout(columns: Iterable[str]) -> list[SensorColumn]:
    """Validate and map the 128 vibration/shock columns to side and sensor type."""

    names = list(columns)
    if len(names) != EXPECTED_COLUMNS:
        raise RailInputError(
            f"Expected {EXPECTED_COLUMNS} columns, found {len(names)}."
        )
    if names[0] != SPEED_COLUMN:
        raise RailInputError(
            f"First column must be {SPEED_COLUMN!r}, found {names[0]!r}."
        )

    layout: list[SensorColumn] = []
    seen: set[tuple[str, int, int]] = set()
    for index, name in enumerate(names[1:], start=1):
        match = _SENSOR_PATTERN.fullmatch(name)
        if match is None:
            raise RailInputError(f"Unrecognized Rail sensor column: {name!r}.")
        raw_type, position_text, car_text = match.groups()
        position = int(position_text)
        car = int(car_text)
        signal_type = raw_type.lower()
        key = (signal_type, position, car)
        if key in seen:
            raise RailInputError(f"Duplicate Rail sensor column: {name!r}.")
        seen.add(key)
        layout.append(
            SensorColumn(
                name=name,
                index=index,
                signal_type=signal_type,
                position=position,
                car=car,
                side="i" if position % 2 else "ii",
            )
        )

    if len(layout) != 128:
        raise RailInputError(f"Expected 128 sensor columns, found {len(layout)}.")
    return layout


def load_recording(
    path: str | Path, config: FeatureConfig | None = None
) -> tuple[pd.DataFrame, list[SensorColumn]]:
    """Load and strictly validate one official Rail CSV."""

    config = config or FeatureConfig()
    source = Path(path)
    try:
        frame = pd.read_csv(source, dtype=np.float32)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise RailInputError(f"Could not read {source}: {exc}") from exc

    layout = parse_sensor_layout(frame.columns)
    if len(frame) != config.expected_samples:
        raise RailInputError(
            f"{source.name} must contain {config.expected_samples} samples; found {len(frame)}."
        )
    values = frame.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(values).all():
        raise RailInputError(f"{source.name} contains missing or non-finite values.")
    return frame, layout


def _aggregate_channel_stat(
    features: dict[str, float], prefix: str, statistic: str, values: np.ndarray
) -> None:
    clean = np.nan_to_num(
        np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    for aggregate, value in (
        ("mean", np.mean(clean)),
        ("median", np.median(clean)),
        ("std", np.std(clean)),
        ("max", np.max(clean)),
    ):
        features[f"{prefix}__{statistic}__{aggregate}"] = float(value)


def _extract_group_features(
    values: np.ndarray,
    prefix: str,
    config: FeatureConfig,
) -> dict[str, float]:
    features: dict[str, float] = {}
    channel_mean = np.mean(values, axis=0, dtype=np.float64)
    channel_std = np.std(values, axis=0, dtype=np.float64)
    channel_rms = np.sqrt(np.mean(np.square(values, dtype=np.float64), axis=0))
    channel_ptp = np.ptp(values, axis=0)
    channel_max_abs = np.max(np.abs(values), axis=0)
    channel_skew = stats.skew(values, axis=0, bias=False)
    channel_kurtosis = stats.kurtosis(values, axis=0, fisher=True, bias=False)
    channel_crest = channel_max_abs / np.maximum(channel_rms, EPSILON)

    for name, channel_values in (
        ("mean", channel_mean),
        ("std", channel_std),
        ("rms", channel_rms),
        ("peak_to_peak", channel_ptp),
        ("max_abs", channel_max_abs),
        ("skewness", channel_skew),
        ("kurtosis", channel_kurtosis),
        ("crest_factor", channel_crest),
    ):
        _aggregate_channel_stat(features, prefix, name, channel_values)

    frequencies, psd = signal.welch(
        values,
        fs=config.sample_rate_hz,
        nperseg=config.welch_nperseg,
        axis=0,
        detrend="constant",
        scaling="density",
    )
    total_power = np.trapezoid(psd, frequencies, axis=0)
    safe_total = np.maximum(total_power, EPSILON)

    for low, high in config.frequency_bands:
        include_high = high == config.sample_rate_hz // 2
        mask = (frequencies >= low) & (
            frequencies <= high if include_high else frequencies < high
        )
        band_power = np.trapezoid(psd[mask], frequencies[mask], axis=0) / safe_total
        _aggregate_channel_stat(
            features,
            prefix,
            f"band_{low}_{high}_fraction",
            band_power,
        )

    non_dc_psd = psd.copy()
    non_dc_psd[0, :] = 0.0
    dominant = frequencies[np.argmax(non_dc_psd, axis=0)]
    centroid = np.sum(frequencies[:, None] * psd, axis=0) / np.maximum(
        np.sum(psd, axis=0), EPSILON
    )
    _aggregate_channel_stat(features, prefix, "dominant_frequency", dominant)
    _aggregate_channel_stat(features, prefix, "spectral_centroid", centroid)
    return features


def extract_features(
    path: str | Path, config: FeatureConfig | None = None
) -> dict[str, float | str]:
    """Extract one deterministic, side-aware feature row from a Rail CSV."""

    config = config or FeatureConfig()
    source = Path(path)
    frame, layout = load_recording(source, config)
    matrix = frame.to_numpy(dtype=np.float32, copy=False)
    speed = matrix[:, 0]

    features: dict[str, float | str] = {"file_id": source.name}
    features["speed__transition_count"] = float(np.count_nonzero(np.diff(speed) != 0))
    features["speed__duty_cycle"] = float(np.mean(speed > 0))
    features["speed__mean"] = float(np.mean(speed))
    features["speed__std"] = float(np.std(speed))

    grouped_features: dict[tuple[str, str], dict[str, float]] = {}
    for side in ("i", "ii"):
        for signal_type in ("vibration", "shock"):
            indices = [
                sensor.index
                for sensor in layout
                if sensor.side == side and sensor.signal_type == signal_type
            ]
            if len(indices) != 32:
                raise RailInputError(
                    f"Expected 32 {signal_type} channels for Side {side.upper()}, found {len(indices)}."
                )
            prefix = f"side_{side}_{signal_type}"
            group = _extract_group_features(matrix[:, indices], prefix, config)
            grouped_features[(side, signal_type)] = group
            features.update(group)

    for signal_type in ("vibration", "shock"):
        side_i = grouped_features[("i", signal_type)]
        side_ii = grouped_features[("ii", signal_type)]
        i_prefix = f"side_i_{signal_type}__"
        ii_prefix = f"side_ii_{signal_type}__"
        for i_name, i_value in side_i.items():
            suffix = i_name.removeprefix(i_prefix)
            ii_name = f"{ii_prefix}{suffix}"
            ii_value = side_ii[ii_name]
            comparison = f"side_comparison_{signal_type}__{suffix}"
            features[f"{comparison}__difference"] = float(i_value - ii_value)
            features[f"{comparison}__log_abs_ratio"] = float(
                np.log((abs(i_value) + EPSILON) / (abs(ii_value) + EPSILON))
            )
            features[f"{comparison}__relative_difference"] = float(
                (i_value - ii_value) / (abs(i_value) + abs(ii_value) + EPSILON)
            )

    numeric_values = np.asarray(
        [value for key, value in features.items() if key != "file_id"], dtype=np.float64
    )
    if not np.isfinite(numeric_values).all():
        raise RailInputError(
            f"Feature extraction produced non-finite values for {source.name}."
        )
    return features
