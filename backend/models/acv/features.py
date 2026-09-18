"""Schema-aware feature extraction for ACV refrigerant-leak localisation."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CAR_COLUMN_PATTERN = re.compile(r"^Car (\d{2}) - (.+)$")
EPSILON = 1e-12

CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "cabin_temperature": (
        "Indoor Average Temperature",
        "Passenger Cabin Temperature Detected Value",
    ),
    "ambient_temperature": (
        "Outdoor Average Temperature",
        "Outside Temperature Sensor Reading",
        "Fresh Air Temperature Detected Value",
    ),
    "cooling_target": (
        "ACV Control Temperature (Cooling)",
        "Target Temperature Value",
    ),
    "running_mode": ("ACV Running Mode",),
    "information_valid": ("ACV Information Valid",),
}

TEMPERATURE_SCORE_FEATURES = (
    "temperature_relative_mean",
    "temperature_relative_q75",
    "temperature_relative_q90",
    "temperature_relative_q95",
    "temperature_positive_mean",
    "temperature_fraction_above_0_25",
    "temperature_fraction_above_0_50",
    "temperature_fraction_above_1_00",
    "target_error_relative_mean",
    "target_error_relative_q90",
    "temperature_longest_hot_run_fraction",
)


class ACVInputError(ValueError):
    """Raised when an input workbook cannot support ACV localisation."""


@dataclass(frozen=True)
class ACVFeatureConfig:
    """Versioned ACV preprocessing and ranking configuration."""

    schema_version: str = "acv-features-v1"
    minimum_temperature: float = 10.0
    maximum_temperature: float = 50.0
    minimum_target: float = 10.0
    maximum_target: float = 35.0
    minimum_valid_samples: int = 20
    minimum_peer_cars: int = 3
    pressure_weight: float = 0.7
    active_cooling_modes: tuple[str, ...] = (
        "automatic cooling",
        "full cooling",
        "half cooling",
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> ACVFeatureConfig:
        return cls(
            schema_version=str(values["schema_version"]),
            minimum_temperature=float(values["minimum_temperature"]),
            maximum_temperature=float(values["maximum_temperature"]),
            minimum_target=float(values["minimum_target"]),
            maximum_target=float(values["maximum_target"]),
            minimum_valid_samples=int(values["minimum_valid_samples"]),
            minimum_peer_cars=int(values["minimum_peer_cars"]),
            pressure_weight=float(values["pressure_weight"]),
            active_cooling_modes=tuple(
                str(value) for value in values["active_cooling_modes"]
            ),
        )


@dataclass(frozen=True)
class ACVWorkbook:
    """Validated workbook contents and its discovered car layout."""

    source: Path
    frame: pd.DataFrame
    car_ids: tuple[str, ...]
    columns: dict[str, dict[str, str]]


def _car_layout(
    columns: Iterable[object],
) -> tuple[tuple[str, ...], dict[str, dict[str, str]]]:
    car_ids: list[str] = []
    layout: dict[str, dict[str, str]] = {}
    for raw_column in columns:
        column = str(raw_column)
        match = CAR_COLUMN_PATTERN.fullmatch(column)
        if match is None:
            continue
        car_id, parameter = match.groups()
        if car_id not in layout:
            car_ids.append(car_id)
            layout[car_id] = {}
        if parameter in layout[car_id]:
            raise ACVInputError(
                f"Duplicate ACV parameter {parameter!r} for car {car_id}."
            )
        layout[car_id][parameter] = column
    if not car_ids:
        raise ACVInputError("No columns matching 'Car <NN> - <parameter>' were found.")
    return tuple(car_ids), layout


def load_workbook(path: str | Path) -> ACVWorkbook:
    """Load the first sheet and validate the dynamic ACV column layout."""

    source = Path(path)
    if source.suffix.lower() != ".xlsx":
        raise ACVInputError(f"ACV input must be an .xlsx workbook: {source}")
    try:
        frame = pd.read_excel(source, sheet_name=0, engine="openpyxl")
    except (OSError, ValueError, TypeError) as exc:
        raise ACVInputError(f"Could not read {source}: {exc}") from exc
    if frame.empty:
        raise ACVInputError(f"{source.name} contains no telemetry rows.")
    if "Time" not in frame.columns:
        raise ACVInputError(f"{source.name} is missing the 'Time' column.")
    timestamps = pd.to_datetime(frame["Time"], errors="coerce")
    if timestamps.notna().sum() < 2:
        raise ACVInputError(f"{source.name} has fewer than two valid timestamps.")
    frame = frame.assign(Time=timestamps).sort_values("Time", kind="stable")
    car_ids, layout = _car_layout(frame.columns)
    return ACVWorkbook(source, frame.reset_index(drop=True), car_ids, layout)


def _canonical_column(workbook: ACVWorkbook, car_id: str, name: str) -> str | None:
    parameters = workbook.columns[car_id]
    for alias in CANONICAL_ALIASES[name]:
        if alias in parameters:
            return parameters[alias]
    return None


def _numeric_series(
    workbook: ACVWorkbook, car_id: str, canonical_name: str
) -> pd.Series:
    column = _canonical_column(workbook, car_id, canonical_name)
    if column is None:
        return pd.Series(np.nan, index=workbook.frame.index, dtype=np.float64)
    return pd.to_numeric(workbook.frame[column], errors="coerce").astype(np.float64)


def _validity_mask(workbook: ACVWorkbook, car_id: str) -> pd.Series:
    column = _canonical_column(workbook, car_id, "information_valid")
    if column is None or workbook.frame[column].isna().all():
        return pd.Series(True, index=workbook.frame.index)
    values = workbook.frame[column].astype("string").str.strip().str.lower()
    return values.eq("valid")


def _active_cooling_mask(
    workbook: ACVWorkbook, car_id: str, config: ACVFeatureConfig
) -> pd.Series:
    column = _canonical_column(workbook, car_id, "running_mode")
    if column is None or workbook.frame[column].isna().all():
        return pd.Series(True, index=workbook.frame.index)
    values = workbook.frame[column].astype("string").str.strip().str.lower()
    return values.isin(config.active_cooling_modes)


def _longest_true_run_fraction(values: pd.Series) -> float:
    array = values.fillna(False).to_numpy(dtype=bool)
    if not array.size or not array.any():
        return 0.0
    changes = np.diff(np.concatenate(([False], array, [False])).astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return float(np.max(ends - starts) / array.size)


def _relative_temperature_features(
    workbook: ACVWorkbook, config: ACVFeatureConfig
) -> pd.DataFrame:
    cabin: dict[str, pd.Series] = {}
    target: dict[str, pd.Series] = {}
    active: dict[str, pd.Series] = {}
    raw_cabin: dict[str, pd.Series] = {}
    for car_id in workbook.car_ids:
        raw = _numeric_series(workbook, car_id, "cabin_temperature")
        valid = raw.between(
            config.minimum_temperature, config.maximum_temperature
        ) & _validity_mask(workbook, car_id)
        target_raw = _numeric_series(workbook, car_id, "cooling_target")
        target_valid = target_raw.between(config.minimum_target, config.maximum_target)
        cooling = _active_cooling_mask(workbook, car_id, config)
        raw_cabin[car_id] = raw
        cabin[car_id] = raw.where(valid & cooling)
        target[car_id] = target_raw.where(target_valid & cooling)
        active[car_id] = cooling

    cabin_frame = pd.DataFrame(cabin)
    target_frame = pd.DataFrame(target)
    enough_peers = cabin_frame.notna().sum(axis=1) >= config.minimum_peer_cars
    fleet_median = cabin_frame.median(axis=1).where(enough_peers)
    relative = cabin_frame.sub(fleet_median, axis=0)
    target_error = cabin_frame - target_frame
    error_median = target_error.median(axis=1).where(enough_peers)
    relative_error = target_error.sub(error_median, axis=0)

    rows: list[dict[str, float | str]] = []
    for car_id in workbook.car_ids:
        values = relative[car_id].dropna()
        errors = relative_error[car_id].dropna()
        sample_count = int(values.size)
        row: dict[str, float | str] = {
            "car_id": car_id,
            "temperature_sample_count": float(sample_count),
            "temperature_coverage": float(sample_count / len(workbook.frame)),
            "cabin_temperature_median": float(cabin_frame[car_id].median()),
            "raw_zero_temperature_fraction": float(raw_cabin[car_id].eq(0).mean()),
            "active_cooling_fraction": float(active[car_id].mean()),
        }
        if sample_count < config.minimum_valid_samples:
            for name in TEMPERATURE_SCORE_FEATURES:
                row[name] = np.nan
            rows.append(row)
            continue
        positive = values.clip(lower=0)
        row.update(
            {
                "temperature_relative_mean": float(values.mean()),
                "temperature_relative_q75": float(values.quantile(0.75)),
                "temperature_relative_q90": float(values.quantile(0.90)),
                "temperature_relative_q95": float(values.quantile(0.95)),
                "temperature_positive_mean": float(positive.mean()),
                "temperature_fraction_above_0_25": float((values > 0.25).mean()),
                "temperature_fraction_above_0_50": float((values > 0.50).mean()),
                "temperature_fraction_above_1_00": float((values > 1.00).mean()),
                "target_error_relative_mean": float(errors.mean())
                if not errors.empty
                else np.nan,
                "target_error_relative_q90": float(errors.quantile(0.90))
                if not errors.empty
                else np.nan,
                "temperature_longest_hot_run_fraction": _longest_true_run_fraction(
                    relative[car_id] > 0.5
                ),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows).set_index("car_id")
    usable = result["temperature_sample_count"] >= config.minimum_valid_samples
    ranks = result.loc[usable, TEMPERATURE_SCORE_FEATURES].rank(
        pct=True, method="average", na_option="keep"
    )
    result["temperature_score"] = ranks.mean(axis=1, skipna=True)
    return result


def _compressor_running(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype("string").str.strip().str.lower()
    known = numeric.notna() | text.isin(
        ["running", "stopped", "on", "off", "yes", "no", "true", "false"]
    )
    running = numeric.eq(1) | text.isin(["running", "on", "yes", "true"])
    return running, known


def _normalized_difference(first: float, second: float) -> float:
    if not np.isfinite(first) or not np.isfinite(second):
        return np.nan
    return float(2 * abs(first - second) / (abs(first) + abs(second) + EPSILON))


def _pressure_features(workbook: ACVWorkbook, config: ACVFeatureConfig) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for car_id in workbook.car_ids:
        row: dict[str, float | str] = {"car_id": car_id}
        systems: dict[int, dict[str, float]] = {}
        for system in (1, 2):
            parameters = workbook.columns[car_id]
            high_name = f"Refrigeration System {system} High Pressure Value"
            low_name = f"Refrigeration System {system} Low Pressure Value"
            run_name = f"Compressor {system} Running"
            if not all(name in parameters for name in (high_name, low_name, run_name)):
                continue
            high = pd.to_numeric(workbook.frame[parameters[high_name]], errors="coerce")
            low = pd.to_numeric(workbook.frame[parameters[low_name]], errors="coerce")
            running, known = _compressor_running(workbook.frame[parameters[run_name]])
            valid = running & high.gt(0) & low.gt(0)
            if int(valid.sum()) < config.minimum_valid_samples:
                continue
            systems[system] = {
                "high": float(high[valid].median()),
                "low": float(low[valid].median()),
                "lift": float((high[valid] - low[valid]).median()),
                "duty": float(running[known].mean()),
                "samples": float(valid.sum()),
            }
        row["pressure_sample_count"] = float(
            min((values["samples"] for values in systems.values()), default=0.0)
        )
        if 1 in systems and 2 in systems:
            first, second = systems[1], systems[2]
            row["pressure_high_mismatch"] = _normalized_difference(
                first["high"], second["high"]
            )
            row["pressure_low_mismatch"] = _normalized_difference(
                first["low"], second["low"]
            )
            row["pressure_lift_mismatch"] = _normalized_difference(
                first["lift"], second["lift"]
            )
            row["compressor_duty_mismatch"] = abs(first["duty"] - second["duty"])
            row["pressure_mismatch"] = float(
                np.mean(
                    [
                        row["pressure_high_mismatch"],
                        row["pressure_low_mismatch"],
                        row["pressure_lift_mismatch"],
                        row["compressor_duty_mismatch"],
                    ]
                )
            )
        else:
            for name in (
                "pressure_high_mismatch",
                "pressure_low_mismatch",
                "pressure_lift_mismatch",
                "compressor_duty_mismatch",
                "pressure_mismatch",
            ):
                row[name] = np.nan
        rows.append(row)
    result = pd.DataFrame(rows).set_index("car_id")
    usable = result["pressure_mismatch"].notna()
    result["pressure_score"] = result.loc[usable, "pressure_mismatch"].rank(
        pct=True, method="average"
    )
    return result


def extract_car_features(
    path: str | Path, config: ACVFeatureConfig | None = None
) -> pd.DataFrame:
    """Extract one row of comparable ranking features per header-discovered car."""

    config = config or ACVFeatureConfig()
    workbook = load_workbook(path)
    temperature = _relative_temperature_features(workbook, config)
    pressure = _pressure_features(workbook, config)
    result = temperature.join(pressure, how="left")
    has_temperature = result["temperature_score"].notna()
    has_pressure = result["pressure_score"].notna()
    result["combined_score"] = np.nan
    result.loc[has_temperature, "combined_score"] = result.loc[
        has_temperature, "temperature_score"
    ]
    both = has_temperature & has_pressure
    result.loc[both, "combined_score"] = (1 - config.pressure_weight) * result.loc[
        both, "temperature_score"
    ] + config.pressure_weight * result.loc[both, "pressure_score"]
    pressure_only = ~has_temperature & has_pressure
    result.loc[pressure_only, "combined_score"] = result.loc[
        pressure_only, "pressure_score"
    ]
    result.insert(0, "file_id", workbook.source.name)
    result.insert(1, "car_id", result.index)
    result.insert(
        2,
        "schema",
        "rich_pressure" if has_pressure.any() else "compact_temperature",
    )
    return result.reset_index(drop=True)


def rank_feature_frame(
    features: pd.DataFrame, score_column: str = "combined_score"
) -> list[str]:
    """Rank every car deterministically while retaining cars without telemetry."""

    if score_column not in features:
        raise ValueError(f"Unknown ACV score column: {score_column}")
    ranked = features.copy()
    ranked["_score"] = pd.to_numeric(ranked[score_column], errors="coerce").fillna(
        -np.inf
    )
    ranked["_coverage"] = pd.to_numeric(
        ranked.get("temperature_coverage", 0.0), errors="coerce"
    ).fillna(0.0)
    ranked = ranked.sort_values(
        ["_score", "_coverage", "car_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    return ranked["car_id"].astype(str).tolist()
