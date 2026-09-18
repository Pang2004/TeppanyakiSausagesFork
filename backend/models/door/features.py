"""Door stream validation, hybrid segmentation, and feature extraction."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DATETIME_COLUMN = "Datetime"
ANALOG_COLUMNS = (
    "Motor current(mA)",
    "Motor Voltage(10mV)",
    "Motor electrodynamic force",
    "Door leaf position",
)
TIMING_COLUMNS = ("Door opening time(.1s)", "Door closing time(.1s)")
BINARY_COLUMNS = (
    "Close command",
    "Open command",
    "DCSR",
    "DCSL",
    "DLSR",
    "DLSL",
    "Door Opened",
    "Door Locked",
    "Door is opening",
    "Door is closing",
)
EXPECTED_COLUMNS = (
    DATETIME_COLUMN,
    *ANALOG_COLUMNS[:3],
    *TIMING_COLUMNS,
    *BINARY_COLUMNS,
    ANALOG_COLUMNS[3],
)

_TIMESTAMP_PATTERN = re.compile(
    r"^(\d{4})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,3})$"
)


class DoorInputError(ValueError):
    """Raised when a Door stream does not match the supplied schema."""


@dataclass(frozen=True)
class DoorFeatureConfig:
    """Versioned segmentation and feature settings stored with the artifact."""

    schema_version: str = "door-features-v1"
    gap_multiplier: float = 5.0
    minimum_gap_seconds: float = 0.1
    terminal_dwell_seconds: float = 0.5
    maximum_operation_seconds: float = 10.0
    resample_points: int = 20

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> DoorFeatureConfig:
        return cls(
            schema_version=str(values["schema_version"]),
            gap_multiplier=float(values["gap_multiplier"]),
            minimum_gap_seconds=float(values["minimum_gap_seconds"]),
            terminal_dwell_seconds=float(values["terminal_dwell_seconds"]),
            maximum_operation_seconds=float(values["maximum_operation_seconds"]),
            resample_points=int(values["resample_points"]),
        )


@dataclass(frozen=True)
class DoorSegment:
    """Half-open row range for one detected door operation."""

    start: int
    end: int
    operation: Literal["Open", "Close"]
    boundary_reason: str
    quality_flags: tuple[str, ...]


def parse_door_timestamp(value: str) -> datetime:
    """Parse the dataset's non-zero-padded timestamp format."""

    match = _TIMESTAMP_PATTERN.fullmatch(str(value).strip())
    if match is None:
        raise DoorInputError(f"Invalid Door timestamp: {value!r}.")
    year, month, day, hour, minute, second, millisecond = map(int, match.groups())
    try:
        return datetime(year, month, day, hour, minute, second, millisecond * 1000)  # noqa: DTZ001
    except ValueError as exc:
        raise DoorInputError(f"Invalid Door timestamp: {value!r}.") from exc


def load_stream(path: str | Path) -> tuple[pd.DataFrame, tuple[datetime, ...]]:
    """Load and strictly validate one Door CSV stream."""

    source = Path(path)
    try:
        frame = pd.read_csv(source)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DoorInputError(f"Could not read {source}: {exc}") from exc
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise DoorInputError(
            f"Door stream must contain the expected 17 columns in order; found {list(frame.columns)!r}."
        )
    if frame.empty:
        raise DoorInputError("Door stream is empty.")

    timestamps = tuple(parse_door_timestamp(value) for value in frame[DATETIME_COLUMN])
    deltas = np.diff(np.asarray(timestamps, dtype="datetime64[us]")).astype(
        "timedelta64[us]"
    )
    if np.any(deltas <= np.timedelta64(0, "us")):
        raise DoorInputError("Door timestamps must be strictly increasing.")

    numeric_columns = [
        column for column in EXPECTED_COLUMNS if column != DATETIME_COLUMN
    ]
    try:
        frame[numeric_columns] = frame[numeric_columns].apply(
            pd.to_numeric, errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise DoorInputError(
            f"Door stream contains non-numeric measurements: {exc}"
        ) from exc
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=np.float64)).all():
        raise DoorInputError("Door stream contains missing or non-finite measurements.")
    return frame, timestamps


def _operation_at(frame: pd.DataFrame, index: int) -> Literal["Open", "Close"] | None:
    open_active = bool(
        frame.at[index, "Open command"] or frame.at[index, "Door is opening"]
    )
    close_active = bool(
        frame.at[index, "Close command"] or frame.at[index, "Door is closing"]
    )
    if open_active and not close_active:
        return "Open"
    if close_active and not open_active:
        return "Close"
    return None


def _infer_operation(
    frame: pd.DataFrame, start: int, end: int
) -> Literal["Open", "Close"]:
    section = frame.iloc[start:end]
    open_votes = float(
        section["Open command"].mean() + section["Door is opening"].mean()
    )
    close_votes = float(
        section["Close command"].mean() + section["Door is closing"].mean()
    )
    if open_votes != close_votes:
        return "Open" if open_votes > close_votes else "Close"
    position = section["Door leaf position"].to_numpy(dtype=np.float64)
    return "Open" if position[-1] > position[0] else "Close"


def _terminal_state(
    frame: pd.DataFrame, start: int, index: int, operation: Literal["Open", "Close"]
) -> bool:
    section = frame.iloc[start : index + 1]
    position = section["Door leaf position"].to_numpy(dtype=np.float64)
    if len(position) < 5:
        return False
    travel = max(float(np.ptp(position)), 1.0)
    recent = position[-5:]
    plateau = float(np.ptp(recent)) <= max(2.0, 0.01 * travel)
    if operation == "Close":
        switches = all(
            bool(frame.at[index, name]) for name in ("DCSR", "DCSL", "DLSR", "DLSL")
        )
        at_extreme = position[-1] <= float(np.min(position)) + max(2.0, 0.01 * travel)
    else:
        switches = bool(frame.at[index, "Door Opened"]) and not any(
            bool(frame.at[index, name]) for name in ("DCSR", "DCSL", "DLSR", "DLSL")
        )
        at_extreme = position[-1] >= float(np.max(position)) - max(2.0, 0.01 * travel)
    return plateau and switches and at_extreme


def _quality_flags(
    frame: pd.DataFrame, start: int, end: int, operation: Literal["Open", "Close"]
) -> tuple[str, ...]:
    section = frame.iloc[start:end]
    position = section["Door leaf position"].to_numpy(dtype=np.float64)
    flags: list[str] = []
    net_motion = float(position[-1] - position[0])
    if operation == "Open" and net_motion <= 0:
        flags.append("unexpected_position_direction")
    if operation == "Close" and net_motion >= 0:
        flags.append("unexpected_position_direction")
    if not _terminal_state(frame, start, end - 1, operation):
        flags.append("terminal_state_not_confirmed")
    return tuple(flags)


def _segment_block(
    frame: pd.DataFrame,
    timestamps: tuple[datetime, ...],
    block_start: int,
    block_end: int,
    cadence_seconds: float,
    config: DoorFeatureConfig,
    block_end_reason: str,
) -> list[DoorSegment]:
    segments: list[DoorSegment] = []
    current_start: int | None = None
    current_operation: Literal["Open", "Close"] | None = None
    terminal_since: datetime | None = None
    awaiting_reset = False

    for index in range(block_start, block_end):
        operation = _operation_at(frame, index)
        if awaiting_reset:
            if operation is None:
                awaiting_reset = False
            else:
                continue

        if current_start is None:
            if operation is None:
                continue
            current_start = index
            current_operation = operation
            terminal_since = None
            continue

        assert current_operation is not None
        if operation is not None and operation != current_operation:
            flags = _quality_flags(frame, current_start, index, current_operation)
            segments.append(
                DoorSegment(
                    current_start, index, current_operation, "operation_change", flags
                )
            )
            current_start = index
            current_operation = operation
            terminal_since = None
            continue
        if operation is None:
            flags = _quality_flags(frame, current_start, index, current_operation)
            segments.append(
                DoorSegment(
                    current_start, index, current_operation, "inactive_state", flags
                )
            )
            current_start = None
            current_operation = None
            terminal_since = None
            continue

        if _terminal_state(frame, current_start, index, current_operation):
            terminal_since = terminal_since or timestamps[index]
            dwell = (
                timestamps[index] - terminal_since
            ).total_seconds() + cadence_seconds
            if dwell >= config.terminal_dwell_seconds:
                flags = _quality_flags(
                    frame, current_start, index + 1, current_operation
                )
                segments.append(
                    DoorSegment(
                        current_start,
                        index + 1,
                        current_operation,
                        "terminal_dwell",
                        flags,
                    )
                )
                current_start = None
                current_operation = None
                terminal_since = None
                awaiting_reset = True
                continue
        else:
            terminal_since = None

        elapsed = (timestamps[index] - timestamps[current_start]).total_seconds()
        if elapsed >= config.maximum_operation_seconds:
            flags = (
                *_quality_flags(frame, current_start, index + 1, current_operation),
                "maximum_duration_reached",
            )
            segments.append(
                DoorSegment(
                    current_start,
                    index + 1,
                    current_operation,
                    "maximum_duration",
                    flags,
                )
            )
            current_start = None
            current_operation = None
            terminal_since = None
            awaiting_reset = True

    if current_start is not None:
        operation = current_operation or _infer_operation(
            frame, current_start, block_end
        )
        flags = _quality_flags(frame, current_start, block_end, operation)
        segments.append(
            DoorSegment(current_start, block_end, operation, block_end_reason, flags)
        )
    return segments


def detect_segments(
    frame: pd.DataFrame,
    timestamps: tuple[datetime, ...],
    config: DoorFeatureConfig | None = None,
) -> list[DoorSegment]:
    """Detect operations using gaps, with a live-stream state-machine fallback."""

    config = config or DoorFeatureConfig()
    delta_seconds = np.asarray(
        [(right - left).total_seconds() for left, right in pairwise(timestamps)]
    )
    cadence_seconds = float(np.median(delta_seconds)) if len(delta_seconds) else 0.02
    gap_threshold = max(
        config.minimum_gap_seconds, config.gap_multiplier * cadence_seconds
    )
    gap_indices = np.flatnonzero(delta_seconds > gap_threshold) + 1
    boundaries = np.concatenate(([0], gap_indices, [len(frame)]))

    segments: list[DoorSegment] = []
    for block_number, (start, end) in enumerate(pairwise(boundaries)):
        end_reason = "gap" if block_number < len(boundaries) - 2 else "end_of_stream"
        segments.extend(
            _segment_block(
                frame,
                timestamps,
                int(start),
                int(end),
                cadence_seconds,
                config,
                end_reason,
            )
        )
    if not segments:
        raise DoorInputError("No door operations were detected in the stream.")
    return segments


def _add_signal_features(
    output: dict[str, float], name: str, values: np.ndarray, resample_points: int
) -> None:
    values = np.asarray(values, dtype=np.float64)
    differences = np.diff(values)
    quantiles = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
    statistics = {
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "q10": quantiles[0],
        "q25": quantiles[1],
        "median": quantiles[2],
        "q75": quantiles[3],
        "q90": quantiles[4],
        "max": np.max(values),
        "range": np.ptp(values),
        "rms": np.sqrt(np.mean(np.square(values))),
        "mean_abs_difference": np.mean(np.abs(differences))
        if len(differences)
        else 0.0,
        "max_abs_difference": np.max(np.abs(differences)) if len(differences) else 0.0,
        "peak_position": np.argmax(np.abs(values)) / max(len(values) - 1, 1),
    }
    for statistic, value in statistics.items():
        output[f"{name}__{statistic}"] = float(value)

    waveform = values
    if name == "position":
        waveform = (values - np.min(values)) / max(float(np.ptp(values)), 1.0)
    sampled = np.interp(
        np.linspace(0.0, 1.0, resample_points),
        np.linspace(0.0, 1.0, len(waveform)),
        waveform,
    )
    for index, value in enumerate(sampled):
        output[f"{name}__waveform_{index:02d}"] = float(value)


def extract_segment_features(
    frame: pd.DataFrame,
    timestamps: tuple[datetime, ...],
    segment: DoorSegment,
    config: DoorFeatureConfig | None = None,
) -> dict[str, float]:
    """Create one deterministic feature row for a detected operation."""

    config = config or DoorFeatureConfig()
    section = frame.iloc[segment.start : segment.end]
    output: dict[str, float] = {
        "segment__n_samples": float(len(section)),
        "segment__duration_seconds": float(
            (timestamps[segment.end - 1] - timestamps[segment.start]).total_seconds()
        ),
        "segment__is_open": float(segment.operation == "Open"),
    }
    signal_names = {
        "Motor current(mA)": "current",
        "Motor Voltage(10mV)": "voltage",
        "Motor electrodynamic force": "back_emf",
        "Door leaf position": "position",
    }
    for column, name in signal_names.items():
        _add_signal_features(
            output,
            name,
            section[column].to_numpy(dtype=np.float64),
            config.resample_points,
        )
    for column in TIMING_COLUMNS:
        key = (
            column.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(".", "")
        )
        output[f"{key}__median"] = float(section[column].median())
    for column in BINARY_COLUMNS:
        if column == "Door Locked":
            continue
        values = section[column].to_numpy(dtype=np.float64)
        key = column.lower().replace(" ", "_")
        output[f"{key}__mean"] = float(np.mean(values))
        output[f"{key}__start"] = float(values[0])
        output[f"{key}__end"] = float(values[-1])
        output[f"{key}__transitions"] = float(np.count_nonzero(np.diff(values)))

    numeric = np.asarray(list(output.values()), dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise DoorInputError("Door feature extraction produced non-finite values.")
    return output
