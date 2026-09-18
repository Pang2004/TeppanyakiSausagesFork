"""Contract and integration tests for the Door pipeline."""

from __future__ import annotations

from datetime import timedelta
from itertools import pairwise
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from backend.models.door.features import (
    DoorFeatureConfig,
    DoorInputError,
    detect_segments,
    extract_segment_features,
    load_stream,
    parse_door_timestamp,
)
from backend.models.door.predict import ARTIFACT_VERSION, DOOR_LABELS, DoorPredictor
from door_dev.metrics import (
    ScoredDoorSegment,
    official_door_score,
    segment_iou,
)
from sklearn.dummy import DummyClassifier

REPOSITORY_ROOT = next(
    parent for parent in Path(__file__).parents if (parent / "PS3").is_dir()
)
DATA_DIR = REPOSITORY_ROOT / "PS3/02_Datasets/Door"


def _format_timestamp(value: pd.Timestamp) -> str:
    return (
        f"{value.year}-{value.month}-{value.day}-{value.hour}-{value.minute}-"
        f"{value.second}-{value.microsecond // 1000}"
    )


def test_timestamp_parser_accepts_native_format() -> None:
    parsed = parse_door_timestamp("2023-7-5-0-0-3-760")
    assert parsed.year == 2023
    assert parsed.microsecond == 760_000


def test_invalid_timestamp_and_schema_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(DoorInputError, match="Invalid Door timestamp"):
        parse_door_timestamp("2023/07/05 00:00:00")

    invalid_stream = tmp_path / "invalid.csv"
    pd.DataFrame({"Datetime": ["2023-7-5-0-0-0-0"]}).to_csv(invalid_stream, index=False)
    with pytest.raises(DoorInputError, match="expected 17 columns"):
        load_stream(invalid_stream)


def test_gap_segmentation_exactly_recovers_training_answers() -> None:
    frame, timestamps = load_stream(DATA_DIR / "Train.csv")
    segments = detect_segments(frame, timestamps)
    answers = pd.read_csv(DATA_DIR / "Train_Segments_Answer.csv")

    assert len(segments) == 110
    assert [
        segment.end - segment.start for segment in segments
    ] == answers.n_rows.tolist()
    assert [segment.operation for segment in segments] == answers.operation.tolist()
    assert [
        frame.at[segment.start, "Datetime"] for segment in segments
    ] == answers.start_time.tolist()
    assert [
        frame.at[segment.end - 1, "Datetime"] for segment in segments
    ] == answers.end_time.tolist()
    assert all(not segment.quality_flags for segment in segments)


def test_test_stream_contains_38_valid_operations() -> None:
    frame, timestamps = load_stream(DATA_DIR / "Test.csv")
    segments = detect_segments(frame, timestamps)
    assert len(segments) == 38
    assert all(left.end <= right.start for left, right in pairwise(segments))
    assert all(not segment.quality_flags for segment in segments)


def test_state_machine_splits_a_continuously_sampled_stream() -> None:
    frame, _ = load_stream(DATA_DIR / "Train.csv")
    answers = pd.read_csv(DATA_DIR / "Train_Segments_Answer.csv")
    first_length, second_length = answers.n_rows.iloc[:2]
    first = frame.iloc[:first_length].copy()
    second = frame.iloc[first_length : first_length + second_length].copy()
    idle = first.iloc[[-1] * 8].copy()
    for column in (
        "Close command",
        "Open command",
        "Door is opening",
        "Door is closing",
    ):
        idle[column] = 0
    continuous = pd.concat([first, idle, second], ignore_index=True)
    start = pd.Timestamp("2023-07-05")
    continuous["Datetime"] = [
        _format_timestamp(start + timedelta(milliseconds=20 * index))
        for index in range(len(continuous))
    ]
    timestamps = tuple(parse_door_timestamp(value) for value in continuous["Datetime"])

    segments = detect_segments(continuous, timestamps)
    assert len(segments) == 2
    assert [segment.operation for segment in segments] == ["Close", "Open"]
    assert segments[0].end == first_length
    assert segments[1].start == first_length + len(idle)


def test_features_are_finite_and_deterministic() -> None:
    frame, timestamps = load_stream(DATA_DIR / "Train.csv")
    segment = detect_segments(frame, timestamps)[0]
    first = extract_segment_features(frame, timestamps, segment)
    second = extract_segment_features(frame, timestamps, segment)
    assert first == second
    assert "door_locked__mean" not in first
    assert np.isfinite(np.asarray(list(first.values()))).all()


def test_official_scorer_penalizes_boundaries_and_labels() -> None:
    start = parse_door_timestamp("2023-7-5-0-0-0-0")
    end = start + timedelta(seconds=4)
    truth = [ScoredDoorSegment(start, end, "Normal")]
    exact = [ScoredDoorSegment(start, end, "Normal")]
    short = [ScoredDoorSegment(start + timedelta(seconds=1), end, "Normal")]
    wrong = [ScoredDoorSegment(start, end, "Abnormal resistance")]
    assert official_door_score(truth, exact) == 1.0
    assert segment_iou(truth[0], short[0]) == 0.75
    assert official_door_score(truth, short) == 0.75
    assert official_door_score(truth, wrong) == 0.0
    extra = [
        *exact,
        ScoredDoorSegment(
            end + timedelta(seconds=1), end + timedelta(seconds=2), "Normal"
        ),
    ]
    assert np.isclose(official_door_score(truth, extra), 2 / 3)


def test_saved_artifact_predicts_complete_stream(tmp_path: Path) -> None:
    frame, timestamps = load_stream(DATA_DIR / "Train.csv")
    segments = detect_segments(frame, timestamps)
    rows = [
        extract_segment_features(frame, timestamps, segment) for segment in segments[:2]
    ]
    feature_names = list(rows[0])
    matrix = np.asarray([[row[name] for name in feature_names] for row in rows])
    estimator = DummyClassifier(strategy="prior").fit(matrix, np.asarray(DOOR_LABELS))
    artifact_path = tmp_path / "door.joblib"
    joblib.dump(
        {
            "artifact_version": ARTIFACT_VERSION,
            "estimator": estimator,
            "feature_names": feature_names,
            "feature_config": DoorFeatureConfig().to_dict(),
            "labels": list(DOOR_LABELS),
            "metadata": {"purpose": "test"},
        },
        artifact_path,
    )

    results = DoorPredictor.from_artifact(artifact_path).predict_stream(
        DATA_DIR / "Test.csv"
    )
    assert len(results) == 38
    assert all(result.prediction in DOOR_LABELS for result in results)
    assert all(0 <= result.confidence <= 1 for result in results)
