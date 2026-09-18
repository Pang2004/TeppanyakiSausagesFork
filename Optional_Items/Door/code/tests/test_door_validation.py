"""Protect the independence and temporal ordering of Door validation."""

from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from backend.models.door.features import DoorFeatureConfig, load_stream
from door_dev import validation
from sklearn.dummy import DummyClassifier

DATA_DIR = (
    next(parent for parent in Path(__file__).parents if (parent / "PS3").is_dir())
    / "PS3/02_Datasets/Door"
)


@pytest.fixture(scope="module")
def reference_data():
    frame, timestamps = load_stream(DATA_DIR / "Train.csv")
    segments, truth = validation.load_reference(DATA_DIR, frame, timestamps)
    return frame, timestamps, segments, truth


def test_reference_loading_does_not_use_detector(monkeypatch, reference_data):
    frame, timestamps, _, expected = reference_data

    def forbidden(*args, **kwargs):
        raise AssertionError("Reference annotations must not come from the detector")

    monkeypatch.setattr(validation, "detect_segments", forbidden)
    segments, truth = validation.load_reference(DATA_DIR, frame, timestamps)
    assert truth == expected
    assert len(segments) == 110


def test_forward_evaluation_penalizes_wrong_detected_boundaries(
    monkeypatch, reference_data
):
    frame, timestamps, segments, truth = reference_data
    original_detector = validation.detect_segments

    def shortened(*args, **kwargs):
        return [
            replace(s, end=s.start + (s.end - s.start) // 2)
            for s in original_detector(*args, **kwargs)
        ]

    monkeypatch.setattr(validation, "detect_segments", shortened)
    result = validation.evaluate_forward(
        "dummy",
        DummyClassifier(strategy="constant", constant="Normal"),
        frame,
        timestamps,
        segments,
        truth,
        DoorFeatureConfig(),
    )
    assert 0 < result["localization_score_mean"] < 0.51
    assert 0 < result["official_score_mean"] < result["localization_score_mean"]
    seen = []
    for fold in result["folds"]:
        assert max(fold["train_indices"]) < min(fold["validation_indices"])
        assert fold["training_end"] < fold["validation_start"]
        assert fold["truth"] == [
            validation.asdict(truth[i]) for i in fold["validation_indices"]
        ]
        seen.extend(fold["validation_indices"])
    assert seen == list(range(20, 110))


def test_missing_samples_preserve_answer_boundaries(reference_data):
    frame, timestamps, segments, truth = reference_data
    stop = segments[0].end
    for scenario in validation.SCENARIOS:
        raw, times, expected = validation.perturb_stream(
            frame.iloc[:stop], timestamps[:stop], truth[:1], scenario
        )
        assert expected == truth[:1]
        assert len(raw) == len(times)
        assert all(a < b for a, b in pairwise(times))
        if scenario == "midpoint_200ms_dropout":
            assert len(raw) == stop - 10
            assert times[0] == timestamps[0] and times[-1] == timestamps[stop - 1]
        if scenario == "last_10_percent_missing":
            assert times[-1] < expected[0].end_time


def test_no_detected_operations_are_scored_as_misses(reference_data):
    frame, timestamps, segments, truth = reference_data
    raw = frame.iloc[: segments[0].end].copy()
    raw[["Open command", "Close command", "Door is opening", "Door is closing"]] = 0
    estimator = DummyClassifier().fit(np.zeros((2, 1)), ["Normal", "Normal"])
    predictions = validation.predict_segments(
        estimator, raw, timestamps[: len(raw)], DoorFeatureConfig(), ["unused"]
    )
    assert predictions == []
    assert validation.score_predictions(truth[:1], predictions)["official_score"] == 0
