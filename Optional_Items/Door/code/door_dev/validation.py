"""Forward validation of raw Door streams against independent answer boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from backend.models.door.features import (
    DoorFeatureConfig,
    DoorInputError,
    DoorSegment,
    _infer_operation,
    detect_segments,
    extract_segment_features,
    load_stream,
    parse_door_timestamp,
)
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from .metrics import ScoredDoorSegment, official_door_score

SCENARIOS = (
    "original",
    "isolated_missing_samples",
    "midpoint_200ms_dropout",
    "midpoint_100ms_inactive_flags",
    "last_10_percent_missing",
    "compressed_operation_gaps",
)


def load_reference(data_dir, frame, timestamps):
    """Map annotations to rows without consulting the segment detector."""
    answers = pd.read_csv(data_dir / "Train_Segments_Answer.csv", dtype=str)
    required = {"segment_id", "start_time", "end_time", "operation", "status", "n_rows"}
    if set(answers.columns) != required or answers.empty:
        raise DoorInputError(
            "Door answer file must contain the expected six columns and labels."
        )
    lookup = {value: index for index, value in enumerate(timestamps)}
    segments, truth = [], []
    for answer in answers.itertuples(index=False):
        start = parse_door_timestamp(answer.start_time)
        end = parse_door_timestamp(answer.end_time)
        if start not in lookup or end not in lookup or end <= start:
            raise DoorInputError(f"Invalid answer boundaries: {answer.segment_id}")
        first, stop = lookup[start], lookup[end] + 1
        if stop - first != int(answer.n_rows) or (
            segments and first < segments[-1].end
        ):
            raise DoorInputError(
                f"Invalid answer rows or ordering: {answer.segment_id}"
            )
        if answer.status not in ("Normal", "Abnormal resistance"):
            raise DoorInputError(f"Unknown Door label: {answer.status}")
        if answer.operation not in ("Open", "Close"):
            raise DoorInputError(f"Unknown Door operation: {answer.operation}")
        # Direction is a model input and is inferred from telemetry, not the answer.
        segments.append(
            DoorSegment(
                first, stop, _infer_operation(frame, first, stop), "reference", ()
            )
        )
        truth.append(ScoredDoorSegment(start, end, answer.status))
    return segments, truth


def feature_matrix(frame, timestamps, segments, config, feature_names=None):
    rows = [
        extract_segment_features(frame, timestamps, segment, config)
        for segment in segments
    ]
    names = feature_names if feature_names is not None else list(rows[0])
    return np.asarray(
        [[row[name] for name in names] for row in rows], dtype=float
    ), names


def perturb_stream(frame, timestamps, truth, scenario):
    """Fixed diagnostic corruptions; labels never enter the detector or classifier.

    Reference intervals locate corruptions only. Except for the explicit time
    translation scenario, reference boundaries remain the original annotations.
    """
    frame = frame.copy().reset_index(drop=True)
    timestamps = list(timestamps)
    keep = np.ones(len(frame), dtype=bool)
    shifted_truth = []
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    for expected in truth:
        indices = np.asarray(
            [
                i
                for i, t in enumerate(timestamps)
                if expected.start_time <= t <= expected.end_time
            ]
        )
        first, stop = int(indices[0]), int(indices[-1]) + 1
        middle = (first + stop) // 2
        if scenario == "isolated_missing_samples":
            keep[first + 10 : stop - 1 : 10] = False
        elif scenario == "midpoint_200ms_dropout":
            keep[middle - 5 : middle + 5] = False
        elif scenario == "midpoint_100ms_inactive_flags":
            frame.loc[
                middle - 2 : middle + 2,
                ["Open command", "Close command", "Door is opening", "Door is closing"],
            ] = 0
        elif scenario == "last_10_percent_missing":
            keep[stop - max(1, len(indices) // 10) : stop] = False
        shifted_truth.append(expected)
    if scenario == "compressed_operation_gaps":
        # Re-time complete recorded operations with 20 ms between them. No idle
        # samples are invented. Same-direction adjacent cycles may be ambiguous.
        frames, times, shifted_truth = [], [], []
        next_start = truth[0].start_time
        for expected in truth:
            indices = [
                i
                for i, t in enumerate(timestamps)
                if expected.start_time <= t <= expected.end_time
            ]
            offset = next_start - expected.start_time
            frames.append(frame.iloc[indices])
            times.extend(timestamps[i] + offset for i in indices)
            shifted_truth.append(
                replace(
                    expected,
                    start_time=expected.start_time + offset,
                    end_time=expected.end_time + offset,
                )
            )
            next_start = shifted_truth[-1].end_time + timedelta(milliseconds=20)
        frame, timestamps = pd.concat(frames, ignore_index=True), times
    else:
        frame = frame.loc[keep].reset_index(drop=True)
        timestamps = [
            t for t, retained in zip(timestamps, keep, strict=True) if retained
        ]
    frame["Datetime"] = [
        f"{t.year}-{t.month}-{t.day}-{t.hour}-{t.minute}-{t.second}-{t.microsecond // 1000}"
        for t in timestamps
    ]
    return frame, tuple(timestamps), shifted_truth


def predict_segments(estimator, frame, timestamps, config, feature_names):
    try:
        segments = detect_segments(frame, timestamps, config)
    except DoorInputError as exc:
        if str(exc) != "No door operations were detected in the stream.":
            raise
        return []
    matrix, _ = feature_matrix(frame, timestamps, segments, config, feature_names)
    labels = estimator.predict(matrix)
    return [
        ScoredDoorSegment(timestamps[s.start], timestamps[s.end - 1], str(label))
        for s, label in zip(segments, labels, strict=True)
    ]


def score_predictions(truth, predicted):
    abnormal = "Abnormal resistance"
    return {
        "official_score": official_door_score(truth, predicted),
        "localization_score": official_door_score(
            [replace(s, label="operation") for s in truth],
            [replace(s, label="operation") for s in predicted],
        ),
        "abnormal_official_score": official_door_score(
            [s for s in truth if s.label == abnormal],
            [s for s in predicted if s.label == abnormal],
        ),
        "truth_count": len(truth),
        "prediction_count": len(predicted),
    }


def evaluate_forward(
    name, estimator, frame, timestamps, segments, truth, config, scenarios=("original",)
):
    matrix, names = feature_matrix(frame, timestamps, segments, config)
    labels = np.asarray([s.label for s in truth])
    folds = []
    for fold, (train, validation) in enumerate(
        TimeSeriesSplit(n_splits=5).split(matrix), 1
    ):
        model = clone(estimator).fit(matrix[train], labels[train])
        first, stop = segments[validation[0]].start, segments[validation[-1]].end
        expected = [truth[i] for i in validation]
        assert truth[train[-1]].end_time < expected[0].start_time
        for scenario in scenarios:
            raw, times, references = perturb_stream(
                frame.iloc[first:stop], timestamps[first:stop], expected, scenario
            )
            predictions = predict_segments(model, raw, times, config, names)
            folds.append(
                {
                    "fold": fold,
                    "scenario": scenario,
                    "training_count": len(train),
                    "train_indices": train.tolist(),
                    "validation_indices": validation.tolist(),
                    "training_end": truth[train[-1]].end_time.isoformat(),
                    "validation_start": expected[0].start_time.isoformat(),
                    **score_predictions(references, predictions),
                    "truth": [asdict(s) for s in references],
                    "predictions": [asdict(s) for s in predictions],
                }
            )
    summary = {}
    for scenario in scenarios:
        rows = [row for row in folds if row["scenario"] == scenario]
        summary[scenario] = {
            **{
                key + "_mean": float(np.mean([row[key] for row in rows]))
                for key in (
                    "official_score",
                    "localization_score",
                    "abnormal_official_score",
                )
            },
            "official_score_std": float(
                np.std([row["official_score"] for row in rows])
            ),
            "truth_count": sum(row["truth_count"] for row in rows),
            "prediction_count": sum(row["prediction_count"] for row in rows),
        }
    return {
        "name": name,
        "protocol": "expanding_window_raw_stream_5_folds",
        **summary["original"],
        "scenarios": summary,
        "folds": folds,
    }


def main():
    from .train import _candidate_models

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Optional_Items/Door/code/outputs/validation_results.json"),
    )
    args = parser.parse_args()
    config = DoorFeatureConfig()
    frame, timestamps = load_stream(args.data_dir / "Train.csv")
    segments, truth = load_reference(args.data_dir, frame, timestamps)
    result = evaluate_forward(
        "balanced_logistic_regression",
        _candidate_models(1)["balanced_logistic_regression"],
        frame,
        timestamps,
        segments,
        truth,
        config,
        SCENARIOS,
    )
    result["input_sha256"] = {
        name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
        for name in ("Train.csv", "Train_Segments_Answer.csv")
    }
    result["feature_config"] = config.to_dict()
    result["limitations"] = [
        "Fixed incumbent classifier previously selected using this same labelled stream.",
        "First 20 operations train the initial fold; 90 later operations are evaluated once each.",
        "Synthetic stresses are diagnostics, not estimates of their deployment frequency.",
        "One recording; new physical doors and operating conditions remain untested.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result["scenarios"], indent=2))


if __name__ == "__main__":
    main()
