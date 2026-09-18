"""Train and select a Door abnormal-resistance classifier."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import (
    DoorFeatureConfig,
    DoorInputError,
    DoorSegment,
    detect_segments,
    extract_segment_features,
    load_stream,
)
from .metrics import ScoredDoorSegment, official_door_score
from .predict import ARTIFACT_VERSION, DOOR_LABELS


def _load_training_data(
    data_dir: Path, config: DoorFeatureConfig
) -> tuple[np.ndarray, np.ndarray, list[str], list[DoorSegment], tuple[datetime, ...]]:
    frame, timestamps = load_stream(data_dir / "Train.csv")
    segments = detect_segments(frame, timestamps, config)
    answers = pd.read_csv(data_dir / "Train_Segments_Answer.csv", dtype=str)
    required = {"segment_id", "start_time", "end_time", "operation", "status", "n_rows"}
    if set(answers.columns) != required:
        raise DoorInputError(
            "Door answer file does not contain the expected six columns."
        )
    answers["n_rows"] = pd.to_numeric(answers["n_rows"], errors="raise")
    if len(segments) != len(answers):
        raise DoorInputError(
            f"Detected {len(segments)} training segments but found {len(answers)} labels."
        )

    for segment, answer in zip(segments, answers.itertuples(index=False), strict=True):
        actual_start = str(frame.at[segment.start, "Datetime"])
        actual_end = str(frame.at[segment.end - 1, "Datetime"])
        if actual_start != answer.start_time or actual_end != answer.end_time:
            raise DoorInputError(
                f"Detected boundary does not match {answer.segment_id}."
            )
        if segment.end - segment.start != int(answer.n_rows):
            raise DoorInputError(
                f"Detected row count does not match {answer.segment_id}."
            )
        if segment.operation != answer.operation:
            raise DoorInputError(
                f"Detected operation does not match {answer.segment_id}."
            )
    invalid = sorted(set(answers["status"]) - set(DOOR_LABELS))
    if invalid:
        raise DoorInputError(f"Unknown Door labels: {invalid}")

    rows = [
        extract_segment_features(frame, timestamps, segment, config)
        for segment in segments
    ]
    feature_names = list(rows[0])
    matrix = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64
    )
    labels = answers["status"].to_numpy(dtype=str)
    return matrix, labels, feature_names, segments, timestamps


def _candidate_models(jobs: int) -> dict[str, Any]:
    return {
        "dummy_normal": DummyClassifier(strategy="constant", constant="Normal"),
        "balanced_logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=5_000, random_state=42
            ),
        ),
        "balanced_rbf_svm": CalibratedClassifierCV(
            make_pipeline(
                StandardScaler(),
                SVC(class_weight="balanced", C=1.0, gamma="scale", random_state=42),
            ),
            method="sigmoid",
            cv=3,
            ensemble=False,
        ),
        "balanced_extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            class_weight="balanced",
            max_features="sqrt",
            random_state=42,
            n_jobs=jobs,
        ),
    }


def _as_scored_segments(
    indices: np.ndarray,
    labels: np.ndarray,
    segments: list[DoorSegment],
    timestamps: tuple[datetime, ...],
) -> list[ScoredDoorSegment]:
    return [
        ScoredDoorSegment(
            timestamps[segments[index].start],
            timestamps[segments[index].end - 1],
            str(label),
        )
        for index, label in zip(indices, labels, strict=True)
    ]


def _evaluate_model(
    name: str,
    estimator: Any,
    matrix: np.ndarray,
    labels: np.ndarray,
    segments: list[DoorSegment],
    timestamps: tuple[datetime, ...],
) -> dict[str, Any]:
    chronological_scores: list[float] = []
    chronological_macro_f1: list[float] = []
    chronological_abnormal_recall: list[float] = []
    all_indices = np.arange(len(labels))
    for validation_indices in np.array_split(all_indices, 5):
        train_indices = np.setdiff1d(all_indices, validation_indices)
        fold_model = clone(estimator).fit(matrix[train_indices], labels[train_indices])
        predicted = fold_model.predict(matrix[validation_indices])
        truth_segments = _as_scored_segments(
            validation_indices, labels[validation_indices], segments, timestamps
        )
        predicted_segments = _as_scored_segments(
            validation_indices, predicted, segments, timestamps
        )
        chronological_scores.append(
            official_door_score(truth_segments, predicted_segments)
        )
        chronological_macro_f1.append(
            float(f1_score(labels[validation_indices], predicted, average="macro"))
        )
        chronological_abnormal_recall.append(
            float(
                recall_score(
                    labels[validation_indices],
                    predicted,
                    pos_label="Abnormal resistance",
                    zero_division=0,
                )
            )
        )

    repeated_scores: list[float] = []
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    for train_indices, validation_indices in splitter.split(matrix, labels):
        fold_model = clone(estimator).fit(matrix[train_indices], labels[train_indices])
        predicted = fold_model.predict(matrix[validation_indices])
        repeated_scores.append(
            float(f1_score(labels[validation_indices], predicted, average="macro"))
        )

    result = {
        "name": name,
        "official_score_mean": float(np.mean(chronological_scores)),
        "official_score_std": float(np.std(chronological_scores)),
        "chronological_macro_f1_mean": float(np.mean(chronological_macro_f1)),
        "abnormal_recall_mean": float(np.mean(chronological_abnormal_recall)),
        "repeated_stratified_macro_f1_mean": float(np.mean(repeated_scores)),
        "chronological_fold_scores": chronological_scores,
    }
    print(
        f"{name}: official score {result['official_score_mean']:.4f} "
        f"(+/- {result['official_score_std']:.4f})"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--model-out", type=Path, default=Path("backend/artifacts/door_pipeline.joblib")
    )
    parser.add_argument(
        "--cv-results", type=Path, default=Path("outputs/door/cv_results.json")
    )
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs == 0 or args.jobs < -1:
        parser.error("--jobs must be -1 or a positive integer")

    config = DoorFeatureConfig()
    matrix, labels, feature_names, segments, timestamps = _load_training_data(
        args.data_dir, config
    )
    candidates = _candidate_models(args.jobs)
    results = [
        _evaluate_model(name, estimator, matrix, labels, segments, timestamps)
        for name, estimator in candidates.items()
    ]
    eligible = [result for result in results if result["name"] != "dummy_normal"]
    winner = max(eligible, key=lambda result: result["official_score_mean"])
    selected_model = str(winner["name"])
    estimator = candidates[selected_model].fit(matrix, labels)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_segments": len(segments),
        "class_counts": {
            label: int(np.count_nonzero(labels == label)) for label in DOOR_LABELS
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "estimator": estimator,
        "feature_names": feature_names,
        "feature_config": config.to_dict(),
        "labels": list(DOOR_LABELS),
        "selected_model": selected_model,
        "cv_results": results,
        "metadata": metadata,
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_out, compress=3)
    args.cv_results.parent.mkdir(parents=True, exist_ok=True)
    args.cv_results.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Selected {selected_model}; saved model to {args.model_out}")


if __name__ == "__main__":
    main()
