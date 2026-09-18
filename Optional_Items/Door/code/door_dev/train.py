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
from backend.models.door.features import (
    DoorFeatureConfig,
    DoorSegment,
    load_stream,
)
from backend.models.door.predict import ARTIFACT_VERSION, DOOR_LABELS
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .validation import evaluate_forward, feature_matrix, load_reference


def _load_training_data(
    data_dir: Path, config: DoorFeatureConfig
) -> tuple[np.ndarray, np.ndarray, list[str], list[DoorSegment], tuple[datetime, ...]]:
    frame, timestamps = load_stream(data_dir / "Train.csv")
    segments, truth = load_reference(data_dir, frame, timestamps)
    matrix, feature_names = feature_matrix(frame, timestamps, segments, config)
    labels = np.asarray([segment.label for segment in truth])
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("app/backend/artifacts/door_pipeline.joblib"),
    )
    parser.add_argument(
        "--cv-results",
        type=Path,
        default=Path("Optional_Items/Door/code/outputs/cv_results.json"),
    )
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs == 0 or args.jobs < -1:
        parser.error("--jobs must be -1 or a positive integer")

    config = DoorFeatureConfig()
    matrix, labels, feature_names, segments, timestamps = _load_training_data(
        args.data_dir, config
    )
    frame, timestamps = load_stream(args.data_dir / "Train.csv")
    segments, truth = load_reference(args.data_dir, frame, timestamps)
    candidates = _candidate_models(args.jobs)
    results = [
        evaluate_forward(name, estimator, frame, timestamps, segments, truth, config)
        for name, estimator in candidates.items()
    ]
    for result in results:
        print(
            f"{result['name']}: forward official score {result['official_score_mean']:.4f}"
        )
    eligible = [result for result in results if result["name"] != "dummy_normal"]
    winner = max(eligible, key=lambda result: result["official_score_mean"])
    selected_model = str(winner["name"])
    estimator = candidates[selected_model].fit(matrix, labels)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_segments": len(segments),
        "validation_protocol": "expanding_window_raw_stream_5_folds",
        "validation_limit": "Candidate comparison on one previously used stream; not nested model selection.",
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
    args.cv_results.write_text(
        json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Selected {selected_model}; saved model to {args.model_out}")


if __name__ == "__main__":
    main()
