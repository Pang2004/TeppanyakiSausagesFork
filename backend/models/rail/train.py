"""Train and select a Rail corrugation classifier."""

from __future__ import annotations

import argparse
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import FeatureConfig, extract_features
from .predict import ARTIFACT_VERSION, RAIL_LABELS


def _natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _find_labels_file(data_dir: Path) -> Path:
    candidates = sorted(data_dir.glob("*.csv"))
    for path in candidates:
        columns = pd.read_csv(path, nrows=0).columns
        if "label" in columns and ({"filename", "file_id"} & set(columns)):
            return path
    raise FileNotFoundError(f"Could not find a filename/label CSV in {data_dir}.")


def _load_training_index(data_dir: Path) -> tuple[list[Path], list[str]]:
    labels_path = _find_labels_file(data_dir)
    labels_frame = pd.read_csv(labels_path, dtype=str)
    labels_frame.columns = labels_frame.columns.str.strip()
    filename_column = "filename" if "filename" in labels_frame else "file_id"
    labels_frame[filename_column] = labels_frame[filename_column].str.strip()
    labels_frame["label"] = labels_frame["label"].str.strip()
    invalid = sorted(set(labels_frame["label"]) - set(RAIL_LABELS))
    if invalid:
        raise ValueError(f"Unknown Rail labels: {invalid}")
    if labels_frame[filename_column].duplicated().any():
        raise ValueError("Rail labels contain duplicate file IDs.")

    train_dir = data_dir / "Train"
    paths = [train_dir / file_id for file_id in labels_frame[filename_column]]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing labelled Rail file: {missing[0]}")
    pairs = sorted(
        zip(paths, labels_frame["label"], strict=True),
        key=lambda pair: _natural_key(pair[0]),
    )
    return [pair[0] for pair in pairs], [str(pair[1]) for pair in pairs]


def _extract_dataset(
    paths: list[Path], config: FeatureConfig, jobs: int
) -> pd.DataFrame:
    rows = Parallel(n_jobs=jobs, verbose=5)(
        delayed(extract_features)(path, config) for path in paths
    )
    return pd.DataFrame(rows)


def _load_or_extract_features(
    paths: list[Path], cache_path: Path, config: FeatureConfig, jobs: int, refresh: bool
) -> pd.DataFrame:
    expected_ids = [path.name for path in paths]
    if cache_path.is_file() and not refresh:
        cached = pd.read_csv(cache_path)
        if cached.get("file_id", pd.Series(dtype=str)).tolist() == expected_ids:
            print(f"Using cached features from {cache_path}")
            return cached
        print("Feature cache does not match the training index; rebuilding it.")

    print(f"Extracting features from {len(paths)} Rail files...")
    frame = _extract_dataset(paths, config, jobs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


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
                SVC(
                    class_weight="balanced",
                    C=1.0,
                    gamma="scale",
                    random_state=42,
                ),
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


def _evaluate_model(
    name: str, estimator: Any, x: np.ndarray, y: np.ndarray
) -> dict[str, Any]:
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    macro_scores: list[float] = []
    per_class: dict[str, list[float]] = {label: [] for label in RAIL_LABELS}
    total_confusion = np.zeros((len(RAIL_LABELS), len(RAIL_LABELS)), dtype=int)
    for train_indices, validation_indices in splitter.split(x, y):
        fold_model = clone(estimator)
        fold_model.fit(x[train_indices], y[train_indices])
        predicted = fold_model.predict(x[validation_indices])
        macro_scores.append(
            float(
                f1_score(
                    y[validation_indices],
                    predicted,
                    average="macro",
                    labels=RAIL_LABELS,
                )
            )
        )
        class_scores = f1_score(
            y[validation_indices],
            predicted,
            average=None,
            labels=RAIL_LABELS,
            zero_division=0,
        )
        for label, score in zip(RAIL_LABELS, class_scores, strict=True):
            per_class[label].append(float(score))
        total_confusion += confusion_matrix(
            y[validation_indices], predicted, labels=RAIL_LABELS
        )

    result = {
        "name": name,
        "macro_f1_mean": float(np.mean(macro_scores)),
        "macro_f1_std": float(np.std(macro_scores)),
        "per_class_f1_mean": {
            label: float(np.mean(scores)) for label, scores in per_class.items()
        },
        "confusion_matrix": total_confusion.tolist(),
        "labels": list(RAIL_LABELS),
    }
    print(
        f"{name}: macro F1 {result['macro_f1_mean']:.4f} "
        f"(+/- {result['macro_f1_std']:.4f})"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--model-out", type=Path, default=Path("backend/artifacts/rail_pipeline.joblib")
    )
    parser.add_argument(
        "--feature-cache", type=Path, default=Path("outputs/rail/train_features.csv")
    )
    parser.add_argument(
        "--cv-results", type=Path, default=Path("outputs/rail/cv_results.json")
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--refresh-features", action="store_true")
    args = parser.parse_args()
    if args.jobs == 0 or args.jobs < -1:
        parser.error("--jobs must be -1 or a positive integer")

    config = FeatureConfig()
    paths, labels = _load_training_index(args.data_dir)
    features = _load_or_extract_features(
        paths, args.feature_cache, config, args.jobs, args.refresh_features
    )
    feature_names = [column for column in features.columns if column != "file_id"]
    x = features[feature_names].to_numpy(dtype=np.float64)
    y = np.asarray(labels)
    if not np.isfinite(x).all():
        raise ValueError("Training features contain non-finite values.")

    candidates = _candidate_models(args.jobs)
    results = [
        _evaluate_model(name, estimator, x, y) for name, estimator in candidates.items()
    ]
    eligible = [result for result in results if result["name"] != "dummy_normal"]
    winner = max(eligible, key=lambda result: result["macro_f1_mean"])
    selected_model = str(winner["name"])
    estimator = candidates[selected_model]
    estimator.fit(x, y)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(paths),
        "class_counts": {label: labels.count(label) for label in RAIL_LABELS},
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
        "labels": list(RAIL_LABELS),
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
