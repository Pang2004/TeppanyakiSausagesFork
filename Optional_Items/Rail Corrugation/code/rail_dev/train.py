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
from backend.models.rail.features import FeatureConfig
from backend.models.rail.predict import ARTIFACT_VERSION, RAIL_LABELS
from backend.models.rail.wavelength import (
    FEATURE_SET,
    WAVELENGTH_CONFIG,
    extract_production_features,
)
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .validation import evaluate_nested, fit_grouped, recording_groups, select_candidate


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
        delayed(extract_production_features)(path, config) for path in paths
    )
    return pd.DataFrame(rows)


def _load_or_extract_features(
    paths: list[Path],
    cache_path: Path,
    config: FeatureConfig,
    jobs: int,
    refresh: bool,
    groups: np.ndarray,
) -> pd.DataFrame:
    expected_ids = [path.name for path in paths]
    manifest_path = cache_path.with_suffix(".manifest.json")
    manifest = {
        "feature_set": FEATURE_SET,
        "base_config": config.to_dict(),
        "wavelength_config": WAVELENGTH_CONFIG,
        "recordings": dict(zip(expected_ids, groups.tolist(), strict=True)),
    }
    manifest = json.loads(json.dumps(manifest))
    if (
        cache_path.is_file()
        and manifest_path.is_file()
        and not refresh
        and json.loads(manifest_path.read_text()) == manifest
    ):
        cached = pd.read_csv(cache_path)
        if cached.get("file_id", pd.Series(dtype=str)).tolist() == expected_ids:
            print(f"Using cached features from {cache_path}")
            return cached
        print("Feature cache does not match the training index; rebuilding it.")

    print(f"Extracting features from {len(paths)} Rail files...")
    frame = _extract_dataset(paths, config, jobs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return frame


def _candidate_models(jobs: int) -> dict[str, Any]:
    """Fixed selected feature family; tune only regularization within grouped folds."""
    candidates = {
        "dummy_normal": DummyClassifier(strategy="constant", constant="Normal")
    }
    for c in (0.01, 0.1, 1.0, 10.0):
        candidates[f"balanced_logistic_C_{c}"] = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c, class_weight="balanced", max_iter=5000, random_state=42
            ),
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("app/backend/artifacts/rail_pipeline.joblib"),
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path(
            "Optional_Items/Rail Corrugation/code/outputs/train_features_wavelength.csv"
        ),
    )
    parser.add_argument(
        "--cv-results",
        type=Path,
        default=Path("Optional_Items/Rail Corrugation/code/outputs/cv_results.json"),
    )
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        default=Path(
            "Optional_Items/Rail Corrugation/code/outputs/oof_predictions.csv"
        ),
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--refresh-features", action="store_true")
    args = parser.parse_args()
    if args.jobs == 0 or args.jobs < -1:
        parser.error("--jobs must be -1 or a positive integer")

    config = FeatureConfig()
    paths, labels = _load_training_index(args.data_dir)
    y = np.asarray(labels)
    groups = recording_groups(paths, y)
    features = _load_or_extract_features(
        paths, args.feature_cache, config, args.jobs, args.refresh_features, groups
    )
    feature_names = [column for column in features.columns if column != "file_id"]
    x = features[feature_names].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Training features contain non-finite values.")

    candidates = _candidate_models(args.jobs)
    results, predictions = evaluate_nested(
        candidates, x, y, groups, [p.name for p in paths]
    )
    results["feature_set"] = FEATURE_SET
    results["protocol"]["selection_scope"] = (
        "Regularization only; original plus wavelength features fixed after exploratory comparison."
    )
    results["protocol"]["limitations"].append(
        "This feature family was chosen after inspecting exploratory outer scores; nested C tuning does not remove that selection bias."
    )
    results["exploratory_context"] = {
        "feature_selection_macro_f1": 0.6858587709826242,
        "source": "backend/models/rail/METHODOLOGY.md#experiment-comparison",
        "meaning": "Outer score of selecting among six feature sets within inner validation, distinct from this fixed-family result.",
    }
    nested = results["nested_selection"]
    print(
        f"Nested grouped macro F1: {nested['macro_f1_mean']:.4f} "
        f"(fold standard deviation {nested['macro_f1_std']:.4f})"
    )
    selected_model, final_scores = select_candidate(candidates, x, y, groups)
    results["final_selection"] = {
        "selected_model": selected_model,
        "inner_macro_f1": final_scores,
        "note": "Full-data inner scores select the artifact; they are not its performance estimate.",
    }
    estimator = fit_grouped(candidates[selected_model], x, y, groups)
    args.oof_predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.oof_predictions, index=False)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(paths),
        "unique_recording_groups": len(set(groups)),
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
        "feature_set": FEATURE_SET,
        "wavelength_config": WAVELENGTH_CONFIG,
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
