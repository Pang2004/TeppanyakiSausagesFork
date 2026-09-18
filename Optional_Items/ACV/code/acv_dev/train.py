"""Validate and package the physics-informed ACV ranking model."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import openpyxl
import pandas as pd
from backend.models.acv.features import (
    ACVFeatureConfig,
    extract_car_features,
    rank_feature_frame,
)
from backend.models.acv.predict import ARTIFACT_VERSION

from .metrics import rank_decay_score


def _load_index(data_dir: Path) -> list[tuple[Path, str]]:
    labels = pd.read_csv(data_dir / "Train_Labels.csv", dtype="string")
    if list(labels.columns) != ["filename", "faulty_car"]:
        raise ValueError(
            "ACV labels must contain filename,faulty_car columns in order."
        )
    if labels["filename"].duplicated().any():
        raise ValueError("ACV labels contain duplicate filenames.")
    index: list[tuple[Path, str]] = []
    for row in labels.itertuples(index=False):
        path = data_dir / "Train" / str(row.filename)
        if not path.is_file():
            raise FileNotFoundError(f"Missing labelled ACV file: {path.name}")
        car_id = str(row.faulty_car).zfill(2)
        index.append((path, car_id))
    return index


def _rank_or_default(features: pd.DataFrame, score_column: str) -> list[str]:
    if score_column not in features or features[score_column].notna().sum() == 0:
        return sorted(features["car_id"].astype(str).tolist())
    return rank_feature_frame(features, score_column)


def _evaluate(
    index: list[tuple[Path, str]],
    config: ACVFeatureConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    feature_frames: list[pd.DataFrame] = []
    cases: list[dict[str, object]] = []
    for path, faulty_car in index:
        print(f"Extracting ACV features from {path.name}...")
        features = extract_car_features(path, config)
        features.insert(3, "faulty_car", faulty_car)
        features.insert(4, "is_faulty", features["car_id"].eq(faulty_car))
        feature_frames.append(features)

        rankings = {
            "fixed_order": sorted(features["car_id"].astype(str).tolist()),
            "hottest_median": _rank_or_default(features, "cabin_temperature_median"),
            "temperature": _rank_or_default(features, "temperature_score"),
            "hybrid": rank_feature_frame(features),
        }
        case_result: dict[str, object] = {
            "file_id": path.name,
            "faulty_car": faulty_car,
            "schema": str(features["schema"].iloc[0]),
        }
        for name, ranking in rankings.items():
            case_result[f"{name}_ranking"] = ranking
            case_result[f"{name}_score"] = rank_decay_score(ranking, faulty_car)
            case_result[f"{name}_rank"] = ranking.index(faulty_car) + 1
        cases.append(case_result)

    all_features = pd.concat(feature_frames, ignore_index=True)
    model_names = ("fixed_order", "hottest_median", "temperature", "hybrid")
    summary: dict[str, object] = {
        "validation_protocol": "file-level evaluation of a fixed unsupervised ranker",
        "case_count": len(cases),
        "cases": cases,
    }
    for name in model_names:
        scores = [float(case[f"{name}_score"]) for case in cases]
        ranks = [int(case[f"{name}_rank"]) for case in cases]
        summary[name] = {
            "mean_rank_decay_score": float(np.mean(scores)),
            "top_1_accuracy": float(np.mean(np.asarray(ranks) == 1)),
            "mean_true_car_rank": float(np.mean(ranks)),
        }
    summary["selected_model"] = "hybrid_temperature_pressure_ranker"
    summary["rich_schema_case_count"] = sum(
        case["schema"] == "rich_pressure" for case in cases
    )
    return all_features, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("app/backend/artifacts/acv_pipeline.joblib"),
    )
    parser.add_argument(
        "--features-output",
        type=Path,
        default=Path("Optional_Items/ACV/code/outputs/train_features.csv"),
    )
    parser.add_argument(
        "--cv-results",
        type=Path,
        default=Path("Optional_Items/ACV/code/outputs/cv_results.json"),
    )
    args = parser.parse_args()

    config = ACVFeatureConfig()
    index = _load_index(args.data_dir)
    features, validation = _evaluate(index, config)
    print(json.dumps(validation, indent=2))

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_files": len(index),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "openpyxl": openpyxl.__version__,
        "joblib": joblib.__version__,
    }
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "feature_config": config.to_dict(),
        "validation": validation,
        "metadata": metadata,
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_out, compress=3)
    args.features_output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.features_output, index=False)
    args.cv_results.parent.mkdir(parents=True, exist_ok=True)
    args.cv_results.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(f"Saved ACV ranking model to {args.model_out}")


if __name__ == "__main__":
    main()
