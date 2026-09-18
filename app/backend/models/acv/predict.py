"""Inference adapter and command-line interface for ACV localisation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .features import ACVFeatureConfig, extract_car_features, rank_feature_frame
from .types import ACVPrediction

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts/acv_pipeline.joblib"
)

ARTIFACT_VERSION = "acv-pipeline-v1"

DIAGNOSTIC_COLUMNS = (
    "combined_score",
    "temperature_score",
    "pressure_score",
    "temperature_coverage",
    "cabin_temperature_median",
    "temperature_relative_mean",
    "temperature_relative_q90",
    "temperature_fraction_above_1_00",
    "pressure_mismatch",
    "raw_zero_temperature_fraction",
)


def _natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _finite_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


class ACVPredictor:
    """Load a versioned ACV ranking configuration and rank one workbook."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        if artifact.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported ACV artifact version: {artifact.get('artifact_version')!r}."
            )
        required = {"feature_config", "validation"}
        missing = required.difference(artifact)
        if missing:
            raise ValueError(f"ACV artifact is missing: {', '.join(sorted(missing))}.")
        self.config = ACVFeatureConfig.from_dict(artifact["feature_config"])
        self.validation = artifact["validation"]
        self.metadata = artifact.get("metadata", {})

    @classmethod
    def from_artifact(cls, path: str | Path) -> ACVPredictor:
        """Load and validate a joblib artifact."""

        source = Path(path)
        try:
            artifact = joblib.load(source)
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"Could not load ACV model from {source}: {exc}") from exc
        if not isinstance(artifact, dict):
            raise TypeError("ACV artifact must contain a dictionary.")
        return cls(artifact)

    def predict_file(self, path: str | Path) -> ACVPrediction:
        """Rank every car in one ACV workbook."""

        features = extract_car_features(path, self.config)
        ranked_cars = tuple(rank_feature_frame(features))
        diagnostics: dict[str, dict[str, float | None]] = {}
        car_scores: dict[str, float] = {}
        for row in features.to_dict(orient="records"):
            car_id = str(row["car_id"])
            score = _finite_or_none(row.get("combined_score"))
            car_scores[car_id] = 0.0 if score is None else score
            diagnostics[car_id] = {
                name: _finite_or_none(row.get(name)) for name in DIAGNOSTIC_COLUMNS
            }
        return ACVPrediction(
            file_id=Path(path).name,
            ranked_cars=ranked_cars,
            car_scores=car_scores,
            diagnostics=diagnostics,
            schema=str(features["schema"].iloc[0]),
        )


def predict_file(
    path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> ACVPrediction:
    """Convenience adapter used by the API integration layer."""

    return ACVPredictor.from_artifact(model_path).predict_file(path)


def _input_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"ACV input does not exist: {source}")
    files = sorted(source.glob("*.xlsx"), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No XLSX files found in {source}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="ACV XLSX file or directory"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--output", required=True, type=Path, help="Submission CSV path"
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        help="Optional JSON file with app diagnostics",
    )
    args = parser.parse_args()

    predictor = ACVPredictor.from_artifact(args.model)
    predictions = [predictor.predict_file(path) for path in _input_files(args.input)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file_id", "ranked_cars"])
        writer.writeheader()
        writer.writerows(
            {
                "file_id": result.file_id,
                "ranked_cars": "|".join(result.ranked_cars),
            }
            for result in predictions
        )
    if args.diagnostics_output:
        args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_output.write_text(
            json.dumps([result.to_dict() for result in predictions], indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
