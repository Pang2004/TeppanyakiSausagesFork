"""Inference adapter and command-line interface for SHM fatigue damage."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .features import SHMFeatureConfig, extract_features
from .types import SHMPrediction

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts/shm_pipeline.joblib"
)

ARTIFACT_VERSION = "shm-pipeline-v1"


def _natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


class SHMPredictor:
    """Load a calibrated Miner model and predict SHM stress files."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        if artifact.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported SHM artifact version: {artifact.get('artifact_version')!r}."
            )
        required = {
            "feature_config",
            "exponent",
            "scale",
            "residual_estimator",
            "residual_feature_names",
            "validation",
        }
        missing = required.difference(artifact)
        if missing:
            raise ValueError(f"SHM artifact is missing: {', '.join(sorted(missing))}.")
        self.config = SHMFeatureConfig.from_dict(artifact["feature_config"])
        self.exponent = int(artifact["exponent"])
        self.scale = float(artifact["scale"])
        self.residual_estimator = artifact["residual_estimator"]
        self.residual_feature_names = tuple(artifact["residual_feature_names"])
        self.validation = artifact["validation"]
        self.metadata = artifact.get("metadata", {})

    @classmethod
    def from_artifact(cls, path: str | Path) -> SHMPredictor:
        """Load and validate a joblib artifact."""

        source = Path(path)
        try:
            artifact = joblib.load(source)
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"Could not load SHM model from {source}: {exc}") from exc
        if not isinstance(artifact, dict):
            raise TypeError("SHM artifact must contain a dictionary.")
        return cls(artifact)

    def predict_file(self, path: str | Path) -> SHMPrediction:
        """Predict cumulative damage for one headerless stress history."""

        row = extract_features(path, self.config)
        proxy = float(row[f"rainflow__miner_proxy_m{self.exponent}"])
        prediction = self.scale * proxy
        if self.residual_estimator is not None and proxy > 0:
            missing = [name for name in self.residual_feature_names if name not in row]
            if missing:
                raise ValueError(f"Extracted SHM features are missing {missing[0]!r}.")
            matrix = np.asarray(
                [[float(row[name]) for name in self.residual_feature_names]],
                dtype=np.float64,
            )
            prediction *= float(np.exp(self.residual_estimator.predict(matrix)[0]))
        prediction = max(0.0, float(prediction))
        return SHMPrediction(
            file_id=Path(path).name,
            prediction=prediction,
            cycle_count=float(row["rainflow__cycle_count"]),
            equivalent_stress_amplitude=float(
                row[f"rainflow__equivalent_amplitude_m{self.exponent}"]
            ),
            maximum_cycle_range=float(row["rainflow__maximum_range"]),
            estimated_percentage_error=float(
                self.validation["p95_absolute_percentage_error"]
            ),
        )


def predict_file(
    path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> SHMPrediction:
    """Convenience adapter used by the API integration layer."""

    return SHMPredictor.from_artifact(model_path).predict_file(path)


def _input_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"SHM input does not exist: {source}")
    files = sorted(source.glob("*.csv"), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {source}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="SHM CSV file or directory"
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

    predictor = SHMPredictor.from_artifact(args.model)
    predictions = [predictor.predict_file(path) for path in _input_files(args.input)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file_id", "prediction"])
        writer.writeheader()
        writer.writerows(
            {"file_id": result.file_id, "prediction": result.prediction}
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
