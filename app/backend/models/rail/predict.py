"""Inference adapter and command-line interface for Rail corrugation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np

from .features import FeatureConfig, extract_features
from .local import CONFIG as LOCAL_CONFIG
from .local import FEATURE_SET as LOCAL_FEATURE_SET
from .local import extract_combined
from .types import RailLabel, RailPrediction
from .wavelength import FEATURE_SET, WAVELENGTH_CONFIG, extract_production_features

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts/rail_pipeline.joblib"
)

ARTIFACT_VERSION = "rail-pipeline-v2"
LOCAL_ARTIFACT_VERSION = "rail-pipeline-v3"
LEGACY_ARTIFACT_VERSION = "rail-pipeline-v1"
RAIL_LABELS: tuple[RailLabel, ...] = ("Normal", "Side I", "Side II")


def _natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


class RailPredictor:
    """Load one fitted artifact and predict validated Rail CSV files."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        if artifact.get("artifact_version") not in (
            ARTIFACT_VERSION,
            LEGACY_ARTIFACT_VERSION,
            LOCAL_ARTIFACT_VERSION,
        ):
            raise ValueError(
                f"Unsupported Rail artifact version: {artifact.get('artifact_version')!r}."
            )
        required = {"estimator", "feature_names", "feature_config", "labels"}
        missing = required.difference(artifact)
        if missing:
            raise ValueError(f"Rail artifact is missing: {', '.join(sorted(missing))}.")
        if tuple(artifact["labels"]) != RAIL_LABELS:
            raise ValueError("Rail artifact contains an unexpected label schema.")

        self.feature_set = artifact.get("feature_set", "base_v1")
        self.artifact_version = artifact["artifact_version"]
        if self.artifact_version == LOCAL_ARTIFACT_VERSION:
            import json

            if (
                self.feature_set != LOCAL_FEATURE_SET
                or json.loads(json.dumps(artifact.get("local_config")))
                != json.loads(json.dumps(LOCAL_CONFIG))
                or artifact.get("wavelength_config") != WAVELENGTH_CONFIG
                or FeatureConfig.from_dict(artifact["feature_config"])
                != FeatureConfig()
            ):
                raise ValueError(
                    "Unsupported or missing Rail local feature configuration."
                )
        elif artifact["artifact_version"] == ARTIFACT_VERSION:
            if (
                self.feature_set != FEATURE_SET
                or artifact.get("wavelength_config") != WAVELENGTH_CONFIG
            ):
                raise ValueError(
                    "Unsupported or missing Rail wavelength feature configuration."
                )
        elif self.feature_set != "base_v1":
            raise ValueError("Legacy Rail artifacts support only base features.")
        self.estimator = artifact["estimator"]
        self.feature_names = tuple(str(name) for name in artifact["feature_names"])
        self.config = FeatureConfig.from_dict(artifact["feature_config"])
        self.metadata = artifact.get("metadata", {})

    @classmethod
    def from_artifact(cls, path: str | Path) -> RailPredictor:
        """Load and validate a joblib artifact."""

        source = Path(path)
        try:
            artifact = joblib.load(source)
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"Could not load Rail model from {source}: {exc}") from exc
        if not isinstance(artifact, dict):
            raise TypeError("Rail artifact must contain a dictionary.")
        return cls(artifact)

    def predict_file(self, path: str | Path) -> RailPrediction:
        """Predict one file and return class scores plus GUI diagnostics."""

        feature_row = (
            extract_combined(path)
            if self.feature_set == LOCAL_FEATURE_SET
            else (
                extract_production_features(path, self.config)
                if self.feature_set == FEATURE_SET
                else extract_features(path, self.config)
            )
        )
        missing = [name for name in self.feature_names if name not in feature_row]
        if missing:
            raise ValueError(f"Extracted Rail features are missing {missing[0]!r}.")
        matrix = np.asarray(
            [[float(feature_row[name]) for name in self.feature_names]],
            dtype=np.float64,
        )
        raw_prediction = str(self.estimator.predict(matrix)[0])
        if raw_prediction not in RAIL_LABELS:
            raise ValueError(
                f"Model returned an invalid Rail label: {raw_prediction!r}."
            )

        scores = {label: 0.0 for label in RAIL_LABELS}
        if hasattr(self.estimator, "predict_proba"):
            probabilities = self.estimator.predict_proba(matrix)[0]
            for label, score in zip(
                self.estimator.classes_, probabilities, strict=True
            ):
                if str(label) in scores:
                    scores[cast(RailLabel, str(label))] = float(score)

        side_energy: dict[str, dict[str, float]] = {}
        dominant_frequency: dict[str, dict[str, float]] = {}
        for side in ("i", "ii"):
            display_side = f"Side {side.upper()}"
            side_energy[display_side] = {}
            dominant_frequency[display_side] = {}
            for signal_type in ("vibration", "shock"):
                prefix = f"side_{side}_{signal_type}"
                side_energy[display_side][signal_type] = float(
                    feature_row[f"{prefix}__rms__mean"]
                )
                dominant_frequency[display_side][signal_type] = float(
                    feature_row[f"{prefix}__dominant_frequency__mean"]
                )

        return RailPrediction(
            file_id=Path(path).name,
            prediction=cast(RailLabel, raw_prediction),
            scores=scores,
            side_energy=side_energy,
            dominant_frequency=dominant_frequency,
        )


def predict_file(
    path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> RailPrediction:
    """Convenience adapter used by the API integration layer."""

    return RailPredictor.from_artifact(model_path).predict_file(path)


def _input_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Rail input does not exist: {source}")
    files = sorted(source.glob("*.csv"), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {source}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="Rail CSV file or directory"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--output", required=True, type=Path, help="Submission CSV path"
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        help="Optional JSON file with GUI diagnostics",
    )
    args = parser.parse_args()

    predictor = RailPredictor.from_artifact(args.model)
    predictions = [predictor.predict_file(path) for path in _input_files(args.input)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["file_id", "prediction"], lineterminator="\n"
        )
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
