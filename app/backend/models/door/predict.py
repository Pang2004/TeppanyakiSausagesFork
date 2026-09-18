"""Inference adapter and command-line interface for Door fault detection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np

from .features import (
    DoorFeatureConfig,
    detect_segments,
    extract_segment_features,
    load_stream,
)
from .types import DoorLabel, DoorPrediction

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts/door_pipeline.joblib"
)

ARTIFACT_VERSION = "door-pipeline-v1"
DOOR_LABELS: tuple[DoorLabel, ...] = ("Normal", "Abnormal resistance")


class DoorPredictor:
    """Load one fitted artifact and process complete or live-style Door streams."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        if artifact.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported Door artifact version: {artifact.get('artifact_version')!r}."
            )
        required = {"estimator", "feature_names", "feature_config", "labels"}
        missing = required.difference(artifact)
        if missing:
            raise ValueError(f"Door artifact is missing: {', '.join(sorted(missing))}.")
        if tuple(artifact["labels"]) != DOOR_LABELS:
            raise ValueError("Door artifact contains an unexpected label schema.")

        self.estimator = artifact["estimator"]
        self.feature_names = tuple(str(name) for name in artifact["feature_names"])
        self.config = DoorFeatureConfig.from_dict(artifact["feature_config"])
        self.metadata = artifact.get("metadata", {})

    @classmethod
    def from_artifact(cls, path: str | Path) -> DoorPredictor:
        """Load and validate a joblib artifact."""

        source = Path(path)
        try:
            artifact = joblib.load(source)
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"Could not load Door model from {source}: {exc}") from exc
        if not isinstance(artifact, dict):
            raise TypeError("Door artifact must contain a dictionary.")
        return cls(artifact)

    def predict_stream(self, path: str | Path) -> list[DoorPrediction]:
        """Detect and classify every operation in a Door CSV stream."""

        frame, timestamps = load_stream(path)
        segments = detect_segments(frame, timestamps, self.config)
        rows = [
            extract_segment_features(frame, timestamps, segment, self.config)
            for segment in segments
        ]
        missing = [name for name in self.feature_names if name not in rows[0]]
        if missing:
            raise ValueError(f"Extracted Door features are missing {missing[0]!r}.")
        matrix = np.asarray(
            [[float(row[name]) for name in self.feature_names] for row in rows],
            dtype=np.float64,
        )
        raw_predictions = self.estimator.predict(matrix)
        probabilities = (
            self.estimator.predict_proba(matrix)
            if hasattr(self.estimator, "predict_proba")
            else None
        )

        results: list[DoorPrediction] = []
        for index, (segment, raw_prediction) in enumerate(
            zip(segments, raw_predictions, strict=True)
        ):
            prediction = str(raw_prediction)
            if prediction not in DOOR_LABELS:
                raise ValueError(
                    f"Model returned an invalid Door label: {prediction!r}."
                )
            confidence = 1.0
            if probabilities is not None:
                scores = {
                    str(label): float(score)
                    for label, score in zip(
                        self.estimator.classes_, probabilities[index], strict=True
                    )
                }
                confidence = scores[prediction]
            results.append(
                DoorPrediction(
                    start_time=str(frame.at[segment.start, "Datetime"]),
                    end_time=str(frame.at[segment.end - 1, "Datetime"]),
                    prediction=cast(DoorLabel, prediction),
                    confidence=float(confidence),
                    operation=segment.operation,
                    boundary_reason=segment.boundary_reason,
                    quality_flags=segment.quality_flags,
                )
            )
        return results


def predict_stream(
    path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> list[DoorPrediction]:
    """Convenience adapter used by the API integration layer."""

    return DoorPredictor.from_artifact(model_path).predict_stream(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="Continuous Door CSV stream"
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

    predictions = DoorPredictor.from_artifact(args.model).predict_stream(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["start_time", "end_time", "prediction"]
        )
        writer.writeheader()
        writer.writerows(
            {
                "start_time": result.start_time,
                "end_time": result.end_time,
                "prediction": result.prediction,
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
