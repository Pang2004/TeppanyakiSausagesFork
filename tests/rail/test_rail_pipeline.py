"""Contract tests for the Rail feature and inference pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from backend.models.rail.features import (
    EXPECTED_SAMPLES,
    FeatureConfig,
    RailInputError,
    extract_features,
    parse_sensor_layout,
)
from backend.models.rail.predict import ARTIFACT_VERSION, RAIL_LABELS, RailPredictor


def _columns() -> list[str]:
    columns = ["Rotating speed"]
    for car in range(1, 9):
        for position in range(1, 9):
            columns.extend(
                [
                    f"Vibration of bearing in position {position} of car {car}",
                    f"Shock of bearing in position {position} of car {car}",
                ]
            )
    return columns


@pytest.fixture(scope="module")
def recording(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("rail") / "Test1.csv"
    time = np.arange(EXPECTED_SAMPLES, dtype=np.float32) / 10_000
    values = np.empty((EXPECTED_SAMPLES, 129), dtype=np.float32)
    values[:, 0] = (time > 0.2).astype(np.float32) * 800
    for index in range(1, 129):
        frequency = 40 + index * 3
        values[:, index] = np.sin(2 * np.pi * frequency * time) * (1 + index / 128)
    pd.DataFrame(values, columns=_columns()).to_csv(path, index=False)
    return path


def test_layout_maps_odd_positions_to_side_i() -> None:
    layout = parse_sensor_layout(_columns())
    for side in ("i", "ii"):
        for signal_type in ("vibration", "shock"):
            sensors = [
                item
                for item in layout
                if item.side == side and item.signal_type == signal_type
            ]
            assert len(sensors) == 32
            expected_remainder = 1 if side == "i" else 0
            assert all(item.position % 2 == expected_remainder for item in sensors)


def test_feature_extraction_is_finite_and_deterministic(recording: Path) -> None:
    first = extract_features(recording)
    second = extract_features(recording)
    assert first.keys() == second.keys()
    assert len(first) > 300
    assert first["file_id"] == recording.name
    numeric = np.asarray([value for key, value in first.items() if key != "file_id"])
    assert np.isfinite(numeric).all()
    np.testing.assert_allclose(
        numeric, [second[key] for key in first if key != "file_id"]
    )


def test_wrong_sample_count_is_rejected(recording: Path, tmp_path: Path) -> None:
    invalid = tmp_path / "short.csv"
    pd.read_csv(recording, nrows=100).to_csv(invalid, index=False)
    with pytest.raises(RailInputError, match="10000 samples"):
        extract_features(invalid)


def test_saved_artifact_predicts_and_exposes_diagnostics(
    recording: Path, tmp_path: Path
) -> None:
    row = extract_features(recording)
    feature_names = [name for name in row if name != "file_id"]
    x = np.asarray([[float(row[name]) for name in feature_names]] * 3)
    estimator = DummyClassifier(strategy="prior").fit(x, np.asarray(RAIL_LABELS))
    artifact_path = tmp_path / "rail.joblib"
    joblib.dump(
        {
            "artifact_version": ARTIFACT_VERSION,
            "estimator": estimator,
            "feature_names": feature_names,
            "feature_config": FeatureConfig().to_dict(),
            "labels": list(RAIL_LABELS),
            "metadata": {"purpose": "test"},
        },
        artifact_path,
    )

    result = RailPredictor.from_artifact(artifact_path).predict_file(recording)
    assert result.prediction in RAIL_LABELS
    assert set(result.scores) == set(RAIL_LABELS)
    assert set(result.side_energy) == {"Side I", "Side II"}
    assert set(result.dominant_frequency) == {"Side I", "Side II"}
