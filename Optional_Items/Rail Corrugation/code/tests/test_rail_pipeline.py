"""Contract tests for the Rail feature and inference pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from backend.models.rail.features import (
    EXPECTED_SAMPLES,
    FeatureConfig,
    RailInputError,
    extract_features,
    parse_sensor_layout,
)
from backend.models.rail.predict import (
    LEGACY_ARTIFACT_VERSION,
    RAIL_LABELS,
    RailPredictor,
)
from sklearn.dummy import DummyClassifier


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


def test_research_mirroring_matches_swapped_raw_sensors(
    recording: Path, tmp_path: Path
):
    from backend.models.rail.wavelength import extract_production_features
    from rail_dev.local_spectra import extract_local
    from rail_dev.side_i_study import mirror_mapping

    frame = pd.read_csv(recording)
    layout = parse_sensor_layout(frame.columns)
    source = {(s.car, s.position, s.signal_type): s.name for s in layout}
    swapped = frame.copy()
    for sensor in layout:
        other_position = sensor.position + (1 if sensor.position % 2 else -1)
        swapped[sensor.name] = frame[
            source[sensor.car, other_position, sensor.signal_type]
        ]
    path = tmp_path / "swapped.csv"
    swapped.to_csv(path, index=False)
    original = extract_production_features(recording)
    mirrored = extract_production_features(path)
    names = [name for name in original if name != "file_id"]
    indices, signs = mirror_mapping(names)
    values = np.array([original[name] for name in names])
    np.testing.assert_allclose(
        values[indices] * signs,
        [mirrored[name] for name in names],
        atol=1e-8,
        rtol=1e-5,
    )
    np.testing.assert_allclose((values[indices] * signs)[indices] * signs, values)
    local = extract_local(recording)
    swapped_local = extract_local(path)
    names = [name for name in local if name != "file_id"]
    indices, signs = mirror_mapping(names)
    values = np.array([local[name] for name in names])
    np.testing.assert_allclose(
        values[indices] * signs,
        [swapped_local[name] for name in names],
        atol=1e-8,
        rtol=1e-5,
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
            "artifact_version": LEGACY_ARTIFACT_VERSION,
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


def test_wavelength_artifact_roundtrip_and_configuration_guard(recording, tmp_path):
    from backend.models.rail.predict import ARTIFACT_VERSION
    from backend.models.rail.wavelength import (
        FEATURE_SET,
        WAVELENGTH_CONFIG,
        extract_production_features,
    )

    row = extract_production_features(recording)
    names = [name for name in row if name != "file_id"]
    assert len(names) == 855
    matrix = np.asarray([[row[name] for name in names]] * 3)
    model = DummyClassifier(strategy="prior").fit(matrix, np.asarray(RAIL_LABELS))
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "feature_set": FEATURE_SET,
        "wavelength_config": WAVELENGTH_CONFIG,
        "feature_config": FeatureConfig().to_dict(),
        "feature_names": names,
        "labels": list(RAIL_LABELS),
        "estimator": model,
    }
    path = tmp_path / "wavelength.joblib"
    joblib.dump(artifact, path)
    result = RailPredictor.from_artifact(path).predict_file(recording)
    assert result.prediction == model.predict(matrix[:1])[0]
    assert sum(result.scores.values()) == pytest.approx(1.0)
    bad = {
        **artifact,
        "wavelength_config": {**WAVELENGTH_CONFIG, "edges_per_rotation": 90},
    }
    with pytest.raises(ValueError, match="wavelength feature configuration"):
        RailPredictor(bad)
    with pytest.raises(ValueError, match="wavelength feature configuration"):
        RailPredictor(
            {key: value for key, value in artifact.items() if key != "feature_set"}
        )


def test_production_cache_invalidates_with_raw_hash_and_feature_configuration(
    tmp_path, monkeypatch
):
    from rail_dev import train

    calls = []
    path = tmp_path / "Train1.csv"
    cache = tmp_path / "features.csv"

    def extract(paths, config, jobs):
        calls.append(paths)
        return pd.DataFrame({"file_id": [p.name for p in paths], "value": [len(calls)]})

    monkeypatch.setattr(train, "_extract_dataset", extract)
    for digest in ("one", "one", "two"):
        train._load_or_extract_features(
            [path], cache, FeatureConfig(), 1, False, np.array([digest])
        )
    assert len(calls) == 2
    # An old filename-only cache must never be reused for the upgraded feature set.
    cache.with_suffix(".manifest.json").unlink()
    train._load_or_extract_features(
        [path], cache, FeatureConfig(), 1, False, np.array(["two"])
    )
    assert len(calls) == 3


def test_local_artifact_roundtrip_rejects_configuration_mismatch(recording, tmp_path):
    from backend.models.rail.local import CONFIG, FEATURE_SET, extract_combined
    from backend.models.rail.local_model import LocalRailClassifier
    from backend.models.rail.predict import LOCAL_ARTIFACT_VERSION
    from backend.models.rail.wavelength import WAVELENGTH_CONFIG

    row = extract_combined(recording)
    names = tuple(name for name in row if name != "file_id")
    x = np.asarray([[row[name] for name in names]] * 3)
    model = LocalRailClassifier(
        names, list(range(len(names))), c=0.1, mirror=True, boost=2
    ).fit(x, np.asarray(RAIL_LABELS))
    artifact = {
        "artifact_version": LOCAL_ARTIFACT_VERSION,
        "feature_set": FEATURE_SET,
        "feature_config": FeatureConfig().to_dict(),
        "wavelength_config": WAVELENGTH_CONFIG,
        "local_config": CONFIG,
        "feature_names": names,
        "labels": list(RAIL_LABELS),
        "estimator": model,
    }
    path = tmp_path / "local.joblib"
    joblib.dump(artifact, path)
    predictor = RailPredictor.from_artifact(path)
    assert predictor.predict_file(recording).prediction == model.predict(x[:1])[0]
    assert predictor.artifact_version == LOCAL_ARTIFACT_VERSION
    with pytest.raises(ValueError, match="local feature configuration"):
        RailPredictor({**artifact, "local_config": {**CONFIG, "nperseg": 1024}})
    with pytest.raises(ValueError, match="local feature configuration"):
        RailPredictor({k: v for k, v in artifact.items() if k != "local_config"})
