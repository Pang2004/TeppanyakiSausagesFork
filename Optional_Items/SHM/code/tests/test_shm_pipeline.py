"""Contract and physics tests for the SHM fatigue pipeline."""

from __future__ import annotations

from math import cos, sin
from pathlib import Path

import joblib
import numpy as np
import pytest
import rainflow
from backend.models.shm.features import (
    SHMFeatureConfig,
    SHMInputError,
    extract_features,
    load_stress,
    miner_proxy,
)
from backend.models.shm.predict import ARTIFACT_VERSION, SHMPredictor
from shm_dev.metrics import (
    calibrate_miner_scale,
    mean_absolute_percentage_error,
    official_shm_score,
)

REPOSITORY_ROOT = next(
    parent for parent in Path(__file__).parents if (parent / "PS3").is_dir()
)
DATA_DIR = REPOSITORY_ROOT / "PS3/02_Datasets/SHM"


def _cycle_arrays(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cycles = np.asarray(list(rainflow.extract_cycles(values)), dtype=np.float64)
    return cycles[:, 0], cycles[:, 2]


def test_headerless_loader_preserves_every_sample() -> None:
    path = DATA_DIR / "Train/train01.csv"
    first_text_value = float(path.read_text(encoding="utf-8").splitlines()[0])
    values = load_stress(path)
    assert values.shape == (581_120,)
    assert values[0] == first_text_value


def test_invalid_header_and_multicolumn_files_are_rejected(tmp_path: Path) -> None:
    header = tmp_path / "header.csv"
    header.write_text("stress\n1\n2\n3\n", encoding="utf-8")
    with pytest.raises(SHMInputError, match="Could not read numeric"):
        load_stress(header)

    multicolumn = tmp_path / "multi.csv"
    np.savetxt(
        multicolumn, np.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]), delimiter=","
    )
    with pytest.raises(SHMInputError, match="exactly one"):
        load_stress(multicolumn)


def test_rainflow_matches_published_package_example() -> None:
    time = [4.0 * index / 200 for index in range(201)]
    series = [
        0.2 + 0.5 * sin(value) + 0.2 * cos(10 * value) + 0.2 * sin(4 * value)
        for value in time
    ]
    expected = [
        (0.04258965150708488, 0.5),
        (0.10973439445727551, 1.0),
        (0.11294628078678906, 0.5),
        (0.20571069989638404, 1.0),
        (0.21467990941625242, 1.0),
        (0.4388985979776988, 1.0),
        (0.48305748051348263, 0.5),
        (0.5286423866535466, 0.5),
        (0.7809330293159786, 0.5),
        (1.4343610172143002, 0.5),
    ]
    actual = rainflow.count_cycles(series)
    np.testing.assert_allclose(actual, expected)


def test_miner_proxy_obeys_fifth_power_stress_scaling() -> None:
    values = np.sin(np.linspace(0, 20 * np.pi, 2_000))
    ranges, counts = _cycle_arrays(values)
    scaled_ranges, scaled_counts = _cycle_arrays(values * 3.0)
    original = miner_proxy(ranges, counts, 5)
    scaled = miner_proxy(scaled_ranges, scaled_counts, 5)
    assert np.isclose(scaled / original, 3**5)


def test_repeated_periodic_history_doubles_damage() -> None:
    period = np.asarray([0.0, 2.0, 0.0, -2.0])
    ten_periods = np.concatenate([np.tile(period, 10), [0.0]])
    twenty_periods = np.concatenate([np.tile(period, 20), [0.0]])
    first_ranges, first_counts = _cycle_arrays(ten_periods)
    second_ranges, second_counts = _cycle_arrays(twenty_periods)
    first_damage = miner_proxy(first_ranges, first_counts, 5)
    second_damage = miner_proxy(second_ranges, second_counts, 5)
    assert np.isclose(second_damage, 2 * first_damage, rtol=0.06)


def test_mape_metric_and_scale_calibration() -> None:
    proxies = np.asarray([1.0, 2.0, 3.0])
    actual = proxies * 0.25
    scale = calibrate_miner_scale(actual, proxies)
    predicted = scale * proxies
    assert scale == 0.25
    assert mean_absolute_percentage_error(actual, predicted) == 0.0
    assert official_shm_score(actual, predicted) == 1.0


def test_features_are_finite_and_deterministic() -> None:
    path = DATA_DIR / "Train/train01.csv"
    first = extract_features(path)
    second = extract_features(path)
    assert first == second
    assert first["file_id"] == path.name
    numeric = np.asarray([value for name, value in first.items() if name != "file_id"])
    assert np.isfinite(numeric).all()
    assert first["rainflow__cycle_count"] > 0


def test_saved_artifact_predicts_positive_damage(tmp_path: Path) -> None:
    artifact_path = tmp_path / "shm.joblib"
    joblib.dump(
        {
            "artifact_version": ARTIFACT_VERSION,
            "feature_config": SHMFeatureConfig().to_dict(),
            "exponent": 5,
            "scale": 1.0e-9,
            "residual_estimator": None,
            "residual_feature_names": [],
            "validation": {"p95_absolute_percentage_error": 0.1},
            "metadata": {"purpose": "test"},
        },
        artifact_path,
    )
    result = SHMPredictor.from_artifact(artifact_path).predict_file(
        DATA_DIR / "Test/test01.csv"
    )
    assert result.file_id == "test01.csv"
    assert result.prediction > 0
    assert result.cycle_count > 0
    assert result.estimated_percentage_error == 0.1
    assert (
        result.to_dict()["error_indicator_kind"]
        == "historical_validation_p95_absolute_percentage_error"
    )
