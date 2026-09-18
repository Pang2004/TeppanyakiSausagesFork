"""Schema, ranking, and artifact tests for the ACV pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from acv_dev.metrics import rank_decay_score
from backend.models.acv.features import (
    ACVFeatureConfig,
    ACVInputError,
    extract_car_features,
    load_workbook,
    rank_feature_frame,
)
from backend.models.acv.predict import ARTIFACT_VERSION, ACVPredictor

REPOSITORY_ROOT = next(
    parent for parent in Path(__file__).parents if (parent / "PS3").is_dir()
)
DATA_DIR = REPOSITORY_ROOT / "PS3/02_Datasets/ACV"


def _compact_frame(hot_car: str = "03", rows: int = 40) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"Time": pd.date_range("2026-01-01", periods=rows, freq="30s")}
    )
    variation = np.sin(np.linspace(0, 2 * np.pi, rows)) * 0.2
    for number in range(1, 9):
        car_id = f"{number:02d}"
        offset = 1.5 if car_id == hot_car else number * 0.01
        frame[f"Car {car_id} - Indoor Average Temperature"] = 24.0 + variation + offset
        frame[f"Car {car_id} - ACV Control Temperature (Cooling)"] = 24.0
        frame[f"Car {car_id} - ACV Running Mode"] = "Automatic Cooling"
        frame[f"Car {car_id} - ACV Information Valid"] = "Valid"
    return frame


def _write_workbook(
    path: Path, frame: pd.DataFrame, sheet_name: str = "Sheet1"
) -> None:
    frame.to_excel(path, index=False, sheet_name=sheet_name)


def test_loader_accepts_non_english_sheet_and_preserves_car_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "case.xlsx"
    _write_workbook(path, _compact_frame(), sheet_name="故障案例")
    workbook = load_workbook(path)
    assert workbook.car_ids == ("01", "02", "03", "04", "05", "06", "07", "08")
    assert len(workbook.frame) == 40


def test_loader_rejects_missing_time_and_wrong_extension(tmp_path: Path) -> None:
    missing_time = tmp_path / "missing.xlsx"
    _write_workbook(missing_time, _compact_frame().drop(columns="Time"))
    with pytest.raises(ACVInputError, match="missing the 'Time'"):
        load_workbook(missing_time)
    with pytest.raises(ACVInputError, match="must be an .xlsx"):
        load_workbook(tmp_path / "case.csv")


def test_temperature_ranker_places_sustained_hot_car_first(tmp_path: Path) -> None:
    path = tmp_path / "hot.xlsx"
    _write_workbook(path, _compact_frame(hot_car="03"))
    features = extract_car_features(path)
    assert rank_feature_frame(features)[0] == "03"
    assert features.loc[features["car_id"].eq("03"), "temperature_score"].item() == 1


def test_invalid_zero_temperature_is_not_leak_evidence(tmp_path: Path) -> None:
    frame = _compact_frame(hot_car="02")
    frame["Car 01 - Indoor Average Temperature"] = 0.0
    path = tmp_path / "invalid-zero.xlsx"
    _write_workbook(path, frame)
    features = extract_car_features(path)
    ranking = rank_feature_frame(features)
    assert ranking[0] == "02"
    assert ranking[-1] == "01"


def test_pressure_circuit_mismatch_overrides_temperature_only_rank(
    tmp_path: Path,
) -> None:
    frame = _compact_frame(hot_car="04")
    for number in range(1, 9):
        car_id = f"{number:02d}"
        for system in (1, 2):
            low = 500.0
            if car_id == "02" and system == 2:
                low = 200.0
            frame[
                f"Car {car_id} - Refrigeration System {system} High Pressure Value"
            ] = 2_000.0
            frame[
                f"Car {car_id} - Refrigeration System {system} Low Pressure Value"
            ] = low
            frame[f"Car {car_id} - Compressor {system} Running"] = 1
    path = tmp_path / "pressure.xlsx"
    _write_workbook(path, frame)
    features = extract_car_features(path)
    assert rank_feature_frame(features, "temperature_score")[0] == "04"
    assert rank_feature_frame(features)[0] == "02"
    assert features["schema"].eq("rich_pressure").all()


def test_metric_uses_linear_rank_decay() -> None:
    ranking = ["01", "02", "03", "04", "05", "06", "07", "08"]
    assert rank_decay_score(ranking, "01") == 1.0
    assert rank_decay_score(ranking, "03") == 0.75
    assert rank_decay_score(ranking, "99") == 0.0


def test_saved_artifact_predicts_every_test_car(tmp_path: Path) -> None:
    artifact_path = tmp_path / "acv.joblib"
    joblib.dump(
        {
            "artifact_version": ARTIFACT_VERSION,
            "feature_config": ACVFeatureConfig().to_dict(),
            "validation": {"hybrid": {"mean_rank_decay_score": 1.0}},
            "metadata": {"purpose": "test"},
        },
        artifact_path,
    )
    result = ACVPredictor.from_artifact(artifact_path).predict_file(
        DATA_DIR / "Test/acv_test_case.xlsx"
    )
    assert result.file_id == "acv_test_case.xlsx"
    assert result.ranked_cars == ("01", "04", "03", "08", "07", "02", "06", "05")
    assert set(result.ranked_cars) == {f"{number:02d}" for number in range(1, 9)}
    assert result.schema == "compact_temperature"


def test_disclosed_fault_ranks_first_in_every_training_case() -> None:
    labels = pd.read_csv(DATA_DIR / "Train_Labels.csv", dtype="string")
    for row in labels.itertuples(index=False):
        features = extract_car_features(DATA_DIR / "Train" / str(row.filename))
        ranking = rank_feature_frame(features)
        assert ranking[0] == str(row.faulty_car).zfill(2), row.filename
