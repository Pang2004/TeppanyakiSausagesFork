"""Check SHM selection isolation and feature cache provenance."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from backend.models.shm.features import SHMFeatureConfig
from shm_dev import train
from shm_dev.validation import evaluate_nested


def test_held_out_label_cannot_change_its_model_selection_or_prediction():
    x = np.linspace(1, 4, 12)
    features = pd.DataFrame(
        {
            "file_id": [f"history{i}" for i in range(12)],
            "rainflow__miner_proxy_m5": x**5,
            "shape": np.sin(x),
        }
    )
    damage = x**5 * np.exp(0.1 * np.sin(x))
    config = SHMFeatureConfig(candidate_exponents=(5,))
    original = evaluate_nested(features, damage, ["shape"], config)
    changed_damage = damage.copy()
    changed_damage[0] *= 10
    changed = evaluate_nested(features, changed_damage, ["shape"], config)
    first, second = original["folds"][0], changed["folds"][0]
    assert first["selection"] == second["selection"]
    assert first["nested_prediction"] == second["nested_prediction"]
    assert first["actual"] != second["actual"]
    for fold in original["folds"]:
        assert fold["validation_index"] not in fold["train_indices"]
        assert len(fold["train_indices"]) == 11
    expected = np.mean(
        [
            abs(row["actual"] - row["nested_prediction"]) / row["actual"]
            for row in original["folds"]
        ]
    )
    assert original["summary"]["nested_selection"]["mape"] == pytest.approx(expected)


def test_cache_rebuilds_for_raw_data_config_and_cache_changes(tmp_path, monkeypatch):
    source = tmp_path / "history.csv"
    source.write_text("1\n2\n3\n")
    cache = tmp_path / "features.csv"
    calls = []

    def extractor(path, config):
        calls.append(path)
        return {
            "file_id": path.name,
            "value": float(path.read_text().splitlines()[0]) * config.block_count,
        }

    monkeypatch.setattr(train, "extract_features", extractor)
    config = SHMFeatureConfig()
    load = lambda cfg: train._load_or_extract_features([source], cache, cfg, 1, False)
    first = load(config)
    pd.testing.assert_frame_equal(first, load(config))
    assert len(calls) == 1
    source.write_text("2\n2\n3\n")
    assert load(config).value.iloc[0] == 256
    assert len(calls) == 2
    config = replace(config, block_count=64)
    assert load(config).value.iloc[0] == 128
    assert len(calls) == 3
    cache.write_text("file_id,value\nhistory.csv,999\n")
    assert load(config).value.iloc[0] == 128
    assert len(calls) == 4


def test_duplicate_histories_cannot_enter_file_level_validation(tmp_path):
    paths = [tmp_path / "one.csv", tmp_path / "two.csv"]
    for path in paths:
        path.write_text("1\n2\n3\n")
    with pytest.raises(ValueError, match="Duplicate SHM histories"):
        train._load_or_extract_features(
            paths, tmp_path / "features.csv", SHMFeatureConfig(), 1, False
        )
