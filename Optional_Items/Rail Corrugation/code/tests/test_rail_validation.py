"""Regression tests for recording leakage and nested Rail evaluation."""

from pathlib import Path

import numpy as np
import pytest
from backend.models.rail.predict import RAIL_LABELS
from rail_dev import validation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@pytest.fixture
def grouped_data():
    # Two copies per recording, with all three classes represented in every level.
    groups = np.repeat(np.arange(60), 2)
    y = np.asarray(RAIL_LABELS)[groups % 3]
    x = np.random.default_rng(42).normal(size=(len(groups), 4))
    x[:, 0] = groups
    return x, y, groups


def test_raw_content_groups_ignore_names_and_reject_conflicting_labels(tmp_path: Path):
    paths = [tmp_path / name for name in ("a.csv", "b.csv", "c.csv")]
    for path, content in zip(paths, (b"same", b"different", b"same"), strict=True):
        path.write_bytes(content)
    groups = validation.recording_groups(paths, np.array(["Normal"] * 3))
    assert groups[0] == groups[2] != groups[1]
    with pytest.raises(ValueError, match="conflicting labels"):
        validation.recording_groups(paths, np.array(["Normal", "Normal", "Side I"]))


def test_all_validation_levels_keep_recordings_together(grouped_data):
    _, y, groups = grouped_data
    for outer_train, outer_test in validation.grouped_splits(y, groups, 5, 42):
        assert not set(groups[outer_train]) & set(groups[outer_test])
        inner_y, inner_groups = y[outer_train], groups[outer_train]
        for inner_train, inner_test in validation.grouped_splits(
            inner_y, inner_groups, 3, 142
        ):
            assert not set(inner_groups[inner_train]) & set(inner_groups[inner_test])
            for calibration_train, calibration_test in validation.grouped_splits(
                inner_y[inner_train], inner_groups[inner_train], 3, 242
            ):
                local_groups = inner_groups[inner_train]
                assert not set(local_groups[calibration_train]) & set(
                    local_groups[calibration_test]
                )
    with pytest.raises(ValueError, match="recording groups"):
        validation.grouped_splits(y[:6], groups[:6], 5, 42)


def test_svm_calibration_uses_local_grouped_folds(grouped_data):
    x, y, groups = grouped_data
    candidate = CalibratedClassifierCV(
        make_pipeline(StandardScaler(), SVC(class_weight="balanced")),
        cv=3,
        ensemble=False,
    )
    model = validation.fit_grouped(candidate, x, y, groups)
    assert candidate.cv == 3  # The prototype must not acquire another fold's indices.
    for train, test in model.cv:
        assert not set(groups[train]) & set(groups[test])
    assert np.isfinite(model.predict_proba(x)).all()


def test_nested_selection_uses_only_outer_training_and_records_every_prediction(
    grouped_data,
    monkeypatch,
):
    x, y, groups = grouped_data
    monkeypatch.setattr(validation, "OUTER_SEEDS", (42, 43))
    original_select = validation.select_candidate
    original_fit = validation.fit_grouped
    selection_groups = []
    active_selection = None

    def checked_fit(estimator, local_x, local_y, local_groups):
        if active_selection is not None:
            assert set(local_groups) <= active_selection
        return original_fit(estimator, local_x, local_y, local_groups)

    def checked_select(candidates, local_x, local_y, local_groups):
        nonlocal active_selection
        active_selection = set(local_groups)
        selection_groups.append(active_selection)
        try:
            return original_select(candidates, local_x, local_y, local_groups)
        finally:
            active_selection = None

    monkeypatch.setattr(validation, "fit_grouped", checked_fit)
    monkeypatch.setattr(validation, "select_candidate", checked_select)
    candidates = {
        "dummy_normal": DummyClassifier(strategy="constant", constant="Normal"),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "prior": DummyClassifier(strategy="prior"),
    }
    report, predictions = validation.evaluate_nested(
        candidates, x, y, groups.astype(str), [f"{i}.csv" for i in range(len(y))]
    )
    for selected_groups, fold in zip(
        selection_groups, report["outer_selections"], strict=True
    ):
        assert selected_groups == set(fold["train_groups"])
        assert not selected_groups & set(fold["validation_groups"])
        assert fold["selected_model"] == max(
            fold["inner_macro_f1"], key=fold["inner_macro_f1"].get
        )
        part = predictions[
            (predictions.repeat == fold["repeat"]) & (predictions.fold == fold["fold"])
        ]
        selected = part[part.evaluation == "nested_selection"].set_index("file_id")
        baseline = part[part.evaluation == fold["selected_model"]].set_index("file_id")
        assert selected.prediction.equals(baseline.prediction)
    for (_, _), part in predictions.groupby(["evaluation", "repeat"]):
        assert len(part) == len(y)
        assert part.file_id.is_unique
    nested = predictions[predictions.evaluation == "nested_selection"]
    expected = f1_score(
        nested.truth, nested.prediction, labels=RAIL_LABELS, average="macro"
    )
    assert report["nested_selection"]["pooled_repeated_predictions"][
        "macro_f1"
    ] == pytest.approx(expected)
