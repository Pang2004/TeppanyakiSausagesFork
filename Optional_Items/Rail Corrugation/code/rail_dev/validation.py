"""Recording-grouped Rail validation, including model selection and calibration."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from backend.models.rail.predict import RAIL_LABELS
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

OUTER_SEEDS = (42, 43, 44)
INNER_SEED = 142
CALIBRATION_SEED = 242


def recording_groups(paths: list[Path], labels: np.ndarray) -> np.ndarray:
    """Hash raw bytes; reject contradictory labels for the same recording."""
    groups = []
    known: dict[str, str] = {}
    for path, label in zip(paths, labels, strict=True):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest in known and known[digest] != label:
            raise ValueError(
                f"Identical Rail recording has conflicting labels: {path.name}"
            )
        known[digest] = str(label)
        groups.append(digest)
    return np.asarray(groups)


def grouped_splits(
    y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return reproducible splits and fail rather than silently lose a class."""
    if len(y) != len(groups):
        raise ValueError("Rail labels and recording groups must align.")
    for label in RAIL_LABELS:
        if len(np.unique(groups[y == label])) < n_splits:
            raise ValueError(f"Need at least {n_splits} recording groups for {label}.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    for train, validation in splits:
        if set(groups[train]) & set(groups[validation]):
            raise ValueError("Recording groups overlap across a validation split.")
        if set(y[train]) != set(RAIL_LABELS) or set(y[validation]) != set(RAIL_LABELS):
            raise ValueError(
                "Grouped split lacks a Rail class; revise the split protocol."
            )
    return splits


def fit_grouped(
    estimator: Any, x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> Any:
    """Fit a fresh model with calibration splits local to this training subset."""
    model = clone(estimator)
    if isinstance(model, CalibratedClassifierCV):
        model.set_params(cv=grouped_splits(y, groups, 3, CALIBRATION_SEED))
    return model.fit(x, y)


def select_candidate(
    candidates: dict[str, Any], x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[str, dict[str, float]]:
    """Choose among the fixed candidates using only inner validation scores."""
    splits = grouped_splits(y, groups, 3, INNER_SEED)
    scores: dict[str, float] = {}
    for name, estimator in candidates.items():
        if name == "dummy_normal":
            continue
        values = []
        for train, validation in splits:
            model = fit_grouped(estimator, x[train], y[train], groups[train])
            values.append(
                f1_score(
                    y[validation],
                    model.predict(x[validation]),
                    labels=RAIL_LABELS,
                    average="macro",
                    zero_division=0,
                )
            )
        scores[name] = float(np.mean(values))
    # Dict order is the deterministic tie break, preferring the simpler logistic model.
    return max(scores, key=scores.get), scores


def classification_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=RAIL_LABELS, zero_division=0
    )
    return {
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            label: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(n),
            }
            for label, p, r, f, n in zip(
                RAIL_LABELS, precision, recall, f1, support, strict=True
            )
        },
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=RAIL_LABELS
        ).tolist(),
    }


def summarize_predictions(frame: pd.DataFrame) -> dict[str, Any]:
    """Report fold variability and complete out-of-fold results per repeat."""
    folds = [
        {
            "repeat": int(repeat),
            "fold": int(fold),
            **classification_metrics(part.truth, part.prediction),
        }
        for (repeat, fold), part in frame.groupby(["repeat", "fold"], sort=True)
    ]
    repeats = [
        {"repeat": int(repeat), **classification_metrics(part.truth, part.prediction)}
        for repeat, part in frame.groupby("repeat", sort=True)
    ]
    values = [fold["macro_f1"] for fold in folds]
    return {
        "macro_f1_mean": float(np.mean(values)),
        "macro_f1_std": float(np.std(values)),
        "per_class_f1_mean": {
            label: float(np.mean([fold["per_class"][label]["f1"] for fold in folds]))
            for label in RAIL_LABELS
        },
        "pooled_repeated_predictions": classification_metrics(
            frame.truth, frame.prediction
        ),
        "folds": folds,
        "repeats": repeats,
    }


def evaluate_nested(
    candidates: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    file_ids: list[str],
    outer_seeds: tuple[int, ...] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate fixed baselines and inner-selected models on identical outer folds."""
    outer_seeds = OUTER_SEEDS if outer_seeds is None else outer_seeds
    rows: list[dict[str, Any]] = []
    selections = []
    for repeat, seed in enumerate(outer_seeds, start=1):
        for fold, (train, validation) in enumerate(
            grouped_splits(y, groups, 5, seed), start=1
        ):
            selected, scores = select_candidate(
                candidates, x[train], y[train], groups[train]
            )
            selections.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "selected_model": selected,
                    "inner_macro_f1": scores,
                    "train_groups": sorted(set(groups[train])),
                    "validation_groups": sorted(set(groups[validation])),
                }
            )
            for name, estimator in candidates.items():
                model = fit_grouped(estimator, x[train], y[train], groups[train])
                predictions = model.predict(x[validation])
                for index, prediction in zip(validation, predictions, strict=True):
                    row = {
                        "repeat": repeat,
                        "fold": fold,
                        "file_id": file_ids[index],
                        "group_id": groups[index],
                        "truth": y[index],
                        "prediction": prediction,
                        "evaluation": name,
                        "selected_model": name,
                    }
                    rows.append(row)
                    if name == selected:
                        rows.append({**row, "evaluation": "nested_selection"})
            print(
                f"Repeat {repeat}, fold {fold}: inner selection {selected}", flush=True
            )
    predictions = pd.DataFrame(rows)
    summaries = {
        name: summarize_predictions(part)
        for name, part in predictions.groupby("evaluation", sort=False)
    }
    group_table = pd.DataFrame({"file_id": file_ids, "group_id": groups, "label": y})
    duplicates = [
        {"group_id": group, "files": part.file_id.tolist(), "label": part.label.iloc[0]}
        for group, part in group_table.groupby("group_id")
        if len(part) > 1
    ]
    result = {
        "report_version": "rail-validation-v2",
        "protocol": {
            "grouping": "SHA-256 of raw recording bytes",
            "outer": "StratifiedGroupKFold",
            "outer_splits": 5,
            "outer_seeds": list(outer_seeds),
            "inner_splits": 3,
            "inner_seed": INNER_SEED,
            "calibration_splits": 3,
            "calibration_seed": CALIBRATION_SEED,
            "selection_metric": "mean inner-fold macro F1",
            "candidate_order": list(candidates),
            "weighting": "one vote per file; duplicates retained but grouped",
            "limitations": [
                "Fold standard deviations are not confidence intervals.",
                "Repeated predictions are correlated, not additional independent samples.",
                "Byte hashes do not identify near duplicates or shared acquisition sessions.",
                "Previously developed feature choices are not independently validated by nesting.",
            ],
        },
        "labels": list(RAIL_LABELS),
        "training_files": len(y),
        "unique_recording_groups": len(set(groups)),
        "duplicate_groups": duplicates,
        "fixed_candidates": {name: summaries[name] for name in candidates},
        "nested_selection": summaries["nested_selection"],
        "outer_selections": selections,
    }
    return result, predictions
