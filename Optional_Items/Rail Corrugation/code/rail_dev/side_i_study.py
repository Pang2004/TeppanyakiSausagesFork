"""Side I experiments with training-only transforms and inner decision tuning.

Run from the repository root using the same PYTHONPATH as rail_dev.recheck.
Reports are exploratory; no production artifact is changed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from backend.models.rail.features import FeatureConfig
from backend.models.rail.predict import RAIL_LABELS
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .train import _load_or_extract_features, _load_training_index
from .validation import (
    INNER_SEED,
    OUTER_SEEDS,
    classification_metrics,
    grouped_splits,
    recording_groups,
    summarize_predictions,
)

OUTPUT = Path("Optional_Items/Rail Corrugation/code/outputs/side_i_study")
BOOSTS = (0.5, 1.0, 2.0, 4.0)


def mirror_mapping(names):
    """Swap side summaries and negate antisymmetric contrasts."""
    lookup = {name: i for i, name in enumerate(names)}
    indices, signs = [], []
    for name in names:
        target, sign = name, 1
        for left, right in (
            ("side_i_", "side_ii_"),
            ("vibration_i_", "vibration_ii_"),
            ("shock_i_", "shock_ii_"),
        ):
            if name.startswith(left):
                target = right + name[len(left) :]
                break
            if name.startswith(right):
                target = left + name[len(right) :]
                break
        if name.startswith("side_comparison_") or "_contrast_" in name:
            sign = -1
        indices.append(lookup[target])
        signs.append(sign)
    return np.array(indices), np.array(signs)


class StudyClassifier(ClassifierMixin, BaseEstimator):
    """Log compression, optional mirrored training samples, optional selection."""

    def __init__(self, names, c=1.0, log=False, mirror=False, k=None, kind="logistic"):
        self.names = names
        self.c = c
        self.log = log
        self.mirror = mirror
        self.k = k
        self.kind = kind

    def transform(self, x):
        return np.sign(x) * np.log1p(np.abs(x)) if self.log else x

    def fit(self, x, y):
        if self.mirror:
            indices, signs = mirror_mapping(self.names)
            x = np.concatenate((x, x[:, indices] * signs))
            swapped = np.array(
                [
                    {"Normal": "Normal", "Side I": "Side II", "Side II": "Side I"}[v]
                    for v in y
                ]
            )
            y = np.concatenate((y, swapped))
        if self.kind == "hierarchical":
            self.scaler_ = StandardScaler().fit(self.transform(x))
            scaled = self.scaler_.transform(self.transform(x))
            fault = y != "Normal"
            self.fault_ = LogisticRegression(
                C=self.c, class_weight="balanced", max_iter=5000, random_state=42
            ).fit(scaled, fault)
            self.side_ = LogisticRegression(
                C=self.c, class_weight="balanced", max_iter=5000, random_state=42
            ).fit(scaled[fault], y[fault])
            self.classes_ = np.asarray(RAIL_LABELS)
            return self
        if self.kind == "extra_trees":
            self.model_ = ExtraTreesClassifier(
                n_estimators=150,
                min_samples_leaf=int(self.c),
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ).fit(self.transform(x), y)
            self.classes_ = self.model_.classes_
            return self
        steps = [StandardScaler()]
        if self.k is not None:
            steps.append(SelectKBest(f_classif, k=self.k))
        steps.append(
            LogisticRegression(
                C=self.c, class_weight="balanced", max_iter=5000, random_state=42
            )
        )
        self.model_ = make_pipeline(*steps).fit(self.transform(x), y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, x):
        if self.kind == "hierarchical":
            scaled = self.scaler_.transform(self.transform(x))
            fault = self.fault_.predict_proba(scaled)[:, 1]
            side = self.side_.predict_proba(scaled)
            return np.column_stack((1 - fault, fault * side[:, 0], fault * side[:, 1]))
        return self.model_.predict_proba(self.transform(x))

    def predict(self, x):
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]


def boosted_prediction(probabilities, boost):
    adjusted = probabilities.copy()
    adjusted[:, RAIL_LABELS.index("Side I")] *= boost
    return np.asarray(RAIL_LABELS)[adjusted.argmax(axis=1)]


def make_candidates(names, stage="transforms"):
    candidates = {}
    for log, mirror, family in (
        (False, False, "original"),
        (True, False, "log"),
        (False, True, "mirror"),
        (True, True, "log_mirror"),
    ):
        for c in (0.01, 0.1, 1.0, 10.0):
            candidates[f"{family}_C{c}"] = {
                "names": names,
                "c": c,
                "log": log,
                "mirror": mirror,
            }
    for c in (0.1, 1.0):
        candidates[f"log_top80_C{c}"] = {"names": names, "c": c, "log": True, "k": 80}
    if stage == "architecture":
        candidates = {k: v for k, v in candidates.items() if k.startswith("original_")}
        for mirror in (False, True):
            for c in (0.01, 0.1, 1.0):
                candidates[f"hierarchical_mirror{mirror}_C{c}"] = {
                    "names": names,
                    "c": c,
                    "mirror": mirror,
                    "kind": "hierarchical",
                }
        for leaf in (1, 3):
            candidates[f"extra_trees_leaf{leaf}"] = {
                "names": names,
                "c": leaf,
                "kind": "extra_trees",
            }
    return candidates


def inner_scores(candidates, x, y, groups):
    splits = grouped_splits(y, groups, 3, INNER_SEED)
    scores = []
    for name, config in candidates.items():
        fold_metrics = {boost: [] for boost in BOOSTS}
        for train, valid in splits:
            model = StudyClassifier(**config).fit(x[train], y[train])
            assert tuple(model.classes_) == RAIL_LABELS
            probabilities = model.predict_proba(x[valid])
            for boost in BOOSTS:
                fold_metrics[boost].append(
                    classification_metrics(
                        y[valid], boosted_prediction(probabilities, boost)
                    )
                )
        for boost, metrics in fold_metrics.items():
            scores.append(
                {
                    "candidate": name,
                    "boost": boost,
                    "macro": float(np.mean([m["macro_f1"] for m in metrics])),
                    "side_i": float(
                        np.mean([m["per_class"]["Side I"]["f1"] for m in metrics])
                    ),
                }
            )
    return scores


def choose(scores):
    original = [
        s for s in scores if s["candidate"].startswith("original_") and s["boost"] == 1
    ]
    baseline = max(original, key=lambda s: s["macro"])
    eligible = [s for s in scores if s["macro"] >= baseline["macro"] - 0.02]
    thresholds = [
        s
        for s in scores
        if s["candidate"].startswith("original_")
        and s["macro"] >= baseline["macro"] - 0.02
    ]
    return {
        "baseline": baseline,
        "threshold_only": max(thresholds, key=lambda s: (s["side_i"], s["macro"])),
        "select_macro": max(scores, key=lambda s: s["macro"]),
        "select_side_i_guarded": max(eligible, key=lambda s: (s["side_i"], s["macro"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("transforms", "architecture"), default="transforms"
    )
    args = parser.parse_args()
    output = OUTPUT if args.stage == "transforms" else OUTPUT / "architecture"
    paths, labels = _load_training_index(Path("PS3/02_Datasets/Rail_Corrugation"))
    y = np.asarray(labels)
    groups = recording_groups(paths, y)
    features = _load_or_extract_features(
        paths,
        OUTPUT.parent / "train_features_wavelength.csv",
        FeatureConfig(),
        4,
        False,
        groups,
    )
    names = tuple(features.columns.drop("file_id"))
    x = features[list(names)].to_numpy(float)
    candidates = make_candidates(names, args.stage)
    rows, selections = [], []
    for repeat, seed in enumerate(OUTER_SEEDS, 1):
        for fold, (train, valid) in enumerate(grouped_splits(y, groups, 5, seed), 1):
            scores = inner_scores(candidates, x[train], y[train], groups[train])
            selected = choose(scores)
            selections.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "selections": selected,
                    "inner_scores": scores,
                }
            )
            for name, config in candidates.items():
                model = StudyClassifier(**config).fit(x[train], y[train])
                probabilities = model.predict_proba(x[valid])
                evaluations = {f"fixed_{name}": 1.0}
                evaluations.update(
                    {
                        key: value["boost"]
                        for key, value in selected.items()
                        if value["candidate"] == name
                    }
                )
                for evaluation, boost in evaluations.items():
                    predictions = boosted_prediction(probabilities, boost)
                    for offset, index in enumerate(valid):
                        rows.append(
                            {
                                "repeat": repeat,
                                "fold": fold,
                                "file_id": paths[index].name,
                                "group_id": groups[index],
                                "truth": y[index],
                                "prediction": predictions[offset],
                                "evaluation": evaluation,
                                "candidate": name,
                                "boost": boost,
                                "score_side_i": float(probabilities[offset, 1]),
                            }
                        )
            print(f"Repeat {repeat}, fold {fold}: {selected}", flush=True)
    predictions = pd.DataFrame(rows)
    summaries = {
        name: summarize_predictions(part)
        for name, part in predictions.groupby("evaluation")
    }
    report = {
        "protocol": {
            "outer_seeds": list(OUTER_SEEDS),
            "outer_folds": 5,
            "inner_seed": INNER_SEED,
            "inner_folds": 3,
            "boosts": BOOSTS,
            "side_i_selection": "Maximum mean inner Side I F1 subject to macro >= inner baseline - 0.02",
            "grouping": "SHA-256 raw recordings; mirrored samples created only after splitting",
            "limitation": "Exploratory on previously inspected data; no independent performance claim",
        },
        "summaries": summaries,
        "selections": selections,
    }
    report["protocol"]["stage"] = args.stage
    output.mkdir(parents=True, exist_ok=True)
    (output / "cv_results.json").write_text(json.dumps(report, indent=2) + "\n")
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    audit = predictions[
        predictions.evaluation.isin(["baseline", "select_macro"])
    ].copy()
    audit["correct"] = audit.truth == audit.prediction
    audit["side_i_false_alarm"] = (audit.truth != "Side I") & (
        audit.prediction == "Side I"
    )
    audit = (
        audit.groupby(["file_id", "truth", "evaluation"])
        .agg(correct=("correct", "sum"), false_alarms=("side_i_false_alarm", "sum"))
        .reset_index()
        .pivot(
            index=["file_id", "truth"],
            columns="evaluation",
            values=["correct", "false_alarms"],
        )
    )
    audit.columns = ["_".join(column) for column in audit.columns]
    audit = (
        audit.reset_index()
        .set_index("file_id")
        .join(
            features.set_index("file_id")[
                [
                    "speed_mps",
                    "side_i_vibration__rms__mean",
                    "side_ii_vibration__rms__mean",
                ]
            ]
        )
    )
    audit.to_csv(output / "file_audit.csv")
    for name, summary in summaries.items():
        print(name, summary["macro_f1_mean"], summary["per_class_f1_mean"], flush=True)


if __name__ == "__main__":
    main()
