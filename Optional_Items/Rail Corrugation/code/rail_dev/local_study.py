"""Evaluate local spectral features on recording-grouped nested folds."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from backend.models.rail.features import FeatureConfig
from sklearn.base import BaseEstimator, ClassifierMixin

from .local_spectra import load_local
from .side_i_study import StudyClassifier, boosted_prediction
from .train import _load_or_extract_features, _load_training_index
from .validation import (
    OUTER_SEEDS,
    evaluate_nested,
    recording_groups,
    summarize_predictions,
)

OUTPUT = Path("Optional_Items/Rail Corrugation/code/outputs/local_study")


class ColumnClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, names, columns, c=0.1, mirror=False, boost=1.0):
        self.names = names
        self.columns = columns
        self.c = c
        self.mirror = mirror
        self.boost = boost

    def fit(self, x, y):
        self.model_ = StudyClassifier(
            names=self.names, c=self.c, mirror=self.mirror
        ).fit(x[:, self.columns], y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, x):
        return boosted_prediction(self.predict_proba(x), self.boost)

    def predict_proba(self, x):
        return self.model_.predict_proba(x[:, self.columns])


class BlendClassifier(ClassifierMixin, BaseEstimator):
    """Average global and local model scores, fitting both within the fold."""

    def __init__(
        self, names, original_columns, local_columns, c=1.0, local_weight=0.5, boost=1.0
    ):
        self.names = names
        self.original_columns = original_columns
        self.local_columns = local_columns
        self.c = c
        self.local_weight = local_weight
        self.boost = boost

    def fit(self, x, y):
        base_names = tuple(self.names[i] for i in self.original_columns)
        local_names = tuple(self.names[i] for i in self.local_columns)
        self.global_ = StudyClassifier(base_names, c=0.1, log=True, mirror=True).fit(
            x[:, self.original_columns], y
        )
        self.local_ = StudyClassifier(local_names, c=self.c, mirror=True).fit(
            x[:, self.local_columns], y
        )
        self.classes_ = self.global_.classes_
        return self

    def predict_proba(self, x):
        return (1 - self.local_weight) * self.global_.predict_proba(
            x[:, self.original_columns]
        ) + self.local_weight * self.local_.predict_proba(x[:, self.local_columns])

    def predict(self, x):
        return boosted_prediction(self.predict_proba(x), self.boost)


def add_family_summaries(report, predictions):
    """Reconstruct restricted inner selection without looking at outer truth."""
    family_reports = {}
    for family in ("original", "base_local", "base_shape", "local_only", "blend"):
        if not any(
            name.startswith(family + "_") for name in report["fixed_candidates"]
        ):
            continue
        selected_parts = []
        for selection in report["outer_selections"]:
            scores = {
                name: score
                for name, score in selection["inner_macro_f1"].items()
                if name.startswith(family + "_")
            }
            # The baseline matches production's non-augmented C search.
            if family == "original":
                scores = {
                    name: score
                    for name, score in scores.items()
                    if "mirror" not in name
                }
            name = max(scores, key=scores.get)
            selected_parts.append(
                predictions[
                    (predictions.repeat == selection["repeat"])
                    & (predictions.fold == selection["fold"])
                    & (predictions.evaluation == name)
                ]
            )
        family_reports[family] = summarize_predictions(
            pd.concat(selected_parts, ignore_index=True)
        )
    report["within_family_nested_selection"] = family_reports
    return report


def write_file_audit(report, predictions, output):
    parts = []
    for selection in report["outer_selections"]:
        fold = predictions[
            (predictions.repeat == selection["repeat"])
            & (predictions.fold == selection["fold"])
        ]
        scores = {
            name: score
            for name, score in selection["inner_macro_f1"].items()
            if name.startswith("original_C")
        }
        for procedure, name in (
            ("baseline", max(scores, key=scores.get)),
            ("new", "nested_selection"),
        ):
            part = fold[fold.evaluation == name].copy()
            part["procedure"] = procedure
            parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    frame["correct"] = frame.truth == frame.prediction
    frame["side_i_false_alarm"] = (frame.truth != "Side I") & (
        frame.prediction == "Side I"
    )
    audit = (
        frame.groupby(["file_id", "truth", "procedure"])
        .agg(
            correct=("correct", "sum"),
            side_i_false_alarms=("side_i_false_alarm", "sum"),
        )
        .reset_index()
        .pivot(
            index=["file_id", "truth"],
            columns="procedure",
            values=["correct", "side_i_false_alarms"],
        )
    )
    audit.columns = ["_".join(column) for column in audit.columns]
    audit.to_csv(output / "file_audit.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(OUTER_SEEDS))
    parser.add_argument("--report-dir", type=Path, default=OUTPUT)
    parser.add_argument("--tune-decisions", action="store_true")
    parser.add_argument("--blend", action="store_true")
    args = parser.parse_args()
    paths, labels = _load_training_index(Path("PS3/02_Datasets/Rail_Corrugation"))
    y = np.asarray(labels)
    groups = recording_groups(paths, y)
    base = _load_or_extract_features(
        paths,
        OUTPUT.parent / "train_features_wavelength.csv",
        FeatureConfig(),
        4,
        False,
        groups,
    )
    local = load_local(paths, groups, OUTPUT / "local_features.csv")
    features = base.merge(local, on="file_id", validate="one_to_one", sort=False)
    names = tuple(features.columns.drop("file_id"))
    base_names = set(base.columns.drop("file_id"))
    families = {
        "original": [i for i, n in enumerate(names) if n in base_names],
        "base_local": [
            i for i, n in enumerate(names) if n in base_names or "_local_" in n
        ],
        "base_shape": [
            i for i, n in enumerate(names) if n in base_names or "_shape_" in n
        ],
        "local_only": [
            i
            for i, n in enumerate(names)
            if n not in base_names or n.startswith("speed")
        ],
    }
    candidates = {}
    for family, columns in families.items():
        selected_names = tuple(names[i] for i in columns)
        for c in (0.01, 0.1, 1.0, 10.0):
            candidates[f"{family}_C{c}"] = ColumnClassifier(
                selected_names, columns, c=c
            )
        for c in (0.1, 1.0):
            candidates[f"{family}_mirror_C{c}"] = ColumnClassifier(
                selected_names, columns, c=c, mirror=True
            )
    if args.tune_decisions:
        candidates = {
            name: model
            for name, model in candidates.items()
            if name.startswith("original_C")
        }
        columns = families["base_local"]
        selected_names = tuple(names[i] for i in columns)
        for c in (0.01, 0.1, 1.0, 10.0):
            for boost in (0.5, 1.0, 2.0, 4.0):
                candidates[f"base_local_mirror_C{c}_boost{boost}"] = ColumnClassifier(
                    selected_names, columns, c=c, mirror=True, boost=boost
                )
    if args.blend:
        candidates = {
            name: model
            for name, model in candidates.items()
            if name.startswith("original_C")
        }
        for c in (0.1, 1.0):
            for weight in (0.25, 0.5, 0.75):
                for boost in (0.5, 1.0, 2.0):
                    candidates[f"blend_C{c}_weight{weight}_boost{boost}"] = (
                        BlendClassifier(
                            names,
                            families["original"],
                            families["base_local"],
                            c=c,
                            local_weight=weight,
                            boost=boost,
                        )
                    )
    x = features[list(names)].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite local study features")
    report, predictions = evaluate_nested(
        candidates, x, y, groups, [p.name for p in paths], outer_seeds=tuple(args.seeds)
    )
    add_family_summaries(report, predictions)
    report["protocol"]["limitations"].append(
        "Local feature families were proposed after inspecting previous errors. "
        "This is exploratory validation, not an independent test."
    )
    report["feature_counts"] = {
        name: len(columns) for name, columns in families.items()
    }
    report["protocol"]["tune_decisions"] = args.tune_decisions
    report["protocol"]["blend"] = args.blend
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "cv_results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    predictions.to_csv(args.report_dir / "oof_predictions.csv", index=False)
    write_file_audit(report, predictions, args.report_dir)
    for name, result in {
        **report["fixed_candidates"],
        "nested_selection": report["nested_selection"],
    }.items():
        print(name, result["macro_f1_mean"], result["per_class_f1_mean"], flush=True)


if __name__ == "__main__":
    main()
