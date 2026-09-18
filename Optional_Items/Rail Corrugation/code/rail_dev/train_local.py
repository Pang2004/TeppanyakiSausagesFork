"""Train the local-spectrum Rail candidate with grouped decision selection."""

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from backend.models.rail.features import FeatureConfig
from backend.models.rail.local import CONFIG, FEATURE_SET, extract_combined
from backend.models.rail.local_model import LocalRailClassifier
from backend.models.rail.predict import LOCAL_ARTIFACT_VERSION, RAIL_LABELS
from backend.models.rail.wavelength import WAVELENGTH_CONFIG

from .local_spectra import load_local
from .train import _load_or_extract_features, _load_training_index
from .validation import (
    evaluate_nested,
    fit_grouped,
    grouped_splits,
    recording_groups,
    select_candidate,
)

OUTPUT = Path("Optional_Items/Rail Corrugation/code/outputs")


def candidates_for(names):
    base = [i for i, n in enumerate(names) if "_local_" not in n and "_shape_" not in n]
    local = [i for i, n in enumerate(names) if "_shape_" not in n]
    candidates = {}
    for c in (0.01, 0.1, 1.0, 10.0):
        candidates[f"original_C{c}"] = LocalRailClassifier(
            tuple(names[i] for i in base), base, c=c
        )
    for c in (0.01, 0.1, 1.0, 10.0):
        for boost in (0.5, 1.0, 2.0, 4.0):
            candidates[f"base_local_mirror_C{c}_boost{boost}"] = LocalRailClassifier(
                tuple(names[i] for i in local), local, c=c, mirror=True, boost=boost
            )
    return candidates


def replay(report_path, candidates, x, y, groups, ids):
    report = json.loads(report_path.read_text())
    if list(candidates) != report["protocol"]["candidate_order"]:
        raise ValueError("Research candidate grid does not match production")
    source = pd.read_csv(report_path.parent / "oof_predictions.csv")
    source = source[source.evaluation == "nested_selection"].copy()
    rows = []
    for repeat, seed in enumerate(report["protocol"]["outer_seeds"], 1):
        for fold, (train, valid) in enumerate(grouped_splits(y, groups, 5, seed), 1):
            selected = next(
                s
                for s in report["outer_selections"]
                if s["repeat"] == repeat and s["fold"] == fold
            )
            if set(groups[train]) != set(selected["train_groups"]) or set(
                groups[valid]
            ) != set(selected["validation_groups"]):
                raise ValueError("Research recording groups do not match")
            name = selected["selected_model"]
            if not name.startswith("base_local_"):
                raise ValueError("Replay selected a different feature family; run fresh local-family validation")
            model = fit_grouped(candidates[name], x[train], y[train], groups[train])
            for index, prediction in zip(valid, model.predict(x[valid]), strict=True):
                rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "file_id": ids[index],
                        "group_id": groups[index],
                        "truth": y[index],
                        "prediction": prediction,
                        "evaluation": "nested_selection",
                        "selected_model": name,
                    }
                )
    predictions = pd.DataFrame(rows)
    keys = ["repeat", "fold", "file_id", "group_id", "truth", "prediction"]
    if (
        not predictions[keys]
        .sort_values(keys)
        .reset_index(drop=True)
        .equals(source[keys].sort_values(keys).reset_index(drop=True))
    ):
        raise ValueError(
            "Production predictions do not exactly reproduce research OOF predictions"
        )
    report = copy.deepcopy(report)
    report["production_verification"] = {
        "source_report": str(report_path),
        "predictions_reproduced": len(predictions),
        "scope": "Refit selected estimator on every outer training fold. Inner-search report retained from research; not rerun in replay mode.",
    }
    return report, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("app/backend/artifacts/rail_pipeline.joblib"),
    )
    parser.add_argument("--cv-results", type=Path, default=OUTPUT / "cv_results.json")
    parser.add_argument(
        "--oof-predictions", type=Path, default=OUTPUT / "oof_predictions.csv"
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--replay-report",
        type=Path,
        help="Verify a completed research evaluation instead of repeating its inner search",
    )
    args = parser.parse_args()
    paths, labels = _load_training_index(args.data_dir)
    y = np.asarray(labels)
    groups = recording_groups(paths, y)
    base = _load_or_extract_features(
        paths,
        OUTPUT / "train_features_wavelength.csv",
        FeatureConfig(),
        args.jobs,
        False,
        groups,
    )
    local = load_local(
        paths, groups, OUTPUT / "local_study/local_features.csv", args.jobs
    )
    features = base.merge(local, on="file_id", validate="one_to_one", sort=False)
    names = tuple(features.columns.drop("file_id"))
    x = features[list(names)].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite Rail training features")
    # Re-extract representative normal and difficult fault recordings from raw input.
    for path in [
        paths[0],
        *[
            p
            for p in paths
            if p.name
            in ("Train106.csv", "Train150.csv", "Train180.csv", "Train185.csv")
        ],
    ]:
        raw = extract_combined(path)
        np.testing.assert_allclose(
            [raw[n] for n in names],
            x[[p.name for p in paths].index(path.name)],
            rtol=1e-9,
            atol=1e-10,
        )
    candidates = candidates_for(names)
    local_candidates = {
        name: model
        for name, model in candidates.items()
        if name.startswith("base_local_")
    }
    ids = [p.name for p in paths]
    report, predictions = (
        replay(args.replay_report, candidates, x, y, groups, ids)
        if args.replay_report
        else evaluate_nested(
            local_candidates, x, y, groups, ids, outer_seeds=(42, 43, 44, 45, 46, 47)
        )
    )
    selected, scores = select_candidate(local_candidates, x, y, groups)
    baseline_selected, baseline_scores = select_candidate(
        {
            name: model
            for name, model in candidates.items()
            if name.startswith("original_")
        },
        x,
        y,
        groups,
    )
    estimator = fit_grouped(candidates[selected], x, y, groups)
    report["feature_set"] = FEATURE_SET
    report["production_family"] = (
        "Original plus local energy, mirrored training. Family adopted after exploratory comparison; only C and decision multiplier are inner-tuned. Original models are comparison baselines."
    )
    report["full_data_baseline_comparison"] = {
        "selected_model": baseline_selected,
        "inner_macro_f1": baseline_scores,
    }
    report["final_selection"] = {
        "selected_model": selected,
        "inner_macro_f1": scores,
        "note": "Full-data inner score selects the model, not its generalisation estimate.",
    }
    report["protocol"]["limitations"].append(
        "Local features and decision search were adopted after exploratory comparison; no independent test performance claim."
    )
    artifact = {
        "artifact_version": LOCAL_ARTIFACT_VERSION,
        "feature_set": FEATURE_SET,
        "feature_config": FeatureConfig().to_dict(),
        "wavelength_config": WAVELENGTH_CONFIG,
        "local_config": CONFIG,
        "feature_names": names,
        "labels": list(RAIL_LABELS),
        "estimator": estimator,
        "selected_model": selected,
        "cv_results": report,
        "metadata": {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_samples": len(y),
            "unique_recording_groups": len(set(groups)),
            "class_counts": {label: labels.count(label) for label in RAIL_LABELS},
            "decision_multiplier_side_i": estimator.boost,
        },
    }
    for path in (args.model_out, args.cv_results, args.oof_predictions):
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_out, compress=3)
    args.cv_results.write_text(json.dumps(report, indent=2) + "\n")
    predictions.to_csv(args.oof_predictions, index=False)
    print(f"Selected {selected}; saved {args.model_out}", flush=True)
    print(
        report["nested_selection"]["macro_f1_mean"],
        report["nested_selection"]["per_class_f1_mean"],
        flush=True,
    )


if __name__ == "__main__":
    main()
