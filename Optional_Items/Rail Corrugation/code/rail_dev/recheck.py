"""Reproduce grouped validation and compare a bounded classifier search.

Run with PYTHONPATH='app:Optional_Items/Rail Corrugation/code' python -m
rail_dev.recheck. This writes research reports only, never a production model.
"""

import json
from pathlib import Path

import numpy as np
from backend.models.rail.features import FeatureConfig
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .train import _candidate_models, _load_or_extract_features, _load_training_index
from .validation import evaluate_nested, recording_groups


def main():
    output = Path("Optional_Items/Rail Corrugation/code/outputs/recheck")
    paths, labels = _load_training_index(Path("PS3/02_Datasets/Rail_Corrugation"))
    y = np.asarray(labels)
    groups = recording_groups(paths, y)
    frame = _load_or_extract_features(
        paths,
        output.parent / "train_features_wavelength.csv",
        FeatureConfig(),
        4,
        False,
        groups,
    )
    x = frame.drop(columns="file_id").to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite training features")
    candidates = _candidate_models(1)
    for c in (0.1, 1.0, 10.0):
        candidates[f"balanced_rbf_svc_C_{c}"] = make_pipeline(
            StandardScaler(), SVC(C=c, class_weight="balanced", gamma="scale")
        )
    candidates["shrinkage_lda"] = make_pipeline(
        StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    )
    report, predictions = evaluate_nested(
        candidates, x, y, groups, [path.name for path in paths]
    )
    report["protocol"]["limitations"].append(
        "Exploratory recheck on previously inspected data and feature family; "
        "not a hidden-test score. Fixed candidates must not be confused with "
        "inner-selected performance."
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "cv_results.json").write_text(json.dumps(report, indent=2) + "\n")
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    for name, result in {
        **report["fixed_candidates"],
        "nested_selection": report["nested_selection"],
    }.items():
        print(name, result["macro_f1_mean"], result["per_class_f1_mean"], flush=True)


if __name__ == "__main__":
    main()
