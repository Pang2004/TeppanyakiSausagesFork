"""File-level validation with SHM family selection inside every outer fold."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import calibrate_miner_scale

RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
MINIMUM_RESIDUAL_GAIN = 0.002


def error_summary(actual, predicted):
    errors = np.abs(actual - predicted) / actual
    return {
        "mape": float(np.mean(errors)),
        "official_score": max(0.0, 1.0 - float(np.mean(errors))),
        "p95_absolute_percentage_error": float(np.quantile(errors, 0.95)),
        "maximum_absolute_percentage_error": float(np.max(errors)),
    }


def fit_residual(matrix, damage, proxies, alpha):
    scale = calibrate_miner_scale(damage, proxies)
    target = np.log(damage / (scale * proxies))
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(matrix, target)
    return scale, model


def select_configuration(features, damage, names, config, seed=42):
    """Select exponent, alpha and family using only the supplied training files.

    Exponent is selected for the physical model first, preserving the existing
    procedure. The residual uses that exponent. The existing gain/tail gate now
    uses inner predictions instead of the outer evaluation labels.
    """
    matrix = features[names].to_numpy(dtype=float)
    splits = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(damage))
    physics_candidates = []
    for exponent in config.candidate_exponents:
        proxies = features[f"rainflow__miner_proxy_m{exponent}"].to_numpy(dtype=float)
        predicted = np.empty(len(damage))
        for train, validation in splits:
            scale = calibrate_miner_scale(damage[train], proxies[train])
            predicted[validation] = scale * proxies[validation]
        physics_candidates.append(
            {"exponent": exponent, **error_summary(damage, predicted)}
        )
    physics = min(physics_candidates, key=lambda row: (row["mape"], row["exponent"]))
    proxies = features[f"rainflow__miner_proxy_m{physics['exponent']}"].to_numpy(
        dtype=float
    )
    residual_candidates = []
    for alpha in RIDGE_ALPHAS:
        predicted = np.empty(len(damage))
        for train, validation in splits:
            scale, model = fit_residual(
                matrix[train], damage[train], proxies[train], alpha
            )
            predicted[validation] = (
                scale * proxies[validation] * np.exp(model.predict(matrix[validation]))
            )
        residual_candidates.append({"alpha": alpha, **error_summary(damage, predicted)})
    residual = min(residual_candidates, key=lambda row: (row["mape"], row["alpha"]))
    use_residual = (
        residual["mape"] <= physics["mape"] - MINIMUM_RESIDUAL_GAIN
        and residual["p95_absolute_percentage_error"]
        <= physics["p95_absolute_percentage_error"]
    )
    return {
        "family": "physics_plus_ridge_residual" if use_residual else "calibrated_miner",
        "exponent": physics["exponent"],
        "alpha": residual["alpha"],
        "inner_seed": seed,
        "physics_candidates": physics_candidates,
        "residual_candidates": residual_candidates,
    }


def evaluate_nested(features, damage, names, config, seed=42):
    """LOO predictions for two fixed families and the complete selection rule."""
    matrix = features[names].to_numpy(dtype=float)
    rows = []
    for held_out in range(len(damage)):
        train = np.delete(np.arange(len(damage)), held_out)
        selected = select_configuration(
            features.iloc[train], damage[train], names, config, seed
        )
        proxies = features[f"rainflow__miner_proxy_m{selected['exponent']}"].to_numpy(
            dtype=float
        )
        scale, model = fit_residual(
            matrix[train], damage[train], proxies[train], selected["alpha"]
        )
        physics = float(scale * proxies[held_out])
        residual = float(physics * np.exp(model.predict(matrix[[held_out]])[0]))
        prediction = (
            residual if selected["family"] == "physics_plus_ridge_residual" else physics
        )
        rows.append(
            {
                "file_id": str(features.iloc[held_out]["file_id"]),
                "actual": float(damage[held_out]),
                "physics_prediction": physics,
                "residual_prediction": residual,
                "nested_prediction": prediction,
                "train_indices": train.tolist(),
                "validation_index": held_out,
                "selection": selected,
            }
        )
    summary = {
        name: error_summary(damage, np.asarray([row[key] for row in rows]))
        for name, key in (
            ("calibrated_miner", "physics_prediction"),
            ("physics_plus_ridge_residual", "residual_prediction"),
            ("nested_selection", "nested_prediction"),
        )
    }
    return {
        "protocol": "leave_one_file_out_with_inner_5_fold_family_selection",
        "inner_seed": seed,
        "minimum_residual_mape_gain": MINIMUM_RESIDUAL_GAIN,
        "summary": summary,
        "outer_family_counts": dict(
            Counter(row["selection"]["family"] for row in rows)
        ),
        "outer_exponent_counts": dict(
            Counter(str(row["selection"]["exponent"]) for row in rows)
        ),
        "outer_alpha_counts": dict(
            Counter(str(row["selection"]["alpha"]) for row in rows)
        ),
        "final_selection": select_configuration(features, damage, names, config, seed),
        "folds": rows,
    }


def main():
    from backend.models.shm.features import SHMFeatureConfig, residual_feature_names

    from .train import _load_index, _load_or_extract_features

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path("Optional_Items/SHM/code/outputs/train_features.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Optional_Items/SHM/code/outputs/validation_sensitivity.json"),
    )
    parser.add_argument("--inner-seeds", nargs="+", type=int, default=[43, 44])
    args = parser.parse_args()
    paths, damage = _load_index(args.data_dir)
    config = SHMFeatureConfig()
    features = _load_or_extract_features(paths, args.feature_cache, config, 1, False)
    names = residual_feature_names(list(features))
    results = [
        evaluate_nested(features, damage, names, config, seed)
        for seed in args.inner_seeds
    ]
    report = {
        "purpose": "Sensitivity only; primary inner seed remains 42. No seed selected by score.",
        "input_provenance": json.loads(
            args.feature_cache.with_suffix(".manifest.json").read_text()
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for result in results:
        print(
            result["inner_seed"],
            result["summary"]["nested_selection"],
            result["outer_family_counts"],
        )


if __name__ == "__main__":
    main()
