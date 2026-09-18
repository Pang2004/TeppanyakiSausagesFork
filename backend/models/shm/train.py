"""Train and validate the physics-first SHM fatigue model."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import rainflow
import scipy
import sklearn
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import SHMFeatureConfig, extract_features, residual_feature_names
from .metrics import (
    calibrate_miner_scale,
    mean_absolute_percentage_error,
    official_shm_score,
    weighted_median,
)
from .predict import ARTIFACT_VERSION

RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)


def _load_index(data_dir: Path) -> tuple[list[Path], np.ndarray]:
    labels = pd.read_csv(data_dir / "Train_Labels.csv")
    if list(labels.columns) != ["filename", "damage"]:
        raise ValueError("SHM labels must contain filename,damage columns in order.")
    if labels["filename"].duplicated().any():
        raise ValueError("SHM labels contain duplicate filenames.")
    damage = pd.to_numeric(labels["damage"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(damage).all() or np.any(damage <= 0):
        raise ValueError("SHM damage labels must be finite and strictly positive.")
    paths = [data_dir / "Train" / str(filename) for filename in labels["filename"]]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing labelled SHM file: {missing[0]}")
    return paths, damage


def _load_or_extract_features(
    paths: list[Path],
    cache_path: Path,
    config: SHMFeatureConfig,
    jobs: int,
    refresh: bool,
) -> pd.DataFrame:
    expected_ids = [path.name for path in paths]
    if cache_path.is_file() and not refresh:
        cached = pd.read_csv(cache_path)
        if cached.get("file_id", pd.Series(dtype=str)).tolist() == expected_ids:
            print(f"Using cached features from {cache_path}")
            return cached
        print("SHM feature cache does not match the training index; rebuilding it.")
    print(f"Extracting rainflow features from {len(paths)} SHM files...")
    rows = Parallel(n_jobs=jobs, verbose=5)(
        delayed(extract_features)(path, config) for path in paths
    )
    frame = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def _mape_errors(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.abs(actual - predicted) / actual


def _proxy(frame: pd.DataFrame, exponent: int) -> np.ndarray:
    return frame[f"rainflow__miner_proxy_m{exponent}"].to_numpy(dtype=np.float64)


def _select_exponent(
    frame: pd.DataFrame,
    damage: np.ndarray,
    indices: np.ndarray,
    config: SHMFeatureConfig,
) -> int:
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    candidates: list[tuple[float, int]] = []
    for exponent in config.candidate_exponents:
        proxies = _proxy(frame, exponent)
        errors: list[float] = []
        for inner_train, inner_validation in splitter.split(indices):
            train_indices = indices[inner_train]
            validation_indices = indices[inner_validation]
            scale = calibrate_miner_scale(damage[train_indices], proxies[train_indices])
            errors.extend(
                _mape_errors(
                    damage[validation_indices], scale * proxies[validation_indices]
                )
            )
        candidates.append((float(np.mean(errors)), exponent))
    return min(candidates)[1]


def _select_residual_alpha(
    matrix: np.ndarray,
    damage: np.ndarray,
    proxies: np.ndarray,
    indices: np.ndarray,
) -> float:
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    candidates: list[tuple[float, float]] = []
    for alpha in RIDGE_ALPHAS:
        errors: list[float] = []
        for inner_train, inner_validation in splitter.split(indices):
            train_indices = indices[inner_train]
            validation_indices = indices[inner_validation]
            scale = calibrate_miner_scale(damage[train_indices], proxies[train_indices])
            target = np.log(damage[train_indices] / (scale * proxies[train_indices]))
            estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(
                matrix[train_indices], target
            )
            predicted = (
                scale
                * proxies[validation_indices]
                * np.exp(estimator.predict(matrix[validation_indices]))
            )
            errors.extend(_mape_errors(damage[validation_indices], predicted))
        candidates.append((float(np.mean(errors)), alpha))
    return min(candidates)[1]


def _loo_predictions(
    features: pd.DataFrame,
    damage: np.ndarray,
    residual_names: list[str],
    config: SHMFeatureConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[float]]:
    count = len(damage)
    all_indices = np.arange(count)
    matrix = features[residual_names].to_numpy(dtype=np.float64)
    physics_predictions = np.empty(count, dtype=np.float64)
    residual_predictions = np.empty(count, dtype=np.float64)
    selected_exponents: list[int] = []
    selected_alphas: list[float] = []
    for validation_index in all_indices:
        train_indices = all_indices[all_indices != validation_index]
        exponent = _select_exponent(features, damage, train_indices, config)
        proxies = _proxy(features, exponent)
        scale = calibrate_miner_scale(damage[train_indices], proxies[train_indices])
        physics_predictions[validation_index] = scale * proxies[validation_index]
        alpha = _select_residual_alpha(matrix, damage, proxies, train_indices)
        target = np.log(damage[train_indices] / (scale * proxies[train_indices]))
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(
            matrix[train_indices], target
        )
        residual_predictions[validation_index] = (
            scale
            * proxies[validation_index]
            * np.exp(estimator.predict(matrix[[validation_index]])[0])
        )
        selected_exponents.append(exponent)
        selected_alphas.append(alpha)
    return (
        physics_predictions,
        residual_predictions,
        selected_exponents,
        selected_alphas,
    )


def _baseline_results(features: pd.DataFrame, damage: np.ndarray) -> dict[str, float]:
    all_indices = np.arange(len(damage))
    constant_predictions = np.empty(len(damage))
    range_predictions = np.empty(len(damage))
    ranges = features["signal__range"].to_numpy(dtype=np.float64)
    for validation_index in all_indices:
        train_indices = all_indices[all_indices != validation_index]
        constant = weighted_median(damage[train_indices], 1.0 / damage[train_indices])
        constant_predictions[validation_index] = constant
        coefficients = np.polyfit(
            np.log(ranges[train_indices]), np.log(damage[train_indices]), 1
        )
        range_predictions[validation_index] = np.exp(
            np.polyval(coefficients, np.log(ranges[validation_index]))
        )
    return {
        "weighted_constant_mape": mean_absolute_percentage_error(
            damage, constant_predictions
        ),
        "range_power_mape": mean_absolute_percentage_error(damage, range_predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--model-out", type=Path, default=Path("backend/artifacts/shm_pipeline.joblib")
    )
    parser.add_argument(
        "--feature-cache", type=Path, default=Path("outputs/shm/train_features.csv")
    )
    parser.add_argument(
        "--cv-results", type=Path, default=Path("outputs/shm/cv_results.json")
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--refresh-features", action="store_true")
    args = parser.parse_args()
    if args.jobs == 0 or args.jobs < -1:
        parser.error("--jobs must be -1 or a positive integer")

    config = SHMFeatureConfig()
    paths, damage = _load_index(args.data_dir)
    features = _load_or_extract_features(
        paths, args.feature_cache, config, args.jobs, args.refresh_features
    )
    names = residual_feature_names(list(features.columns))
    matrix = features[names].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("SHM training features contain non-finite values.")

    physics, residual, exponents, alphas = _loo_predictions(
        features, damage, names, config
    )
    physics_errors = _mape_errors(damage, physics)
    residual_errors = _mape_errors(damage, residual)
    physics_mape = float(np.mean(physics_errors))
    residual_mape = float(np.mean(residual_errors))
    use_residual = residual_mape <= physics_mape - 0.002 and np.quantile(
        residual_errors, 0.95
    ) <= np.quantile(physics_errors, 0.95)
    selected_errors = residual_errors if use_residual else physics_errors
    baselines = _baseline_results(features, damage)

    all_indices = np.arange(len(damage))
    exponent = _select_exponent(features, damage, all_indices, config)
    proxies = _proxy(features, exponent)
    scale = calibrate_miner_scale(damage, proxies)
    residual_estimator: Any | None = None
    selected_alpha: float | None = None
    if use_residual:
        selected_alpha = _select_residual_alpha(matrix, damage, proxies, all_indices)
        target = np.log(damage / (scale * proxies))
        residual_estimator = make_pipeline(
            StandardScaler(), Ridge(alpha=selected_alpha)
        ).fit(matrix, target)

    results = {
        **baselines,
        "physics_loo_mape": physics_mape,
        "physics_loo_score": official_shm_score(damage, physics),
        "physics_p95_absolute_percentage_error": float(
            np.quantile(physics_errors, 0.95)
        ),
        "residual_loo_mape": residual_mape,
        "residual_loo_score": official_shm_score(damage, residual),
        "residual_p95_absolute_percentage_error": float(
            np.quantile(residual_errors, 0.95)
        ),
        "selected_model": "physics_plus_ridge_residual"
        if use_residual
        else "calibrated_miner",
        "selected_loo_mape": float(np.mean(selected_errors)),
        "selected_loo_score": max(0.0, 1.0 - float(np.mean(selected_errors))),
        "p95_absolute_percentage_error": float(np.quantile(selected_errors, 0.95)),
        "selected_exponent": exponent,
        "outer_exponent_counts": {
            str(value): exponents.count(value) for value in sorted(set(exponents))
        },
        "outer_alpha_counts": {
            str(value): alphas.count(value) for value in sorted(set(alphas))
        },
        "selected_alpha": selected_alpha,
    }
    print(json.dumps(results, indent=2))

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_files": len(paths),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "rainflow": rainflow.__version__,
    }
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "feature_config": config.to_dict(),
        "exponent": exponent,
        "scale": scale,
        "residual_estimator": residual_estimator,
        "residual_feature_names": names,
        "validation": results,
        "metadata": metadata,
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_out, compress=3)
    args.cv_results.parent.mkdir(parents=True, exist_ok=True)
    args.cv_results.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {results['selected_model']} to {args.model_out}")


if __name__ == "__main__":
    main()
