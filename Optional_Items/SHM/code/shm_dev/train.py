"""Train and validate the physics-first SHM fatigue model."""

from __future__ import annotations

import argparse
import hashlib
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
from backend.models.shm import features as feature_module
from backend.models.shm.features import (
    SHMFeatureConfig,
    extract_features,
    residual_feature_names,
)
from backend.models.shm.predict import ARTIFACT_VERSION
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import (
    calibrate_miner_scale,
    mean_absolute_percentage_error,
    weighted_median,
)
from .validation import evaluate_nested


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
    manifest_path = cache_path.with_suffix(".manifest.json")
    identity = {
        "feature_config": json.loads(json.dumps(config.to_dict())),
        "raw_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "extractor_sha256": hashlib.sha256(
            Path(feature_module.__file__).read_bytes()
        ).hexdigest(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "rainflow": rainflow.__version__,
    }
    hashes = list(identity["raw_sha256"].values())
    if len(set(hashes)) != len(hashes):
        raise ValueError(
            "Duplicate SHM histories require grouped validation before training."
        )
    if cache_path.is_file() and manifest_path.is_file() and not refresh:
        try:
            manifest = json.loads(manifest_path.read_text())
            cache_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()
            if manifest == {**identity, "cache_sha256": cache_hash}:
                cached = pd.read_csv(cache_path)
                if cached.get("file_id", pd.Series(dtype=str)).tolist() == expected_ids:
                    print(f"Using verified cached features from {cache_path}")
                    return cached
        except (ValueError, OSError):
            pass
    print("SHM feature cache requires extraction or provenance verification.")
    print(f"Extracting rainflow features from {len(paths)} SHM files...")
    rows = Parallel(n_jobs=jobs, verbose=5)(
        delayed(extract_features)(path, config) for path in paths
    )
    frame = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                **identity,
                "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    return frame


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
        "--model-out",
        type=Path,
        default=Path("app/backend/artifacts/shm_pipeline.joblib"),
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path("Optional_Items/SHM/code/outputs/train_features.csv"),
    )
    parser.add_argument(
        "--cv-results",
        type=Path,
        default=Path("Optional_Items/SHM/code/outputs/cv_results.json"),
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

    nested = evaluate_nested(features, damage, names, config)
    final_selection = nested["final_selection"]
    use_residual = final_selection["family"] == "physics_plus_ridge_residual"
    exponent = final_selection["exponent"]
    proxies = features[f"rainflow__miner_proxy_m{exponent}"].to_numpy(dtype=float)
    scale = calibrate_miner_scale(damage, proxies)
    residual_estimator: Any | None = None
    selected_alpha: float | None = None
    if use_residual:
        selected_alpha = final_selection["alpha"]
        target = np.log(damage / (scale * proxies))
        residual_estimator = make_pipeline(
            StandardScaler(), Ridge(alpha=selected_alpha)
        ).fit(matrix, target)

    physics = nested["summary"]["calibrated_miner"]
    residual = nested["summary"]["physics_plus_ridge_residual"]
    selected = nested["summary"]["nested_selection"]
    results = {
        **_baseline_results(features, damage),
        "physics_loo_mape": physics["mape"],
        "physics_loo_score": physics["official_score"],
        "physics_p95_absolute_percentage_error": physics[
            "p95_absolute_percentage_error"
        ],
        "residual_loo_mape": residual["mape"],
        "residual_loo_score": residual["official_score"],
        "residual_p95_absolute_percentage_error": residual[
            "p95_absolute_percentage_error"
        ],
        "selected_model": final_selection["family"],
        "selected_loo_mape": selected["mape"],
        "selected_loo_score": selected["official_score"],
        "p95_absolute_percentage_error": selected["p95_absolute_percentage_error"],
        "error_indicator_kind": "historical_outer_validation_p95_absolute_percentage_error",
        "error_indicator_description": "Same historical error percentile for every file; not a per-file prediction interval or coverage guarantee.",
        "selected_exponent": exponent,
        "outer_exponent_counts": nested["outer_exponent_counts"],
        "outer_alpha_counts": nested["outer_alpha_counts"],
        "selected_alpha": selected_alpha,
        "nested_validation": nested,
        "input_provenance": {
            "features": json.loads(
                args.feature_cache.with_suffix(".manifest.json").read_text()
            ),
            "labels_sha256": hashlib.sha256(
                (args.data_dir / "Train_Labels.csv").read_bytes()
            ).hexdigest(),
        },
    }
    print(
        json.dumps(
            {
                key: value
                for key, value in results.items()
                if key not in ("nested_validation", "input_provenance")
            },
            indent=2,
        )
    )

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_files": len(paths),
        "validation_protocol": nested["protocol"],
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
