"""MAPE-based SHM evaluation and robust Miner calibration."""

from __future__ import annotations

import numpy as np


def mean_absolute_percentage_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return MAPE for strictly positive SHM reference damage values."""

    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if np.any(actual <= 0):
        raise ValueError("SHM reference damage values must be positive for MAPE.")
    return float(np.mean(np.abs(actual - predicted) / actual))


def official_shm_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return the official max(0, 1 - MAPE) score."""

    return max(0.0, 1.0 - mean_absolute_percentage_error(actual, predicted))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Calculate a deterministic weighted median."""

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(sorted_values[np.searchsorted(cumulative, cumulative[-1] / 2.0)])


def calibrate_miner_scale(actual: np.ndarray, proxies: np.ndarray) -> float:
    """Fit the multiplicative Miner scale that minimizes training MAPE."""

    actual = np.asarray(actual, dtype=np.float64)
    proxies = np.asarray(proxies, dtype=np.float64)
    valid = (actual > 0) & (proxies > 0)
    if not np.any(valid):
        raise ValueError(
            "Cannot calibrate SHM damage without positive labels and proxies."
        )
    ratios = actual[valid] / proxies[valid]
    relative_error_weights = proxies[valid] / actual[valid]
    return weighted_median(ratios, relative_error_weights)
