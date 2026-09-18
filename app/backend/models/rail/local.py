"""Runtime features preserving local wavelength energy and spectral shape."""

import itertools
from pathlib import Path

import numpy as np
from scipy.signal import welch

from .features import load_recording
from .wavelength import WAVELENGTH_EDGES, speed_metres_per_second

CONFIG = {
    "version": "local-spectra-v1",
    "sample_rate": 10000,
    "nperseg": 2048,
    "wavelength_edges": WAVELENGTH_EDGES,
    "pooling": "car mean, paired axle contrasts, channel shape quantiles",
}


def integrate_band(freq, psd, left, right):
    left, right = max(float(left), freq[0]), min(float(right), freq[-1])
    if right <= left:
        return np.zeros(psd.shape[1])
    mask = (freq > left) & (freq < right)
    edges = np.array(
        [
            [np.interp(bound, freq, psd[:, c]) for c in range(psd.shape[1])]
            for bound in (left, right)
        ]
    )
    return np.trapezoid(
        np.vstack((edges[0], psd[mask], edges[1])),
        np.r_[left, freq[mask], right],
        axis=0,
    )


def spectral_rows(matrix, layout, speed):
    """No labels or dataset statistics are used in feature extraction."""
    row = {}
    for kind in ("vibration", "shock"):
        lookup = {(s.car, s.position): s.index for s in layout if s.signal_type == kind}
        # Matched positions in canonical car/axle order, independent of CSV order.
        indices = [
            lookup[car, position]
            for side in (0, 1)
            for car in range(1, 9)
            for position in range(1 + side, 9, 2)
        ]
        values = matrix[:, indices]
        freq, psd = welch(values, fs=10000, nperseg=2048, axis=0)
        total = np.maximum(np.trapezoid(psd, freq, axis=0), 1e-15)
        bands = {}
        for low, high in itertools.pairwise(WAVELENGTH_EDGES):
            power = integrate_band(freq, psd, speed / high, speed / low)
            bands[f"wave_{low}_{high}_energy"] = power
            bands[f"wave_{low}_{high}_fraction"] = power / total
        bands["total_energy"] = total
        for name, vector in bands.items():
            pair = vector.reshape(2, 8, 4)
            for side, channels in zip(("i", "ii"), pair, strict=True):
                for car, value in enumerate(np.mean(channels, axis=1), 1):
                    # Same prefix convention as the verified mirroring utility.
                    row[f"{kind}_{side}_local_{name}_car{car}"] = float(np.log1p(value))
            contrast = (pair[0] - pair[1]) / (np.abs(pair[0]) + np.abs(pair[1]) + 1e-15)
            for car, value in enumerate(contrast.mean(axis=1), 1):
                row[f"{kind}_contrast_local_{name}_car{car}"] = float(value)
            # Odd transforms commute with exchanging sides.
            row[f"{kind}_contrast_local_{name}_median"] = float(np.median(contrast))
            row[f"{kind}_contrast_local_{name}_agreement"] = float(
                np.mean(np.sign(contrast))
            )
            row[f"{kind}_contrast_local_{name}_strong"] = float(np.mean(contrast**3))

        normalized = psd[1:] / np.maximum(psd[1:].sum(axis=0), 1e-15)
        entropy = -np.sum(normalized * np.log(np.maximum(normalized, 1e-15)), axis=0)
        peak = np.max(normalized, axis=0)
        top5 = np.sort(normalized, axis=0)[-5:].sum(axis=0)
        flatness = np.exp(np.log(np.maximum(psd[1:], 1e-15)).mean(axis=0)) / np.maximum(
            psd[1:].mean(axis=0), 1e-15
        )
        for name, vector in {
            "entropy": entropy,
            "peak": peak,
            "top5": top5,
            "flatness": flatness,
        }.items():
            pair = vector.reshape(2, 8, 4)
            for side, channels in zip(("i", "ii"), pair, strict=True):
                for car, value in enumerate(channels.mean(axis=1), 1):
                    row[f"{kind}_{side}_shape_{name}_car{car}"] = float(value)
            contrast = (pair[0] - pair[1]) / (np.abs(pair[0]) + np.abs(pair[1]) + 1e-15)
            for car, value in enumerate(contrast.mean(axis=1), 1):
                row[f"{kind}_contrast_shape_{name}_car{car}"] = float(value)
    return row


def extract_local(path):
    frame, layout = load_recording(path)
    matrix = frame.to_numpy(float)
    row = {
        "file_id": Path(path).name,
        **spectral_rows(matrix, layout, speed_metres_per_second(matrix[:, 0])),
    }
    if not np.isfinite(np.array(list(row.values())[1:], dtype=float)).all():
        raise ValueError(f"Non-finite local spectra: {path}")
    return row


FEATURE_SET = "base_local_spectra_v1"


def extract_combined(path):
    from .wavelength import extract_production_features

    return {**extract_production_features(path), **extract_local(path)}
