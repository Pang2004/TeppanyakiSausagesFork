"""Versioned speed-adjusted Rail spectra shared by training and inference."""

from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy import signal

from .features import FeatureConfig, extract_features, load_recording

WAVELENGTH_EDGES = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64)
FEATURE_SET = "base_wavelength_v1"
WAVELENGTH_CONFIG = {
    "schema_version": "rail-wavelength-v1",
    "sample_rate_hz": 10000,
    "welch_nperseg": 2048,
    "wheel_diameter_metres": 0.85,
    "edges_per_rotation": 180,
    "speed_blocks": 10,
    "wavelength_edges_metres": list(WAVELENGTH_EDGES),
}


def speed_metres_per_second(tachometer: np.ndarray, sample_rate=10000) -> float:
    """Each tooth contributes a rising and falling edge; 90 teeth per rotation."""
    if len(tachometer) < 2:
        return 0.0
    binary = np.asarray(tachometer) > 0.5
    transitions = np.count_nonzero(np.diff(binary))
    return float(transitions / ((len(binary) - 1) / sample_rate) / 180 * np.pi * 0.85)


def wavelength_fraction(frequencies, power, speed, low, high):
    """Integrate PSD in the frequency interval speed/high .. speed/low."""
    if speed <= 0:
        return np.zeros(power.shape[1])
    # Interpolate exact band boundaries; this avoids empty narrow-bin artifacts.
    left, right = max(frequencies[0], speed / high), min(frequencies[-1], speed / low)
    if right <= left:
        return np.zeros(power.shape[1])
    interior = (frequencies > left) & (frequencies < right)
    grid = np.concatenate(([left], frequencies[interior], [right]))
    edges = np.array(
        [
            [np.interp(bound, frequencies, power[:, c]) for c in range(power.shape[1])]
            for bound in (left, right)
        ]
    )
    selected = np.vstack((edges[0], power[interior], edges[1]))
    total = np.maximum(np.trapezoid(power, frequencies, axis=0), 1e-12)
    return np.trapezoid(selected, grid, axis=0) / total


def extract_wavelength(path: Path):
    frame, layout = load_recording(path)
    matrix = frame.to_numpy(float)
    speed = speed_metres_per_second(matrix[:, 0])
    row = {
        "file_id": path.name,
        "speed_mps": speed,
        "speed_available": float(speed > 0),
    }
    blocks = np.array_split(matrix[:, 0], 10)
    row["speed_block_cv"] = float(
        np.std([speed_metres_per_second(b) for b in blocks]) / max(speed, 1e-12)
    )
    for kind in ("vibration", "shock"):
        side_features = {}
        for side in ("i", "ii"):
            values = matrix[
                :, [s.index for s in layout if s.side == side and s.signal_type == kind]
            ]
            frequencies, power = signal.welch(values, fs=10000, nperseg=2048, axis=0)
            quantities = {
                f"wavelength_{low}_{high}": wavelength_fraction(
                    frequencies, power, speed, low, high
                )
                for low, high in pairwise(WAVELENGTH_EDGES)
            }
            dominant = frequencies[1:][np.argmax(power[1:], axis=0)]
            quantities["dominant_wavenumber"] = (
                dominant / speed if speed > 0 else np.zeros(32)
            )
            local = {}
            for name, v in quantities.items():
                for agg, val in (
                    ("mean", np.mean(v)),
                    ("median", np.median(v)),
                    ("std", np.std(v)),
                    ("max", np.max(v)),
                ):
                    key = f"{name}_{agg}"
                    local[key] = float(val)
                    row[f"{kind}_{side}_{key}"] = float(val)
            side_features[side] = local
        for name, left in side_features["i"].items():
            right = side_features["ii"][name]
            row[f"{kind}_contrast_{name}"] = (left - right) / (
                abs(left) + abs(right) + 1e-12
            )
    if not np.isfinite(np.array(list(row.values())[1:], dtype=float)).all():
        raise ValueError(f"Nonfinite wavelength features: {path.name}")
    return row


def extract_production_features(path: str | Path, config: FeatureConfig | None = None):
    """Extract the original 684 features and the validated 171 spectral features."""
    config = config or FeatureConfig()
    if config != FeatureConfig():
        raise ValueError(
            "The wavelength feature set requires the validated default base configuration."
        )
    return {**extract_features(path, config), **extract_wavelength(Path(path))}
