"""Physical-unit checks for production wavelength features."""

import numpy as np
import pytest
from backend.models.rail.wavelength import speed_metres_per_second, wavelength_fraction
from scipy import signal


def test_speed_uses_both_edges_per_tooth():
    tachometer = np.tile(np.r_[np.zeros(5), np.ones(5)], 1000)
    edges = np.count_nonzero(np.diff(tachometer))
    expected = edges / 0.9999 / 180 * np.pi * 0.85
    assert speed_metres_per_second(tachometer) == pytest.approx(expected)
    assert speed_metres_per_second(np.zeros(10000)) == 0


def test_wavelength_band_tracks_frequency_change_with_speed():
    t = np.arange(10000) / 10000
    fractions = []
    for speed, frequency in [(10.0, 400.0), (20.0, 800.0)]:
        frequencies, power = signal.welch(
            np.sin(2 * np.pi * frequency * t)[:, None], fs=10000, nperseg=2048, axis=0
        )
        fractions.append(wavelength_fraction(frequencies, power, speed, 0.02, 0.04)[0])
        assert wavelength_fraction(frequencies, power, 0, 0.02, 0.04)[0] == 0
    assert min(fractions) > 0.99
    assert fractions[0] == pytest.approx(fractions[1], abs=0.001)
