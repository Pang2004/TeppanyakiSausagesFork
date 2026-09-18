"""Physical and localisation checks for the local spectral research features."""

import numpy as np
import pytest
from backend.models.rail.features import parse_sensor_layout
from rail_dev.local_spectra import integrate_band, spectral_rows
from scipy.signal import welch


def test_band_integration_tracks_speed_and_handles_stationary_recordings():
    t = np.arange(10000) / 10000
    for speed, frequency in ((10, 400), (20, 800)):
        freq, psd = welch(
            np.sin(2 * np.pi * frequency * t)[:, None], fs=10000, nperseg=2048, axis=0
        )
        power = integrate_band(freq, psd, speed / 0.04, speed / 0.02)[0]
        assert power == pytest.approx(0.5, abs=0.001)
        assert integrate_band(freq, psd, 0, 0)[0] == 0


def test_local_band_identifies_car_and_side():
    columns = ["Rotating speed"] + [
        f"{kind} of bearing in position {position} of car {car}"
        for car in range(1, 9)
        for position in range(1, 9)
        for kind in ("Vibration", "Shock")
    ]
    layout = parse_sensor_layout(columns)
    t = np.arange(10000) / 10000
    matrix = np.zeros((10000, 129))
    for sensor in layout:
        amplitude = 3 if sensor.car == 3 and sensor.side == "i" else 1
        matrix[:, sensor.index] = amplitude * np.sin(2 * np.pi * 400 * t)
    row = spectral_rows(matrix, layout, 10)
    assert (
        row["vibration_i_local_wave_0.02_0.04_energy_car3"]
        > row["vibration_ii_local_wave_0.02_0.04_energy_car3"]
    )
    assert row["vibration_contrast_local_wave_0.02_0.04_energy_car3"] == pytest.approx(
        0.8, abs=0.001
    )
    assert row["vibration_contrast_local_wave_0.02_0.04_energy_car2"] == pytest.approx(
        0
    )
