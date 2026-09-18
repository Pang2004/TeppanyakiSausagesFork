"""
Core ACV (refrigerant leak) ranking logic.

No dependency on Streamlit, Flask, or any app framework — this is a plain
Python module so it can be dropped into any app-maker, wrapped in a CLI,
exposed as a REST endpoint, or imported directly.

Status: heuristic (not a trained ML model), validated against 6 labeled
training cases (perfect rank-decay score) + stability-tested via
bootstrap resampling. See project notes for caveats (small sample size,
one weak-margin case) before treating this as production-grade.
"""
import io
import re
from typing import Union

import numpy as np
import pandas as pd


def _get_car_numbers(columns):
    return sorted(set(re.findall(r"Car (\d+) -", " ".join(columns))))


def _score_cars(df: pd.DataFrame, car_nums: list) -> dict:
    """Internal: compute a raw anomaly score per car. Higher = more
    likely to be the faulty car."""
    cols = list(df.columns)
    has_indoor_temp = any("Indoor Average Temperature" in c for c in cols)
    has_compressor = any(
        "Compressor 1 Running" in c or "Compressor 2 Running" in c for c in cols
    )

    scores = {}

    if has_indoor_temp:
        raw = {}
        for num in car_nums:
            col = f"Car {num} - Indoor Average Temperature"
            if col not in df.columns:
                raw[num] = (0, np.nan)
                continue
            s = df[col]
            zero_count = int((s == 0).sum())
            valid = s[s > 0]
            mean_valid = valid.mean() if len(valid) > 0 else np.nan
            raw[num] = (zero_count, mean_valid)

        valid_means = {k: v[1] for k, v in raw.items() if not pd.isna(v[1])}
        peer_avg = np.mean(list(valid_means.values())) if valid_means else np.nan

        for num in car_nums:
            zc, mv = raw[num]
            dev = (mv - peer_avg) if not pd.isna(mv) else -999
            scores[num] = zc * 1000 + dev

    elif has_compressor:
        run_frac = {}
        for num in car_nums:
            vals = []
            for comp in ["Compressor 1 Running", "Compressor 2 Running"]:
                col = f"Car {num} - {comp}"
                if col in df.columns and df[col].notna().any():
                    vals.append(df[col].mean())
            run_frac[num] = np.mean(vals) if vals else np.nan

        valid_frac = {k: v for k, v in run_frac.items() if not pd.isna(v)}
        peer_avg = np.mean(list(valid_frac.values())) if valid_frac else np.nan

        for num in car_nums:
            v = run_frac[num]
            scores[num] = (peer_avg - v) if not pd.isna(v) else -999
    else:
        for num in car_nums:
            scores[num] = 0

    return scores


def _load_dataframe(data: Union[str, bytes, io.BytesIO, pd.DataFrame]) -> pd.DataFrame:
    """Accepts a file path, raw bytes, a BytesIO buffer, or an already-
    loaded DataFrame — whichever is most convenient for the caller."""
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, (bytes, bytearray)):
        return pd.read_excel(io.BytesIO(data))
    if isinstance(data, io.BytesIO):
        return pd.read_excel(data)
    if isinstance(data, str):
        return pd.read_excel(data)
    raise TypeError(f"Unsupported input type for ACV data: {type(data)}")


def rank_cars(data: Union[str, bytes, io.BytesIO, pd.DataFrame]) -> list:
    """
    Returns the list of car identifiers (as strings, e.g. '04'), ordered
    from most- to least-likely to be the faulty car.
    """
    df = _load_dataframe(data)
    car_nums = _get_car_numbers(df.columns)
    scores = _score_cars(df, car_nums)
    return sorted(car_nums, key=lambda n: scores[n], reverse=True)


def predict(
    data: Union[str, bytes, io.BytesIO, pd.DataFrame],
    file_id: str = None,
    output_format: str = "dataframe",
):
    """
    Main entry point.

    Parameters
    ----------
    data : file path, raw bytes, BytesIO, or a pandas DataFrame
    file_id : the source filename to report in output (only used for
        "dataframe" and "dict" output formats — ignored for "list")
    output_format : one of "dataframe", "dict", "list"
        - "dataframe": pandas DataFrame with columns file_id, ranked_cars
          (matches the hackathon submission schema exactly)
        - "dict": {"file_id": ..., "ranked_cars": "04|07|05|..."}
        - "list": just the ranked list of car ids, e.g. ["04","07",...]

    Returns
    -------
    pandas.DataFrame, dict, or list depending on output_format.
    """
    ranked = rank_cars(data)
    ranked_str = "|".join(ranked)

    if output_format == "list":
        return ranked
    if output_format == "dict":
        return {"file_id": file_id, "ranked_cars": ranked_str}
    if output_format == "dataframe":
        return pd.DataFrame({"file_id": [file_id], "ranked_cars": [ranked_str]})

    raise ValueError(f"Unknown output_format: {output_format}")
