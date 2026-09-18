# acv_model

Standalone ACV (refrigerant leak) fault-ranking module. No app-framework
dependency — pure Python + pandas/numpy. Drop this into any app-maker,
wrap it in a CLI, or expose it as a REST endpoint.

## Install

```
pip install -e .
```

or just copy the `acv_model/` folder into your project — it has no
dependency on the rest of this repo.

## Basic usage

```python
from acv_model import predict

# from a file path
df = predict("acv_test_case.xlsx", file_id="acv_test_case.xlsx")

# from raw bytes (e.g. an uploaded file's .read())
df = predict(file_bytes, file_id="acv_test_case.xlsx")

# from an already-loaded DataFrame
df = predict(some_dataframe, file_id="acv_test_case.xlsx")

# different output shapes
predict(path, file_id="x.xlsx", output_format="dict")   # {"file_id": ..., "ranked_cars": "04|07|..."}
predict(path, file_id="x.xlsx", output_format="list")   # ["04", "07", "05", ...]
```

## Changing the interface further

If your app-maker expects something different from any of the above
(e.g. a specific JSON shape, a class-based interface, an async
function, a different parameter order), that's a thin wrapper around
`rank_cars()` — the core function that does the actual work and always
returns a plain Python list of car ids ranked most→least likely. Write
whatever adapter function your app-maker needs and have it call
`rank_cars()` internally; the ranking logic itself never has to change.

## CLI

```
python predict.py --input acv_test_case.xlsx --output acv_predictions.csv
```

## Status / caveats

Heuristic (not a trained ML model). Validated against the 6 provided
training cases (perfect rank-decay score) and stability-tested via
bootstrap resampling — but one training case had a very thin margin
between the top two candidates, and the underlying signals (sensor
dropout, temperature deviation from peers, compressor duty-cycle
suppression) haven't been confirmed against domain literature as true
leak indicators vs. possible confounds (e.g. car position / sun
exposure). Treat as a strong baseline, not a guaranteed-correct model.
