# SHM Cumulative-Fatigue Methodology

## Task and Physical Model

The Structural Health Monitoring pipeline predicts one positive cumulative fatigue-damage value for each dynamic-stress history. The official score is `max(0, 1 - MAPE)`, so model selection directly minimizes percentage error rather than squared error.

The dataset contains 64 labelled training files and 16 held-out files. Every official file is a headerless, single-column CSV with 581,120 stress samples. Each complete history is one independent example; individual samples are never split between training and validation. Filenames are randomized identifiers and are not features.

The reference description states that targets are based on rainflow cycle counting and Miner's linear damage rule. For an S–N relation `N = C / amplitude^m`, cumulative damage is proportional to:

```text
sum(cycle_count * (cycle_range / 2) ** m)
```

The S–N exponent and multiplicative constant are not supplied. They are therefore calibrated from labels rather than assumed from an undocumented material specification.

## Rainflow and Signal Features

The loader preserves the first stress sample by explicitly reading the CSV without a header and rejects multi-column, non-numeric, non-finite, or very short histories. Rainflow cycles are extracted locally with `rainflow==3.2.0`, which implements ASTM E1049-style counting. Prediction makes no network request and does not upload stress data.

For candidate exponents `m = 3, 4, 5, 6, 7`, the pipeline computes a Miner proxy and equivalent stress amplitude. Additional cycle features include total cycle count, maximum and weighted mean range, range standard deviation, mean cycle stress, and weighted range quantiles through the 99.9th percentile.

Compact residual features describe information not completely captured by the Miner sum: signal mean, standard deviation, RMS, range, skewness, kurtosis, tail quantiles, centered-amplitude quantiles, first-difference statistics, and zero-crossing rate. The history is also divided into 128 blocks whose range distribution captures non-stationary high-load periods. These features are used only to correct the physical prediction; the main magnitude still comes from accumulated rainflow damage.

## Nested Validation and Final Model

Outer leave-one-file-out validation estimates generalization at the only safe grouping level: an entire stress history. Within each outer training set, five-fold inner validation selects the S–N exponent and Ridge regularization strength. The Miner scale is a weighted median of target-to-proxy ratios, which is the robust multiplicative calibration associated with percentage error.

Four approaches are evaluated:

1. A MAPE-optimal constant prediction.
2. A power relationship between global stress range and damage.
3. The calibrated rainflow/Miner model.
4. The Miner estimate multiplied by a Ridge prediction of its log residual.

The correction model is accepted only if it improves outer-fold MAPE by at least `0.002` and does not worsen 95th-percentile percentage error. This gate prevents a small average improvement from hiding less stable worst-case behaviour. Every outer fold selected exponent `m=5`. Ridge alpha `1.0` was selected in 62 folds and `10.0` in two; retraining on all 64 files selected `m=5`, alpha `1.0`, and scale `1.3602318550841183e-09`.

## Validation-Backed Estimate

| Approach | MAPE | Official-score estimate | 95th-percentile error |
| --- | ---: | ---: | ---: |
| Weighted constant | `58.31%` | `0.4169` | — |
| Global-range power model | `27.65%` | `0.7235` | — |
| Calibrated Miner model | `2.54%` | `0.9746` | `8.95%` |
| Miner + Ridge residual | **`2.27%`** | **`0.9773`** | **`6.71%`** |

The selected leave-one-file-out result is the best available expected score because every prediction is made for a file excluded from fitting. It is still based on only 64 histories from the supplied distribution. Different structural details, stress units, recording lengths, or load regimes could shift calibration. The model permits different lengths because Miner damage accumulates with cycle count, but the measured estimate comes from files of equal length.

The 16 generated held-out predictions range from `0.02794` to `0.82158`, with median `0.06471`. These values are finite and positive; without held-out reference damage they are not accuracy measurements.

## Reproduction and Interface

```bash
python -m backend.models.shm.train \
  --data-dir PS3/02_Datasets/SHM \
  --model-out backend/artifacts/shm_pipeline.joblib

python -m backend.models.shm.predict \
  --input PS3/02_Datasets/SHM/Test \
  --model backend/artifacts/shm_pipeline.joblib \
  --output outputs/shm_predictions.csv \
  --diagnostics-output outputs/shm/diagnostics.json
```

The callable adapter returns predicted damage, counted cycles, equivalent amplitude, maximum cycle range, and the validation-derived uncertainty indicator. The official CSV contains only `file_id,prediction`. Feature tables, validation reports, diagnostics, predictions, and the trained artifact are tracked so teammates reproduce the same final deliverables.
