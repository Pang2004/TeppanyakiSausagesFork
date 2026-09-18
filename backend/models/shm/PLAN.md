# SHM Cumulative Fatigue Damage Plan

## Objective and Data Interpretation

Predict one positive cumulative fatigue-damage value for every dynamic-stress file. This is regression, scored as `max(0, 1 - MAPE)`. Each input is a headerless, single-column stress history: all 64 training and 16 test files contain 581,120 samples. Filenames are randomly assigned identifiers and are never model features.

The reference targets were produced from rainflow cycle counting and Miner's linear damage rule. Under the S–N relationship `N = C / amplitude^m`, cumulative damage is proportional to:

```text
sum(cycle_count * (cycle_range / 2) ** m)
```

The S–N constants are not supplied, so exponent and scale must be calibrated using only labelled training files. Exploratory leave-one-file-out validation selected `m = 5`; its calibrated Miner proxy achieved approximately `2.6%` MAPE, substantially outperforming generic regression.

## Input and Feature Strategy

Load every CSV with no header so the first stress value is preserved. Require one finite numeric column and at least three samples, but permit different recording lengths because Miner damage accumulates with cycle count.

Use the local `rainflow==3.2.0` package, which implements ASTM E1049-style counting. It is an installed Python dependency, not an external API: inference makes no network requests and uploads no sensor data.

Extract cycle ranges, means, half/full counts, maximum and weighted range quantiles, equivalent amplitude, and Miner proxies for exponents 3–7. Add compact residual features: distribution/tail statistics, first-difference statistics, zero-crossing rate, and summaries of 128 block ranges. Units and sampling frequency are undocumented, so do not attach invented physical units or Hz labels.

## Train, Validate, and Predict

Use nested leave-one-file-out validation. Select the S–N exponent inside each outer fold, calibrate the multiplicative scale to minimize MAPE, and compare:

1. MAPE-optimal constant baseline.
2. Log damage versus global stress range.
3. Calibrated rainflow/Miner model.
4. Miner model plus a scaled Ridge correction on log residuals.

Adopt the residual model only when it improves outer-fold MAPE by at least `0.002` and does not worsen 95th-percentile relative error; otherwise keep the interpretable Miner model. Retrain on all 64 files and store the exponent, scale, optional residual estimator, feature schema, validation metrics, and dependency versions.

The implemented validation selected the hybrid model. The calibrated Miner model achieved `2.54%` leave-one-file-out MAPE (`0.9746` score); Ridge residual correction improved this to `2.27%` MAPE (`0.9773` score) with `6.71%` 95th-percentile relative error. Every outer fold and the final fit selected `m = 5`; the final Ridge alpha is `1.0`.

```bash
python -m backend.models.shm.train \
  --data-dir PS3/02_Datasets/SHM \
  --model-out backend/artifacts/shm_pipeline.joblib

python -m backend.models.shm.predict \
  --input PS3/02_Datasets/SHM/Test \
  --model backend/artifacts/shm_pipeline.joblib \
  --output outputs/shm_predictions.csv
```

The callable adapter returns predicted damage, counted cycles, equivalent amplitude, maximum cycle range, and validation uncertainty. The official CSV contains only `file_id,prediction`, naturally ordered and preserving `.csv` extensions.

## Acceptance Checks

- Confirm that official files load all 581,120 samples and contain finite values.
- Match a published rainflow example and verify fifth-power scaling under stress rescaling.
- Keep complete files inside validation folds; never split samples from one file across folds.
- Beat constant and range-only baselines on leave-one-file-out MAPE.
- Produce 16 unique, positive, finite Test predictions from `test01.csv` through `test16.csv`.
- Save caches and validation reports only below ignored `outputs/shm/` paths.
