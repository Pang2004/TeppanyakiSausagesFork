# SHM Cumulative-Fatigue Methodology

## Task and Physical Model

The Structural Health Monitoring pipeline predicts one positive cumulative fatigue-damage value for each dynamic-stress history. The official score is `max(0, 1 - MAPE)`, so model selection directly minimizes percentage error rather than squared error.

The dataset contains 64 labelled training files and 16 held-out files. Every official file is a headerless, single-column CSV with 581,120 stress samples. Each complete history is one validation unit; individual samples are never split between training and validation. Statistical independence between files is not established: the reference describes measurements from two lines and two load conditions, but supplies no per-file group identifiers or recording chronology. All recordings are from healthy operating conditions; the target is accumulated fatigue damage, not observed structural failure. Filenames are randomized identifiers and are not features.

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

Outer leave-one-file-out validation holds out one complete stress history at a time. Within the remaining 63 files, shuffled five-fold validation selects the S–N exponent from `3,4,5,6,7` using physics-model MAPE. Using that exponent, it selects Ridge alpha from `0.1,1,10,100`. Every inner fit estimates the Miner scale and, for Ridge, fits standardization and regression using only that inner training subset. The scale is the weighted median of target-to-proxy ratios, with weights proxy/target, which minimizes training MAPE for multiplicative calibration.

The existing correction gate is now applied **inside each outer training fold**: inner residual MAPE must improve by at least `0.002` (0.2 percentage points), without worsening the inner 95th-percentile absolute percentage error. The selected family is then fitted on all 63 outer training files and predicts the untouched 64th file. The outer label enters only the final scoring. Inner scores used for tuning are not reported as generalization estimates.

Previously, the gate compared the two families' outer errors and reported the winning family's outer score. That evaluated hyperparameter tuning but did not independently evaluate family selection. The corrected primary result evaluates the entire selection procedure. Its fixed seed remains 42; seeds 43 and 44 are sensitivity checks and are not selected by their outcomes.

The primary run selects exponent `m=5` in every outer fold. Alpha `1.0` is selected in 62 folds and `10.0` in two. The family gate selects Miner + Ridge in 63 folds and Miner alone in one. Applied to all 64 training files, the inner selection chooses Miner + Ridge, `m=5`, alpha `1.0`. The retained scale is `1.3602318550841183e-09`. The production artifact keeps its exact existing fitted parameters and refreshes validation metadata; all 16 test-file predictions agree with the existing submission CSV to numerical precision.

## Validation Results

| Approach | MAPE | Official score | 95th-percentile absolute percentage error |
| --- | ---: | ---: | ---: |
| Weighted constant | 58.3121% | 0.416879 | — |
| Global-range power model | 27.6468% | 0.723532 | — |
| Calibrated Miner family, inner exponent selection | 2.5444% | 0.974556 | 8.9458% |
| Miner + Ridge family, inner exponent/alpha selection | 2.2694% | 0.977306 | 6.7090% |
| **Complete nested family-selection procedure, seed 42** | **2.2978%** | **0.977022** | **6.7090%** |

The fixed-family rows are useful comparisons; choosing a family from those outer results and quoting its winning score would reuse the evaluation data. The primary selected-procedure result is therefore 0.977022, replacing the earlier headline 0.977306. This change corrects the estimate; it is not a measured improvement to the deployed predictor. Maximum absolute percentage error for the primary selected procedure is 8.7531%.

### Inner-split sensitivity

| Inner split seed | Nested MAPE | Nested official score | Outer folds selecting Ridge | Full-data family |
| --- | ---: | ---: | ---: | --- |
| 42 (primary) | 2.2978% | 0.977022 | 63 / 64 | Miner + Ridge |
| 43 | 2.5858% | 0.974142 | 42 / 64 | Miner + Ridge |
| 44 | 2.3514% | 0.976486 | 60 / 64 | Miner + Ridge |

All three full-data selections retain the existing family. The outer gate is sensitive to the small sample and the modest residual improvement: seed 43's selected procedure is slightly worse than the primary fixed-physics result. These runs reuse the same 64 histories and are not 192 independent test examples or a confidence interval. They support retaining the model while avoiding claims that the small Ridge gain is robust across all splits.

### Data and cache checks

All 64 raw training files have distinct SHA-256 hashes. Fresh feature extraction from every raw history agrees with the stored feature table within `1e-12` relative/absolute numerical tolerance. Cache reuse now verifies raw hashes, feature configuration, extractor source hash, package versions and the feature CSV's own hash. A changed input/configuration or edited cache triggers extraction. Duplicate histories stop training with a request for grouped validation rather than allowing duplicates across file-level folds. Distinct hashes do not rule out related or overlapping source recordings.

`outputs/shm/cv_results.json` contains the primary inner candidate scores, outer memberships, per-file predictions for both families and the selection procedure, final selection, and input provenance. `outputs/shm/validation_sensitivity.json` records the two additional seeds. `outputs/shm/train_features.manifest.json` records cache provenance. The validation module and tests support the final model; no unused experiment runner or experiment test directory is retained.

### Error indicator and remaining limits

The API field `estimated_percentage_error` is retained for compatibility. Its value, approximately **0.06709**, is the **historical 95th percentile of absolute percentage errors across the 64 outer predictions**. It is the same number for every uploaded file. Diagnostics explicitly identify it with `error_indicator_kind`; it is not a file-specific error estimate, a ±6.71% prediction interval, or a guarantee of 95% coverage. The percentile is based on a small sample and varies with the validation split.

These results are retrospective checks on previously used labelled histories, not an untouched external benchmark. File-level validation cannot establish transfer to an unseen line, vehicle, load group or damaged structure without the missing group labels and new recordings. Different structural details, stress units, recording lengths or load regimes could shift calibration. The physical sum supports varying history lengths, but the residual model and measured scores have only been validated on equal-length files.

The 16 unlabelled test predictions range from `0.02794` to `0.82158`, with median `0.06471`. These values are finite and positive; without held-out reference damage they are not accuracy measurements.

## Reproduction and Interface

```bash
python -m backend.models.shm.train \
  --data-dir PS3/02_Datasets/SHM \
  --model-out backend/artifacts/shm_pipeline.joblib

python -m backend.models.shm.validation \
  --data-dir PS3/02_Datasets/SHM \
  --inner-seeds 43 44

python -m backend.models.shm.predict \
  --input PS3/02_Datasets/SHM/Test \
  --model backend/artifacts/shm_pipeline.joblib \
  --output outputs/shm_predictions.csv \
  --diagnostics-output outputs/shm/diagnostics.json
```

The callable adapter returns predicted damage, counted cycles, equivalent amplitude, maximum cycle range, and the historical validation-error percentile. The official CSV contains only `file_id,prediction`. Feature tables, validation reports, diagnostics, predictions, and the trained artifact are tracked so teammates reproduce the same final deliverables.
