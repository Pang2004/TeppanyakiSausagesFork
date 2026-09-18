# Local spectral investigation — 19 September 2026

The best completed decision-tuning procedure improves six-repeat mean fold
macro F1 from **0.7095 to 0.7609**, and Side I F1 from **0.4097 to 0.5124**.
One validation repeat reaches **0.6207 Side I / 0.7977 macro**, but 0.60 Side I
is not stable across repeats. These are exploratory results on the existing
labelled data. Production and submission predictions remain unchanged.

## What changed

The production features summarize each side across the train. The new research
features preserve wavelength-band energy for each of eight cars and compare
matched axle positions across the sides. Each recording supplies 702 local
energy features, plus 192 spectral-shape features tested separately.

Local energy features cover the existing six wavelength intervals, both
vibration and shock, absolute power and power fraction, and total power.
Per-car side power uses log1p compression. Paired-side contrasts retain per-car
means, the median, directional agreement and a signed third moment. No label or
dataset-wide statistic enters feature extraction. Wavelength conversion uses
the existing tachometer speed estimate; zero-speed bands are zero.

Experiments tested the original features, original plus local energy, original
plus spectral shape, and local features alone. Within each family, the search
used logistic C values 0.01, 0.1, 1 and 10, and side-mirrored training at C=0.1
and C=1. Original plus local energy with mirrored training was promising;
spectral shape and local-only inputs were weaker.

The follow-up decision search compares the four original logistic models
against original plus local energy with mirrored training, four C values and
Side I score multipliers 0.5, 1, 2 and 4. All settings are selected by mean
inner-fold macro F1. A multiplier adjusts the classification decision, not the
calibration of a fault probability.

## Validation results

Every evaluation uses five outer StratifiedGroupKFold splits and three grouped
inner splits (inner seed 142). Exact byte duplicates remain together. All
scaling, training augmentation and decision selection occur after splitting.

| Procedure | Seeds | Macro F1 | Side I F1 | Side II F1 |
| --- | --- | ---: | ---: | ---: |
| Original procedure | 42–44 | 0.7212 | 0.4359 | 0.7694 |
| Nested search over local feature families | 42–44 | 0.7620 | 0.4881 | 0.8285 |
| Local energy + decision search | 42–44 | **0.7857** | **0.5665** | 0.8178 |
| Original procedure | 45–47 | 0.6977 | 0.3834 | — |
| Nested search over local feature families | 45–47 | 0.7383 | 0.4816 | 0.7668 |
| Local energy + decision search | 45–47 | 0.7362 | 0.4583 | 0.7834 |
| Original procedure, all six repeats | 42–47 | 0.7095 | 0.4097 | 0.7623 |
| Local energy + decision search, all six | 42–47 | **0.7609** | **0.5124** | **0.8006** |
| Global/local blend, all six | 42–47 | 0.7444 | 0.4832 | 0.7842 |

These table values average the F1 scores of individual outer folds. A pooled
out-of-fold score for a whole repeat is a different aggregation:

| Seed | Decision search: pooled macro F1 | Pooled Side I F1 |
| --- | ---: | ---: |
| 42 | 0.7977 | **0.6207** |
| 43 | 0.7755 | 0.5161 |
| 44 | 0.7836 | 0.5625 |
| 45 | 0.7246 | 0.4516 |
| 46 | 0.7674 | 0.5161 |
| 47 | 0.7144 | 0.4000 |

Additional seeds test split sensitivity on the same recordings, not an
independent dataset. Feature families, decision tuning and subsequent blending
were investigated after earlier error inspection. Nested tuning does not remove
this broader research-selection bias. Repeated predictions are correlated:
there are still only 14 independent Side I recordings, not 84.

## Error changes

Across six repeats, the original procedure has 39 Side I true positives,
62 false positives and 45 false negatives. The local decision procedure has
47 true positives, 53 false positives and 37 false negatives. Its pooled Side I
precision is 0.470 and recall 0.560. Both missed detections and false alarms
still limit F1; increasing recall alone is insufficient.

| Side I recording | Original correct / 6 | Local decision correct / 6 |
| --- | ---: | ---: |
| Train100 | 1 | 5 |
| Train180 | 2 | 6 |
| Train170 | 1 | 3 |
| Train137 | 5 | 6 |
| Train62 | 5 | 6 |
| Train202 | 5 | 0 |
| Train106 | 0 | 0 |
| Train121 | 0 | 0 |
| Train150 | 0 | 0 |
| Train185 | 0 | 0 |

Train202 is an important regression. It motivates testing a blend of global
and local models, rather than assuming that the new features dominate every
recording. The completed blend search averaged the previous log-compressed,
mirrored global model (C=0.1) with a mirrored local-energy model (C=0.1 or 1),
using local weights 0.25, 0.5 or 0.75 and Side I multipliers 0.5, 1 or 2.
Inner-selected blending scored 0.7444 macro / 0.4832 Side I and did not recover
Train202. It is weaker than the local decision procedure and is not recommended
as a replacement. The remaining consistent misses should be reviewed against
independent signal/label evidence; validation errors alone do not justify
changing their labels.

## Reproduction

From the repository root, use
`PYTHONPATH='app:Optional_Items/Rail Corrugation/code'` and single-threaded BLAS
for each command:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH='app:Optional_Items/Rail Corrugation/code' \
  .venv/bin/python -m rail_dev.local_study

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH='app:Optional_Items/Rail Corrugation/code' \
  .venv/bin/python -m rail_dev.local_study --seeds 45 46 47 \
  --report-dir 'Optional_Items/Rail Corrugation/code/outputs/local_study/sensitivity'

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH='app:Optional_Items/Rail Corrugation/code' \
  .venv/bin/python -m rail_dev.local_study --tune-decisions --seeds 42 43 44 45 46 47 \
  --report-dir 'Optional_Items/Rail Corrugation/code/outputs/local_study/decisions'

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH='app:Optional_Items/Rail Corrugation/code' \
  .venv/bin/python -m rail_dev.local_study --blend --seeds 42 43 44 45 46 47 \
  --report-dir 'Optional_Items/Rail Corrugation/code/outputs/local_study/blend'
```

Each directory contains `cv_results.json`, `oof_predictions.csv` and
`file_audit.csv`. Reports retain fold memberships, inner scores, selections,
per-class metrics and confusion matrices. The local feature cache manifest
includes source recording hashes, feature configuration and extractor-code hash.

All 17 Rail tests passed, including checks that synthetic wavelength energy
lands in the expected band and car, that side mirroring matches raw sensor
swapping, and that recording groups remain separate during nested selection.
