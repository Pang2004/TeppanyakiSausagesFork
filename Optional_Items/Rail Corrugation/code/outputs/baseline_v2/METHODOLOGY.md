# Rail Corrugation Methodology

Further research is recorded in [Local spectral investigation](LOCAL_SPECTRA_STUDY.md).
The local-energy and decision-tuning candidate reached 0.7609 macro F1 and
0.5124 Side I F1 across six validation repeats, compared with 0.7095 / 0.4097
for the original procedure on those same splits. These exploratory results
have not replaced the production model described below.

## Current production model

The production pipeline classifies each one-second recording as Normal, Side I or Side II. It uses the original 684 statistical/frequency features plus 171 speed-adjusted spectral features (855 total), standardized inside a class-balanced logistic regression pipeline. Grouped inner validation on all labelled training data selected C=10; the saved estimator is fitted on all 272 labelled files.

The official metric is macro F1. Labels comprise 234 Normal, 14 Side I and 24 Side II files. The 68 test inputs have no published reference labels, so their prediction outputs are not accuracy measurements.

## Physical interpretation and features

Each recording has 10,000 samples at 10 kHz and 129 columns: a wheel tachometer followed by vibration and shock for 64 axle boxes. Headers determine sensor identity. Positions 1, 3, 5 and 7 belong to Side I; positions 2, 4, 6 and 8 belong to Side II. Filenames identify files but never enter the model.

The original features summarize time-domain statistics and Welch spectral power for each side and signal type, plus differences and ratios between sides and four tachometer summaries. They retain absolute signal magnitude and fixed-frequency information.

The additional features estimate mean speed from tachometer transitions. The documented wheel diameter is 0.85 m and the toothed wheel has 90 teeth. Counting rising and falling edges gives 180 edges per revolution:

`speed_mps = edge_count / observed_duration_seconds / 180 * pi * 0.85`

A wavelength interval maps to frequencies `speed / upper_wavelength` through `speed / lower_wavelength`. Using a 2,048-sample Welch window, normalized power is integrated between interpolated band boundaries for fixed wavelength edges of 0.01, 0.02, 0.04, 0.08, 0.16, 0.32 and 0.64 metres. Per-channel band fractions and dominant wavenumber are summarized by mean, median, standard deviation and maximum for each side/signal type, with corresponding side contrasts. Mean speed, speed availability and variability across ten blocks complete the 171 features.

This uses mean recording speed, not instantaneous order tracking. Zero-transition recordings receive zero wavelength/wavenumber features and a speed-unavailable indicator. A stationary recording and a failed tachometer cannot be distinguished from this signal alone. Original features remain available in either case.

## Validation and selection

SHA-256 hashes identify 270 unique byte contents among the 272 files. Train107/Train115 and Train165/Train187 are duplicate pairs, all Normal. Each pair stays together in every training/validation split. Duplicates retain one vote per file; conflicting labels for identical recordings are rejected. Byte hashes do not identify near duplicates or shared acquisition sessions.

Validation uses five-fold StratifiedGroupKFold with seeds 42, 43 and 44. Within each outer training subset, three grouped folds (seed 142) select C from [0.01, 0.1, 1, 10] by mean macro F1. Ties prefer the first C. Scaling is fitted only on training subsets. The selected model then predicts the untouched outer fold. The feature family is fixed during this procedure. The always-Normal model is a baseline only.

After outer evaluation, the same inner procedure selects C on the complete labelled dataset, followed by a full-data fit. The final C=10 is not chosen from outer scores or the test inputs. Different outer training subsets selected different C values: 9 folds selected C=0.01, 4 folds selected C=0.1, 1 folds selected C=1.0, 1 folds selected C=10.0. This variation reflects the small training sample; no single C is claimed to be universally optimal.

## Measured results and limitations

| Model / evaluation | Mean fold macro F1 | Side I mean fold F1 | Side II mean fold F1 |
| --- | ---: | ---: | ---: |
| Previous production model on grouped folds | 0.6805 | 0.3255 | 0.7617 |
| Original features with inner C tuning | 0.6860 | 0.3525 | 0.7529 |
| Adopted original + wavelength features, inner C tuning | **0.7212** | **0.4359** | **0.7694** |
| Earlier nested selection among six feature families | 0.6859 | 0.3479 | 0.7539 |

The adopted model's Normal mean fold F1 is 0.9583. Its macro F1 fold standard deviation is 0.0650; this describes fold variability, not a confidence interval. All 816 outer predictions (272 files × three repeats) reproduced the corresponding experiment exactly after production integration and fresh raw feature extraction.

The 0.7212 result excludes each validation file and all its exact duplicates from fitting and inner tuning. However, the wavelength feature family was adopted after comparing multiple experiments on these outer folds. Therefore 0.7212 is an exploratory candidate estimate, not an unbiased score for the entire search procedure or confirmed hidden-test performance. The 0.6859 figure describes selecting feature families solely within inner validation and must remain visible alongside the chosen candidate. Repeated predictions are correlated, not additional independent examples. Side I still has only 14 independent labelled fault recordings.

The original ungrouped report of 0.6910 macro F1 is superseded: identical recordings crossed train/validation in eight of its fifteen folds. The earlier 0.6454 grouped spot check used different group identifiers and fold assignments; it is not the production validation protocol.

## Artifacts, interface and reproduction

`app/backend/artifacts/rail_pipeline.joblib` is the current `rail-pipeline-v2` artifact. It stores feature names in exact order, the base and wavelength configurations, selected estimator and validation report. The callable `RailPredictor`/`predict_file` interface and output schema are unchanged. Training and inference share `wavelength.extract_production_features`; inference does not import experiment runners. Unknown or missing wavelength configurations are rejected.

`Optional_Items/Rail Corrugation/code/outputs/train_features_wavelength.csv` is the current training cache. Its adjacent `.manifest.json` records source hashes and both feature configurations. A mismatch or missing manifest triggers extraction. The original feature cache is preserved under `Optional_Items/Rail Corrugation/code/outputs/baseline_v1/train_features.csv` for rollback reference.

`Optional_Items/Rail Corrugation/code/outputs/cv_results.json` reports fixed-C candidates and nested C selection, inner scores, duplicate groups, per-class metrics, confusion matrices and exact fold memberships. `Optional_Items/Rail Corrugation/code/outputs/oof_predictions.csv` stores one row per file/repeat/evaluation; filter on `evaluation == nested_selection` for the adopted procedure. Repeated support counts must not be interpreted as unique files.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/model.py rail train \
  --data-dir PS3/02_Datasets/Rail_Corrugation \
  --model-out app/backend/artifacts/rail_pipeline.joblib

python scripts/model.py rail predict \
  --input PS3/02_Datasets/Rail_Corrugation/Test \
  --model app/backend/artifacts/rail_pipeline.joblib \
  --output "Optional_Items/Rail Corrugation/code/outputs/rail_predictions.csv" \
  --diagnostics-output "Optional_Items/Rail Corrugation/code/outputs/diagnostics.json"
```

The regenerated test output contains 58 Normal, 5 Side I and 5 Side II predictions across all 68 test files. One label differs from the previous model; unknown test labels prevent judging that change as correct or incorrect.

The official CSV contains only `file_id,prediction`. The adapter also returns class scores, side-energy summaries and dominant frequencies. Class-balanced logistic scores are not independently calibrated fault probabilities.

The previous artifact is preserved as `app/backend/artifacts/rail_pipeline_baseline_v1.joblib`, with its validation reports and predictions in `Optional_Items/Rail Corrugation/code/outputs/baseline_v1/`. The predictor supports this legacy artifact directly; passing it with `--model` reproduces baseline inference without changing production code.

## Experiment comparison

These historical results use mean fold F1 on the same grouped outer splits. The comparison is retained here after removal of the temporary experiment runners, caches and output folders.

| Experiment | Macro F1 | Side I F1 | Side II F1 |
| --- | ---: | ---: | ---: |
| Previous production model | 0.6805 | 0.3255 | 0.7617 |
| Tuned original features | 0.6860 | 0.3525 | 0.7529 |
| Shared side detector | 0.6779 | 0.3568 | 0.7249 |
| Full spatial features only | 0.6204 | 0.1756 | 0.7366 |
| Original + full spatial | 0.6426 | 0.2379 | 0.7395 |
| Compact spatial only | 0.6643 | 0.3387 | 0.7034 |
| Original + compact spatial | 0.7057 | 0.3862 | 0.7733 |
| Speed-adjusted spectra only | 0.5319 | 0.2582 | 0.4435 |
| Adopted original + speed-adjusted spectra | 0.7212 | 0.4359 | 0.7694 |
| Original + compact spatial + speed-adjusted spectra | 0.7154 | 0.3960 | 0.7918 |
| Nested selection across remaining feature sets | 0.6859 | 0.3479 | 0.7539 |

Shared-side training used a common binary detector and an inner-selected decision threshold. Full spatial inputs retained sensor, matched-axle, car and bogie statistics; compact inputs retained 196 summaries of paired-side contrasts and regional maximum RMS. All learned preprocessing and regularization selection occurred inside training folds. These comparisons informed the choice of the production feature family; they do not constitute independent held-out test results.
