# Rail Corrugation Methodology

## Current model: local spectral pipeline v3

The classifier uses the original 855 statistical/wavelength features plus 702 local wavelength-energy features. These retain per-car band power and matched-axle side contrasts. The shared extractor also computes 192 spectral-shape features retained in the ordered input schema; the adopted classifier excludes them. Runtime extraction and the fitted estimator live entirely in `app/backend/models/rail/`, without development-package dependencies.

Training mirrors the two sides only inside training folds, including swapping Side I/Side II labels and reversing signed contrasts. A standardized, class-balanced logistic model classifies each recording. Grouped inner validation selected **C=0.01 and a Side I decision-score multiplier of 4** for the full-data fit. Returned class scores remain uncalibrated model scores; the decision multiplier means their raw maximum need not equal the final predicted class.

The feature family was adopted after exploratory comparisons. Within it, three grouped inner folds (seed 142) select C from `[0.01, 0.1, 1, 10]` and the Side I multiplier from `[0.5, 1, 2, 4]`, using mean macro F1. The full-data inner score is 0.7110. The old feature family's best inner score is slightly higher (0.7165, C=10), while repeated outer validation favours the adopted family. That discrepancy is retained in the report; a single small inner split is not a guarantee of which model will generalise better.

There are 272 labelled files: 234 Normal, 14 Side I and 24 Side II. SHA-256 grouping identifies 270 unique recordings; exact duplicate Normal pairs stay in the same fold. Scaling, augmentation and decision selection occur only within training data. Filenames are identifiers, not classifier inputs.

## Validation and integration checks

Five grouped outer folds are repeated using seeds 42–47. The same recordings are reused across repeats; 1,632 predictions do not represent 1,632 independent recordings.

| Procedure, same six-repeat protocol | Macro F1 | Side I F1 | Side II F1 |
| --- | ---: | ---: | ---: |
| Old original/wavelength procedure | 0.7095 | 0.4097 | 0.7623 |
| Adopted local-energy + decision tuning | **0.7609** | **0.5124** | **0.8006** |

These are means of fold scores. Seed 42's pooled out-of-fold result is 0.7977 macro / 0.6207 Side I, but Side I does not consistently reach 0.60 across repeats. The feature family was selected after earlier experiments and error review: the table is an exploratory validation estimate, not an independent hidden-test result. Labels for the 68 supplied test files are unavailable.

Production integration refitted the selected estimator on every outer training fold and **reproduced all 1,632 held-out predictions exactly**. The saved research inner searches were retained rather than rerun in replay mode. Original raw-file hashes were checked, local spectra were freshly extracted for all 272 files, and combined features were independently re-extracted for representative normal and difficult Side I recordings before fitting. The normal training command reruns nested selection from scratch; replay mode checks the existing completed evaluation.

See [Local spectral investigation](LOCAL_SPECTRA_STUDY.md) for the full search, split-sensitivity results, remaining errors and failed blending experiment. Train202 is a known regression; the consistently missed examples and false alarms remain important limitations.

## Runtime, reproduction and rollback

The active artifact is `app/backend/artifacts/rail_pipeline.joblib`, version `rail-pipeline-v3`. `RailPredictor` checks base, wavelength and local configurations before inference, and still supports older v1/v2 artifacts. The callable API and official `file_id,prediction` CSV schema are unchanged.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/model.py rail train \
  --data-dir PS3/02_Datasets/Rail_Corrugation

# Reproduce the integration using the completed research inner-search report:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/model.py rail train \
  --data-dir PS3/02_Datasets/Rail_Corrugation \
  --replay-report 'Optional_Items/Rail Corrugation/code/outputs/local_study/decisions/cv_results.json'

.venv/bin/python scripts/model.py rail predict \
  --input PS3/02_Datasets/Rail_Corrugation/Test \
  --output 'Optional_Items/Rail Corrugation/code/outputs/rail_predictions.csv'
```

The previous active model is preserved at `app/backend/artifacts/rail_pipeline_baseline_v2.joblib`; its reports, predictions and methodology are under `code/outputs/baseline_v2/`. Passing that artifact with `--model` reproduces previous CLI inference; setting `RAIL_MODEL_PATH` to its path and restarting the API restores it in the app. Training scripts do not automatically create rollback copies, so use `--model-out` when experimenting.

The original feature cache uses source hashes and configuration checks. The local feature cache additionally records its extractor-code hash. The legacy v2 trainer remains available explicitly as `rail_dev.train`; `scripts/model.py rail train` now invokes `rail_dev.train_local`.
