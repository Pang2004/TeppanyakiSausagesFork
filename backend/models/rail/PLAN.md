# Rail Corrugation ML Plan

## Objective

Build a file-level classifier that predicts exactly one label for each one-second axle-box recording: `Normal`, `Side I`, or `Side II`. The official metric is macro F1, so performance on both minority fault classes matters as much as performance on the Normal class.

The labelled development set contains 272 files: 234 Normal, 14 Side I, and 24 Side II. The unlabelled test set contains 68 files. Never use filenames as features or use the test set for model selection.

## Train, Validate, and Test

1. Read each training CSV as one sample; never split its 10,000 rows across train and validation sets.
2. Extract and cache one feature row per file.
3. Evaluate candidates with `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)` and macro F1.
4. Compare balanced logistic regression, balanced RBF SVM, and class-weighted ExtraTrees. Use a Normal-only dummy classifier as the baseline.
5. Select the highest mean cross-validation macro F1, inspect per-class F1 and confusion matrices, then retrain on all 272 labelled files.
6. Run the final pipeline on all 68 test files and write `rail_predictions.csv` with `file_id,prediction`.

## Current Benchmark

The implemented pipeline was evaluated with the split above. Balanced logistic regression was selected with mean macro F1 `0.691` (`Normal: 0.955`, `Side I: 0.438`, `Side II: 0.680`). The dummy, calibrated RBF SVM, and ExtraTrees candidates scored `0.308`, `0.582`, and `0.480`, respectively. Retraining on all 272 labelled files produced `backend/artifacts/rail_pipeline.joblib`.

## Signal and Feature Strategy

Each CSV has 10,000 samples and 129 columns: rotating speed followed by vibration/shock pairs from 64 axle boxes. Preserve the physical mapping:

- Positions 1, 3, 5, and 7 belong to Side I.
- Positions 2, 4, 6, and 8 belong to Side II.
- Keep vibration and shock signals separate.

Process one file at a time as `float32`. For every side and signal type, calculate per-channel mean, standard deviation, RMS, peak-to-peak range, maximum absolute value, skewness, kurtosis, and crest factor. Add speed-transition and duty-cycle features.

Use Welch PSD at 10 kHz to calculate normalized power in `0–50`, `50–100`, `100–250`, `250–500`, `500–1000`, `1000–2000`, and `2000–5000 Hz`, plus dominant frequency and spectral centroid. Aggregate channel features with mean, median, standard deviation, and maximum, then add corresponding Side I/Side II differences and ratios.

Start with classical models rather than a neural network: there are only 272 labelled files, the classes are imbalanced, and development is CPU-only.

## Planned Module Contract

The Rail package will contain feature extraction, training, and prediction modules plus one compact final artifact. Generated feature caches, checkpoints, and prediction files belong under ignored output paths.

```bash
python -m backend.models.rail.train \
  --data-dir PS3/02_Datasets/Rail_Corrugation \
  --model-out backend/artifacts/rail_pipeline.joblib

python -m backend.models.rail.predict \
  --input PS3/02_Datasets/Rail_Corrugation/Test \
  --output outputs/rail_predictions.csv
```

This branch provides a callable Python adapter that returns the predicted class, three model scores, Side I/Side II energy summaries, and dominant-frequency diagnostics. The integration lead owns the `POST /api/predict/rail` route and can call `backend.models.rail.predict_file`. Only `file_id` and the official class label belong in the submitted CSV.

## Acceptance Checklist

- Validate 129 columns, 10,000 samples, numeric values, and exact side mapping.
- Ensure extracted features are finite and deterministic.
- Keep every source file wholly inside one validation fold.
- Beat the approximately 0.31 macro-F1 Normal-only baseline and obtain non-zero F1 for both fault classes.
- Produce exactly 68 unique test rows using only the three accepted labels.
- Preserve source filenames, including `.csv`, and use natural numeric ordering.
- Save Python, NumPy, SciPy, pandas, scikit-learn, and feature-schema versions with the model artifact.
