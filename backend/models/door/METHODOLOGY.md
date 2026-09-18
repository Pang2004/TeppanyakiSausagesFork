# Door Fault-Diagnosis Methodology

## Task and Data Interpretation

The Door pipeline receives one continuous telemetry stream and must first find each door movement, then classify that movement as `Normal` or `Abnormal resistance`. One prediction represents one opening or one closing operation—not an open/close pair. The supplied data contains no train, car, or physical door identifier, so the model does not claim to identify which installed door produced an operation.

The training stream contains 110 labelled operations: 80 Normal and 30 Abnormal resistance. Samples inside an operation are normally 20 ms apart. Recorded operations are separated by gaps of at least 10 seconds, whereas the test stream contains 38 operations in the same format. The official score combines temporal overlap and the predicted label, so reliable boundaries are as important as classification.

## Segmentation

Timestamps are parsed from `year-month-day-hour-minute-second-millisecond` strings and checked for increasing order. The pipeline estimates the median positive sampling interval and splits recorded blocks where the time delta exceeds five times that cadence, with a minimum split threshold of 100 ms. This gap-based method exactly reproduces all 110 supplied training boundaries, operations, and row counts.

A direction-aware state machine is also implemented for genuinely continuous uploads where large gaps do not exist. It starts an operation from command and activity flags, infers direction from the door position and switch states, and requires a stable terminal state for 500 ms before ending the segment. A 10-second maximum prevents an incomplete movement from consuming the rest of the stream. Position is interpreted relative to the movement's local range instead of assuming that `0` and `700` are universal calibrations. The labelled data includes a short terminal dwell, so segmentation deliberately does not stop at the first terminal-position sample.

Each segment carries its boundary reason and quality flags. These diagnostics make malformed or incomplete operations visible to the app without silently discarding them.

## Feature Engineering

The final classifier uses 177 deterministic features per operation. Duration, sample count, and opening-versus-closing direction provide operation context. Motor current, voltage, back-EMF, and door position each contribute mean, standard deviation, extrema, quantiles, range, RMS, first-difference magnitude, and peak location. Each signal is also interpolated to 20 normalized-time points, preserving waveform shape even when operations have different durations. Position is locally normalized before interpolation.

The configured opening and closing times contribute median values. Command, activity, opened, close-switch, and lock-switch channels contribute start value, end value, mean, and transition count. `Door Locked` is omitted because it is constant in the supplied streams. All preprocessing is performed inside the fitted pipeline, and extracted features must be finite.

## Model Selection

Four candidates were compared:

| Candidate | Purpose |
| --- | --- |
| Always Normal | Establish the class-imbalance baseline |
| Standardized balanced logistic regression | Linear, interpretable classifier with minority-class weighting |
| Standardized balanced RBF SVM with sigmoid calibration | Non-linear comparison model |
| 500-tree balanced ExtraTrees | Tree-based comparison model |

The primary validation uses five chronological blocks so later operations are not randomly mixed into every fold. Each fold is scored using the official greedy same-label temporal-IoU metric. Macro F1 and abnormal-resistance recall are secondary checks. A repeated stratified 5-fold split with three repeats measures label-classification stability independently of chronology.

All three learned candidates achieved `1.000` mean official score on the chronological folds. The stored artifact uses balanced logistic regression as the simpler tie-break choice and retrains it on all 110 operations. Its repeated-stratified macro F1 was `0.996`; chronological macro F1 and abnormal-resistance recall were both `1.000`.

## Validation-Backed Estimate

| Result | Score |
| --- | ---: |
| Always-Normal official-score baseline | `0.727` |
| Selected model chronological official score | `1.000 ± 0.000` |
| Selected model chronological macro F1 | `1.000` |
| Selected model abnormal-resistance recall | `1.000` |
| Selected model repeated-stratified macro F1 | `0.996` |

These figures are the best available estimate, not a guaranteed organiser score. Segments originate from one supplied stream and may share operating conditions, so even chronological validation can be easier than a new train, door mechanism, or resistance pattern. The perfect temporal score also assumes that the gap structure of the held-out recording remains recoverable; the continuous-stream fallback is tested but has less labelled evidence.

The generated held-out output contains 38 segments: 30 predicted Normal and 8 predicted Abnormal resistance. Their labels are unknown, so this distribution is a prediction summary rather than an accuracy result.

## Reproduction and Interface

```bash
python -m backend.models.door.train \
  --data-dir PS3/02_Datasets/Door \
  --model-out backend/artifacts/door_pipeline.joblib

python -m backend.models.door.predict \
  --input PS3/02_Datasets/Door/Test.csv \
  --model backend/artifacts/door_pipeline.joblib \
  --output outputs/door_predictions.csv \
  --diagnostics-output outputs/door/diagnostics.json
```

The Python adapter returns timestamps, label, confidence, inferred operation, boundary reason, and quality flags. The official CSV intentionally contains only `start_time,end_time,prediction`, ordered chronologically.
