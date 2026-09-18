# Door Fault Diagnosis Plan

## Objective and Data Interpretation

Detect every door operation in a continuous sensor stream and classify it as `Normal` or `Abnormal resistance`. One scored cycle is one opening or one closing operation—not an Open/Close pair. The supplied files do not contain train, car, or door identifiers, so the pipeline must not claim physical door identity.

`Train.csv` contains 110 operations and `Test.csv` contains 38. Samples within an operation are 20 ms apart, while recorded operations are separated by timestamp gaps of at least 10 seconds. Gaps are the evidence-backed benchmark boundary; a direction-aware state machine provides a fallback for genuinely continuous feeds.

## Stream Column Reference

| Exact column | Meaning and use |
|---|---|
| `Datetime` | Timestamp formatted as `year-month-day-hour-minute-second-millisecond`. It establishes order, sampling cadence, and recorded gaps. |
| `Motor current(mA)` | Motor current in milliamperes. Resistance can change its magnitude and waveform. |
| `Motor Voltage(10mV)` | Voltage in 10 mV units. Values remain raw for modelling; multiply by `0.01` for volts when displaying them. |
| `Motor electrodynamic force` | Motor back-EMF. Its physical unit is undocumented, so treat it as a raw controller value. |
| `Door opening time(.1s)` | Recorded/configured opening duration in tenths of a second; it is not the sample timestamp. |
| `Door closing time(.1s)` | Recorded/configured closing duration in tenths of a second. |
| `Close command` | Binary request to close the door. |
| `Open command` | Binary request to open the door. |
| `DCSR` | Door Close Switch Right; actuates during closing and releases during opening. |
| `DCSL` | Door Close Switch Left; same transition meaning as DCSR. |
| `DLSR` | Door Locked Switch Right; actuates while closing/locking and releases while opening. |
| `DLSL` | Door Locked Switch Left; same transition meaning as DLSR. |
| `Door Opened` | Binary indication that the fully open state has been reached. |
| `Door Locked` | Binary locked indication. It is constant at `0` in the supplied streams and is excluded as a zero-variance feature. |
| `Door is opening` | Binary active-opening state. |
| `Door is closing` | Binary active-closing state. |
| `Door leaf position` | Raw position. Supplied values are near `0` closed and `700` open, but no physical unit or universal calibration is documented. |

The general header reference mentions `Car Type`, `Car Number`, and `Door Number`; these columns are absent from the actual Train and Test streams.

### Training answer columns

| Column | Meaning |
|---|---|
| `segment_id` | Unique ground-truth operation identifier. |
| `start_time` / `end_time` | First and last sample timestamps for the operation. |
| `operation` | Informational `Open` or `Close` value; it is not a submission target. |
| `status` | Classification target: `Normal` or `Abnormal resistance`. |
| `n_rows` | Number of stream rows belonging to the operation. |

## Segmentation and Features

Calculate the median positive sampling interval and split at deltas greater than five times that cadence, with a minimum threshold of 100 ms. This exactly recovers the supplied labels. Do not end at the first terminal position: the labels include approximately 0.22–0.48 seconds of terminal dwell.

Within uninterrupted data, start on command/activity flags and finish when they become inactive or when the terminal state remains stable for 500 ms. Closing requires decreasing relative position and actuated close/lock switches; opening requires increasing relative position, `Door Opened`, and released close/lock switches. A ten-second maximum prevents unbounded segments. Position uses relative progress and local extrema rather than fixed `0`/`700` thresholds. Validation failures become diagnostics and do not silently remove a segment.

For every segment, extract duration, operation, robust statistics, RMS, first-difference magnitudes, peak locations, binary transition counts, and 20 normalized-time waveform samples for current, voltage, back-EMF, and position.

## Train, Validate, and Predict

Evaluate a Normal-only baseline, balanced logistic regression, calibrated balanced RBF SVM, and class-weighted ExtraTrees. Use five chronological segment folds and the official IoU-weighted score for selection; use repeated stratified macro F1 and abnormal recall as secondary diagnostics. Resolve equal top scores in favour of logistic regression, then retrain on all 110 labelled segments.

The implemented comparison scored `0.727` for the Normal-only baseline and `1.000` for each of the three learned candidates on chronological validation. The tie-break selected balanced logistic regression. Its repeated-stratified macro F1 was `0.996`; chronological macro F1 and abnormal recall were `1.000`.

```bash
python -m backend.models.door.train \
  --data-dir PS3/02_Datasets/Door \
  --model-out backend/artifacts/door_pipeline.joblib

python -m backend.models.door.predict \
  --input PS3/02_Datasets/Door/Test.csv \
  --model backend/artifacts/door_pipeline.joblib \
  --output outputs/door_predictions.csv
```

The Python adapter returns confidence, inferred operation, boundary reason, and quality flags for the app. The official CSV contains only `start_time,end_time,prediction`, ordered chronologically.

## Acceptance Checks

- Reproduce all 110 training boundaries, operations, and row counts exactly.
- Produce 38 ordered, non-overlapping Test segments with accepted labels.
- Beat the approximately `0.727` all-Normal official-score baseline.
- Keep preprocessing inside each validation fold and save feature/dependency versions with the artifact.
- Test malformed timestamps, schema errors, continuous-feed fallback, artifact reload, and official greedy IoU matching.
