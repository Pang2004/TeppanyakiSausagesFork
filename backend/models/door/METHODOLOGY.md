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

### Forward validation against independent annotations

The production training command now compares these fixed candidates with five expanding-window folds. Training sizes are 20, 38, 56, 74 and 92 operations; each following validation block contains 18 operations. This evaluates 90 distinct operations once each. No validation operation or later operation enters that fold's fitting, scaling or SVM calibration. SVM calibration uses only the earlier training subset.

Supervised training features use the annotated intervals, with direction inferred from telemetry. For validation, the detector runs afresh on each contiguous raw validation block. Ground-truth times and labels come directly from `Train_Segments_Answer.csv`, independently of the detector. Missing detections, extra detections, wrong boundaries and wrong labels all affect the official score. Splits occur between annotated operations; this evaluates complete operations in held-out blocks, not arbitrary upload starts or online prediction latency.

`outputs/door/cv_results.json` records all four candidates, fold memberships, training cutoffs, independently sourced truth, and actual predicted intervals. Training selects the largest mean official score; the existing candidate order breaks learned-model ties in favour of logistic regression. This comparison is not nested model selection. The incumbent was already developed on this stream, so the results are a stronger retrospective check, not an untouched independent benchmark.

| Candidate | Earlier-to-later official score | Localization score | Abnormal-only official score |
| --- | ---: | ---: | ---: |
| Always Normal | 0.7444 | 1.0000 | 0.0000 |
| Balanced logistic regression | 1.0000 | 1.0000 | 1.0000 |
| Balanced RBF SVM | 1.0000 | 1.0000 | 1.0000 |
| Balanced ExtraTrees | 1.0000 | 1.0000 | 1.0000 |

All learned models score 1.0 in each of the five folds. Logistic regression is retained. The final estimator is fitted on all 110 operations. Its fitted scaler and classifier parameters are exactly identical to the prior artifact; all 38 unlabelled test predictions, probabilities and diagnostics are unchanged. The artifact's validation metadata has been refreshed.

### Historical results and corrected interpretation

The old five-block validation trained on the complement of each block, including later operations. It was blocked cross-validation, not a forward-only forecast. It also constructed both truth and predictions from detected segment boundaries, so it could not measure segmentation mistakes independently. Its official scores were 1.000 for all learned models and 0.7273 for Always Normal. The prior logistic repeated-stratified macro F1 was 0.99596. These historical numbers use different folds and should not be treated as gains or losses against the new 90-operation evaluation.

### Recording robustness

`python -m backend.models.door.validation` audits the fixed incumbent with the same forward folds. Training is clean in every scenario; only held-out streams are transformed. The six transformations were fixed before inspecting their scores. They do not use fault labels to choose where to corrupt the data. Annotated intervals locate the corruptions; original reference times are retained except for the explicitly time-translated scenario.

| Held-out recording condition | Official score | Localization score | Detected / true operations |
| --- | ---: | ---: | ---: |
| Original stream | 1.0000 | 1.0000 | 90 / 90 |
| Remove every tenth interior sample; retain endpoints | 1.0000 | 1.0000 | 90 / 90 |
| Remove 10 midpoint samples (200 ms of samples) per operation | 0.1703 | 0.3114 | 180 / 90 |
| Set command and activity flags inactive for 5 midpoint samples (100 ms) | 0.1753 | 0.3220 | 180 / 90 |
| Remove the last 10% of samples per operation | 0.5510 | 0.9019 | 90 / 90 |
| Compress inter-operation gaps to 20 ms without inserting idle rows | 0.2989 | 0.3212 | 30 / 90 |

Scores are means over five folds. Localization uses the same official matcher with all labels replaced by a common operation label. It isolates boundary/detection quality; it is not the competition metric. Abnormal-only scores, fold results and the full prediction/truth pairs are available in `outputs/door/validation_results.json`, alongside input hashes and feature configuration.

A ten-sample loss creates a 220 ms timestamp gap, exceeding the current 100 ms split threshold. Brief inactive flags also end a movement immediately. Both therefore split one movement into two; classification on the resulting partial waveforms deteriorates as well. Truncation directly limits recoverable end-time overlap and changes classifier inputs. Compressed gaps expose limited support for adjacent movements without an idle/reset interval, especially successive movements in the same direction. This synthetic scenario does not establish that such timing is physically representative.

These findings support retaining the current classifier for the supplied recording format, while limiting claims about continuous or interrupted streams. They do not justify choosing gap/debounce thresholds from these outer validation scores. A segmentation change should be evaluated on separately annotated interrupted recordings, with threshold selection confined to training data. The current module and tests remain as production validation tools; there are no unused experiment runners or experiment test folders.

## Validation-Backed Estimate

The strongest available retrospective estimate is **1.000 official score on 90 later operations**, with exact boundaries and correct labels for all 90. This is not a guaranteed organiser score. Only one source recording is available, physical door identifiers are absent, and the model has previously been developed on these labels. Performance on a new door, operating condition, or recording interruption remains uncertain.

The unlabelled test output contains 38 segments: 30 predicted Normal and 8 predicted Abnormal resistance. This distribution is a prediction summary, not an accuracy measurement.

## Reproduction and Interface

```bash
python -m backend.models.door.validation \
  --data-dir PS3/02_Datasets/Door \
  --output outputs/door/validation_results.json

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
