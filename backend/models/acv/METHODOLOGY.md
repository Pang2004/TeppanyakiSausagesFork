# ACV Refrigerant-Leak Localisation Methodology

## Task and Dataset Constraints

The ACV pipeline ranks every car in a workbook from most to least likely to have a refrigerant leak. The official linear rank-decay metric gives partial credit when the faulty car is near the top, so the output is a complete ordering rather than one class label.

Only six labelled cases are available, with exactly one disclosed faulty car per case. Five cases use compact temperature and controller telemetry. One case has a 483-column rich schema with direct refrigerant pressures, but only four of its eight header-declared cars contain usable readings. The held-out workbook uses the same compact 67-column schema as training cases 01–03. This sample size is insufficient for a neural network or a flexible supervised classifier, so the implemented model is a fixed, physics-informed within-train anomaly ranker.

## Schema Normalization and Data Quality

The loader reads the first worksheet regardless of its name, which supports the non-English held-out sheet name. Cars are discovered dynamically from `Car <NN> - <parameter>` headers, and their two-digit identifiers are preserved exactly for submission.

Equivalent headers are mapped into canonical cabin temperature, ambient temperature, cooling target, running mode, information-valid status, compressor state, and high/low refrigerant pressure. This mapping supports both compact schema variants and the rich workbook without assuming a fixed column order.

`Invalid`, missing, and impossible temperature values outside `10–50` are excluded from physical scoring. A zero temperature is retained only as a data-quality diagnostic; it is not evidence that a refrigerant leak exists. Cars with insufficient telemetry remain in the required ranking but appear after cars with usable evidence. Deterministic ties use data coverage and then the original car identifier.

## Temperature Anomaly Score

Analysis is restricted to active cooling modes: Automatic Cooling, Full Cooling, and Half Cooling. At each timestamp, the model calculates the median cabin temperature across cars with valid readings. A minimum of three valid cars, including the candidate car when valid, is required for the fleet median. This is not a leave-one-car-out reference.

For each car, the model measures both cabin-temperature deviation from the fleet median and cabin-to-target error relative to the other cars. Leakage evidence is directional: sustained warmer behaviour is suspicious, whereas an unusually cold car is not assigned an equivalent fault score. The feature set includes positive mean deviation, 75th/90th/95th percentiles, fractions above `0.25`, `0.50`, and `1.00`, relative target-error statistics, and the longest sustained excursion above `0.50`.

Every feature is converted into a within-workbook percentile rank and averaged. Ranking within the same train reduces sensitivity to absolute weather, target setpoint, and sensor calibration differences between cases.

## Pressure-Circuit Score

When both refrigeration circuits provide sufficient high pressure, low pressure, and compressor-state samples, the model compares the two circuits inside each car. It calculates normalized differences in median high pressure, median low pressure, pressure lift, and compressor duty cycle. A leak affecting one circuit should create a larger internal mismatch than normal control variation shared across cars.

The pressure mismatch is itself percentile-ranked across usable cars. When both evidence types exist, the final score is 70% pressure and 30% temperature. When pressure is unavailable—as in the held-out compact schema—the temperature score is used alone. This conditional design avoids fabricating pressure values or rejecting otherwise valid compact workbooks.

## Model Comparison and Validation

The fixed hybrid ranker is compared with three baselines: car-identifier order, hottest median cabin temperature, and the temperature anomaly score without pressure. Evaluation keeps each workbook intact and applies the official rank-decay formula to the disclosed faulty car.

| Ranking method | Mean rank-decay score | Top-1 accuracy | Mean true-car rank |
| --- | ---: | ---: | ---: |
| Fixed identifier order | `0.7708` | `33.3%` | `2.83` |
| Hottest median | `0.8125` | `66.7%` | `2.50` |
| Temperature anomaly | `0.9792` | `83.3%` | `1.17` |
| Hybrid temperature/pressure | **`1.0000`** | **`100%`** | **`1.00`** |

Temperature evidence ranks the fault first in all five compact cases and second in the rich case. Pressure imbalance moves the rich case's car 01 to first, producing six retrospective top-ranked faults.

This `1.000` result must be interpreted cautiously. The ranker is deterministic rather than fitted fold-by-fold, and feature design was inspected against only six disclosed cases. Most importantly, there is only one labelled rich-pressure case, so pressure performance is a feasibility result rather than an independent generalization estimate. The temperature-only `0.9792` score covers all six cases, including the rich case where its fault ranks second. On the five comparable compact cases alone, its score is `1.0000`; these are still retrospective results.

The generated held-out ordering is `01|04|03|08|07|02|06|05`. The correct car is undisclosed, so this is the model output—not a confirmed result.

## Gap and Missing-Signal Robustness Audit

The current ranker was compared with one fixed candidate that breaks a hot run at invalid or non-increasing timestamps, or at gaps exceeding twice the median positive sampling interval of that workbook portion. No threshold was tuned against labels. The denominator remains the number of rows; this corrects continuity without claiming to estimate an elapsed-time exposure fraction. The current implementation counts consecutive hot rows and can bridge recording gaps.

Each of the six labelled workbooks was checked whole and in four consecutive equal-row portions. Additional checks removed all cooling targets, all pressure/compressor columns, each individual cabin-temperature column in turn, and each of the eleven temperature scoring features in turn. These scenarios retain every car identifier in the required ranking. Removing the faulty car's cabin reading is reported separately because it can remove the evidence needed for diagnosis.

| Check | Cases/scenarios per candidate | Current rank-decay | Gap-aware candidate | Current/candidate top-1 rate |
| --- | ---: | ---: | ---: | ---: |
| Full workbook | 6 | 1.0000 | 1.0000 | 100% / 100% |
| Four consecutive portions per workbook | 24 | 0.9688 | 0.9688 | 83.3% / 83.3% |
| All cooling targets removed | 6 | 1.0000 | 1.0000 | 100% / 100% |
| All pressure/compressor data removed | 6 | 0.9792 | 0.9792 | 83.3% / 83.3% |
| Another car's cabin sensor removed | 42 | 1.0000 | 1.0000 | 100% / 100% |
| One temperature scoring feature removed | 66 | 1.0000 | 1.0000 | 100% / 100% |
| Faulty car's cabin sensor removed (information-loss check) | 6 | 0.2708 | 0.2708 | 16.7% / 16.7% |

Results average scenarios within each workbook and then average workbooks equally. The fixed adoption gate required no whole-workbook rank regression, a strictly higher quarter-window score, and no regression in the other-cabin/target/pressure/feature-removal checks. The candidate did not pass because no faulty-car rank improved. The production ranker and saved predictions are retained. Correcting continuity changed persistence features, but those changes did not improve the measured fault ranking; this does not prove that gap handling could never help another recording.

The portion-level ranks for both methods are case 01: `1,1,1,1`; case 02: `2,1,1,1`; case 03: `1,1,1,4`; case 04: `1,1,1,1`; case 05: `2,2,1,1`; case 06: `1,1,1,1`. The compact-only portion score is `0.9625` with 80% top-1; the single rich case scores `1.0000`. Its pressure component also permits localisation when its faulty cabin sensor is removed. Temperature-only compact cases cannot support reliable diagnosis once that fault evidence is removed; ranking unsupported cars last must not be interpreted as evidence that they are healthy.

These are robustness observations, not independent validation. The six cases informed original feature development, portions from one workbook are correlated, and the disclosed label does not specify when the fault signature becomes detectable. Missing-signal scenarios are controlled perturbations, not additional documented faults. The pressure branch still has only one labelled case. The held-out test workbook was not used to choose the candidate or evaluate the adoption gate.

`outputs/acv/robustness_results.json` records the exact protocol, data cadence/gap counts, schema-specific summaries, feature changes, all 312 paired observation rows and the decision. Current full-workbook rankings were checked against the existing validation report, and each observation was checked against the official rank-decay formula.

## Reproduction and Interface

```bash
python -m backend.models.acv.train \
  --data-dir PS3/02_Datasets/ACV \
  --model-out backend/artifacts/acv_pipeline.joblib

python -m backend.models.acv.predict \
  --input PS3/02_Datasets/ACV/Test \
  --model backend/artifacts/acv_pipeline.joblib \
  --output outputs/acv_predictions.csv \
  --diagnostics-output outputs/acv/diagnostics.json
```

The callable adapter returns the ordered cars, comparable scores, schema type, coverage, temperature evidence, pressure evidence, and invalid-value diagnostics. The official CSV contains only `file_id,ranked_cars`, with every discovered identifier included exactly once and joined by `|`.
