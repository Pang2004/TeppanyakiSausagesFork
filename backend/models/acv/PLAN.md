# ACV Refrigerant-Leak Localisation Plan

## Objective and Data Interpretation

Rank every car in an ACV workbook from most to least likely to have a refrigerant leak. The official metric gives partial credit according to the true car's rank. Each workbook is one indivisible validation case; timestamp rows from a case must never be split between training and validation.

Only six labelled cases are available. Five use compact temperature/control telemetry, while one exposes direct refrigerant pressures and has usable readings for only four of its eight header-declared cars. The held-out workbook uses the same compact 67-column schema as cases 01–03. This is too little independent data for a neural network or flexible supervised classifier.

## Schema and Feature Strategy

Read the first worksheet regardless of its name and discover cars from `Car <NN> - <parameter>` headers. Map equivalent compact and rich names into cabin temperature, ambient temperature, cooling target, running mode, compressor state, and refrigerant pressure. Invalid strings, missing cells, and impossible zero temperatures are data-quality flags—not leakage evidence.

During active cooling, compare every cabin temperature and cabin-to-target error with the timestamp-wise fleet median. Aggregate only the one-sided warm deviation using its mean, upper quantiles, exceedance fractions, and longest sustained excursion. Convert the features to within-file percentile ranks and average them so different trains and operating temperatures remain comparable.

When both refrigeration circuits have valid pressure telemetry, also compare their median high pressure, low pressure, pressure lift, and compressor duty cycle. Weight this pressure-circuit score at 70% and temperature evidence at 30%. Cars without usable telemetry remain in the required output but rank last.

## Validate and Predict

Compare the hybrid ranker against fixed-order, hottest-median, and temperature-only baselines using whole-workbook evaluation and the official rank-decay score. Report each true-car rank, mean score, top-1 accuracy, and the rich-schema limitation. Package the selected fixed configuration and dependency versions in the artifact.

The implemented temperature ranker achieved a mean rank-decay score of `0.9792`, placing five faults first and the rich-schema fault second. Adding pressure-circuit evidence placed every disclosed fault first for a retrospective score and top-1 accuracy of `1.0`. The hottest-median and fixed-order baselines scored `0.8125` and `0.7708`, respectively. Because only one labelled rich-schema case exists, its pressure result is evidence of feasibility rather than an independent generalisation estimate.

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

The callable adapter returns the ordered cars, comparable scores, data coverage, temperature evidence, pressure evidence, and schema type. The official CSV contains only `file_id,ranked_cars`, with every two-digit identifier joined by `|`.

## Acceptance Checks

- Preserve header car identifiers and include each exactly once.
- Rank all six disclosed training faults first retrospectively.
- Never use filename, train number, timestamp, or car identity as a feature.
- Test compact/rich schemas, invalid telemetry, blank cars, non-English sheet names, deterministic ties, artifact reload, and official CSV formatting.
- Keep the tracked artifact and all tracked `outputs/` reports current.
