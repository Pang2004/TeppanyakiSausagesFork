# Side I improvement study — 19 September 2026

Side I improved from **0.4359 to 0.5013 mean fold F1** in the strongest
inner-selected procedure tested here. Macro F1 improved from 0.7212 to 0.7320,
while Side II decreased from 0.7694 to 0.7392. **0.60 Side I F1 was not achieved
across repeated validation.** Production remains unchanged.

## Same-fold comparison

All rows use five SHA-256-grouped outer folds, repeated with seeds 42–44.
Three inner folds select configurations and decision adjustments. Each outer
validation recording, including exact duplicates, is excluded from fitting,
scaling, feature selection, augmentation and decision tuning.

| Procedure | Side I F1 | Macro F1 | Side II F1 |
| --- | ---: | ---: | ---: |
| Existing logistic C selection | 0.4359 | 0.7212 | 0.7694 |
| Side I decision adjustment and logistic C selection | 0.4446 | 0.7195 | 0.7630 |
| Inner macro selection over transforms/augmentation | **0.5013** | **0.7320** | 0.7392 |
| Inner Side I selection with macro guard, transforms | 0.4562 | 0.7259 | 0.7671 |
| Inner macro selection including hierarchy/Extra Trees | 0.4280 | 0.7334 | 0.8104 |
| Inner Side I selection with macro guard, architectures | 0.4284 | 0.7107 | 0.7488 |

The guard accepts configurations whose inner macro F1 is no more than 0.02
below the original procedure's inner score. It does not guarantee the same
constraint on outer validation. Decision adjustment multiplies the Side I class
score by 0.5, 1, 2 or 4 before choosing the largest score; these are decision
scores, not calibrated probabilities.

The first search includes raw/signed-log features, mirrored training examples,
their combinations, and selection of 80 features inside training folds.
Mirroring exchanges Side I/II feature summaries, reverses signed contrasts and
exchanges fault labels. It assumes useful fault patterns transfer between sides;
it does not create independent recordings. A raw-sensor swap test confirms the
feature mapping. The second search tests fault-versus-normal followed by fault
side classification, plus balanced Extra Trees models. These did not improve
Side I relative to the first search.

For exploratory fixed configurations, log compression plus mirrored training
at C=0.1 scored 0.5084 Side I / 0.7398 macro. Mirroring alone at C=0.1 scored
0.4888 Side I / 0.7448 macro. Those rows were compared after seeing outer results;
they must not be described as independently selected performance estimates.

All experiments reuse previously inspected data and features. Nested tuning
avoids direct outer-fold tuning but does not remove the bias from comparing
these research procedures. Further independent validation is needed before
claiming a reliable gain or replacing production.

## Which recordings need attention?

Correct predictions out of three repeats, comparing the existing procedure
against the transform search selected by inner macro F1:

| Side I recording | Existing | New | Observation |
| --- | ---: | ---: | --- |
| Train100 | 0/3 | 2/3 | Newly recovered in two repeats |
| Train121 | 0/3 | 1/3 | Partially recovered |
| Train170 | 1/3 | 3/3 | More stable detection |
| Train106 | 0/3 | 0/3 | Still consistently missed |
| Train150 | 0/3 | 0/3 | Still consistently missed |
| Train180 | 0/3 | 0/3 | Still consistently missed |
| Train185 | 0/3 | 0/3 | Still consistently missed |
| Train163 | 2/3 | 1/3 | Regression |
| Train202 | 3/3 | 2/3 | Regression |

Train62, Train83, Train137 and Train213 remain correct in all repeats;
Train194 remains correct in two. Improvements therefore include tradeoffs.

Train180 and Train185 run at approximately 18.55 and 18.40 m/s and have lower
mean vibration RMS on Side I than Side II. Train106 and Train150 have almost
equal mean vibration RMS between sides. This is an observation, not evidence
of incorrect labels: corrugation need not make the whole side's mean RMS larger.
Speed-adjusted spectral evidence and local sensor behaviour deserve inspection.

Across all repeats the new procedure has 23 Side I true positives, 26 false
positives and 19 false negatives, compared with 20/28/22 originally. These are
42 correlated Side I prediction events from 14 recordings. Pooled Side I F1 is
0.5055, slightly different from mean fold F1 of 0.5013. Per-repeat pooled Side I
F1 is 0.5161, 0.5185 and 0.4848: the improvement is not a single 0.60 split.

## Practical route toward 0.60

1. Keep mirrored training as a research candidate: it transfers information
   from the 24 Side II examples. Threshold adjustment by itself was insufficient.
2. Inspect speed-normalized spectra for Train106/150/180/185 alongside normal
   recordings at similar speed. Test compact per-car or paired-axle band-energy
   contrasts and consistency across sensors, which may retain local fault
   evidence lost in whole-side summaries. This is a hypothesis for the next
   experiment, not a demonstrated improvement; prior broad spatial features
   were unsuccessful as documented in METHODOLOGY.md.
3. Address false alarms as well as missed faults. On a holdout containing all
   14 Side I recordings, nine correct detections, five misses and seven false
   alarms would give F1 = 18 / (18 + 5 + 7) = 0.60. This is an illustrative target,
   not the achieved confusion matrix or a mean-fold score.
4. Compare the other model on these exact folds. A 0.60 result on a different
   local split is useful evidence to investigate, but is not yet a comparable
   benchmark. Additional independently labelled Side I recordings would also
   help distinguish reliable improvements from split variation.

## Reproduction and outputs

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH='app:Optional_Items/Rail Corrugation/code' \
  .venv/bin/python -m rail_dev.side_i_study

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH='app:Optional_Items/Rail Corrugation/code' \
  .venv/bin/python -m rail_dev.side_i_study --stage architecture
```

Results and per-recording predictions are under `code/outputs/side_i_study/`
and its `architecture/` subdirectory. `file_audit.csv` compares each recording's
correct detections and Side I false alarms across repeats. Feature caches are
validated against raw-file hashes and configuration before reuse.

All 15 Rail tests passed, including raw-sensor mirroring, the inner macro guard,
and hierarchical class-score composition. No production model or submission
predictions were replaced.
