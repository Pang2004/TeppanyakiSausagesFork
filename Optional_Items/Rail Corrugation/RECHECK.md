# Rail model recheck — 19 September 2026

The current training procedure was rerun against the labelled recordings, using
the feature cache after validating its source hashes and configuration. All saved
out-of-fold rows reproduced exactly. All 12 Rail tests passed. Raw features were
not freshly extracted in this recheck. The production artifact was not replaced.

## Comparable results

Values below are mean F1 across the same 15 outer folds (five grouped folds,
three seeds). Exact duplicate recordings remain in the same fold. Scaling and
model selection occur within training subsets.

| Evaluation | Macro F1 | Side I F1 | Side II F1 |
| --- | ---: | ---: | ---: |
| Current procedure, inner selection of logistic C | 0.7212 | 0.4359 | 0.7694 |
| Fixed logistic C=10 | 0.7379 | 0.4683 | 0.7866 |
| RBF SVC C=0.1, balanced | 0.6485 | 0.3833 | 0.6616 |
| RBF SVC C=1, balanced | 0.6422 | 0.2489 | 0.7220 |
| RBF SVC C=10, balanced | 0.5266 | 0.0444 | 0.5891 |
| Automatic-shrinkage LDA | 0.6933 | 0.3801 | 0.7438 |
| Inner selection among logistic, SVC and LDA | 0.7192 | 0.4295 | 0.7697 |

The saved production estimator uses C=10, chosen by inner validation on all
training data. Its fixed-C evaluation and the nested evaluation of the selection
procedure answer different questions. Selecting the best row after seeing outer
scores is exploratory. Previously inspected feature choices also limit all these
estimates; none is a hidden-test score.

## What the reported 0.60 / 0.76 would mean

The other model's split, predictions and averaging method were not available at
the time of this recheck. Its reported values cannot yet be compared directly.
The official metric is macro F1 over Normal, Side I and Side II; weighted F1,
accuracy, a single holdout, pooled out-of-fold F1 and mean fold F1 are distinct.

For the current procedure, pooled out-of-fold macro F1 for individual repeats is
0.7510, 0.7019 and 0.7170; corresponding Side I F1 is 0.5000, 0.3871 and 0.4444.
Only 14 unique labelled Side I recordings are available. A single favourable
split can therefore differ materially from repeated validation. These findings
do not establish how the other model obtained its scores.

Across the three repeats, Side I has 20 correct predictions out of 42 prediction
events, with 16 misclassified as Normal and six as Side II. These are correlated
repeated predictions, not 42 independent recordings. Train100, Train121,
Train180 and Train185 are predicted Normal in every repeat; Train106 and Train150
are predicted Side II in every repeat. Both missed faults and false Side I
predictions matter: 21 Normal and seven Side II prediction events are classified
as Side I.

The next useful comparison is the other model on these exact grouped folds, or
both models on the same independently labelled holdout. The bounded classifier
search here provides no evidence for replacing production. The consistently
missed recordings are useful targets for further signal/feature investigation,
without treating error inspection as independent validation.

## Reproduction

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH='app:Optional_Items/Rail Corrugation/code' \
  .venv/bin/python -m rail_dev.recheck
```

This writes `code/outputs/recheck/cv_results.json` and
`code/outputs/recheck/oof_predictions.csv`, including fold memberships, inner
scores, per-class metrics and predictions. It does not update the model or test
submission predictions.
