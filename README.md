# TeppanyakiSausages — PS3 Train Condition Monitoring

This repository contains our NebulaX 2026 Hackathon solution for Problem Statement 3 (PS3). We plan to attempt all four independent train-condition tasks because each contributes 25% to the overall score:

| Subsystem | Task | Scoring metric | Submission file |
| --- | --- | --- | --- |
| Door | Detect door cycles and classify abnormal resistance | IoU-weighted F1 | `door_predictions.csv` |
| ACV | Rank cars by likelihood of refrigerant leakage | Linear rank-decay | `acv_predictions.csv` |
| Rail Corrugation | Classify `Normal`, `Side I`, or `Side II` | Macro F1 | `rail_predictions.csv` |
| SHM | Estimate cumulative fatigue damage | `max(0, 1 - MAPE)` | `shm_predictions.csv` |

Start with the [official PS3 specification](PS3/01_Problem_Statement_3_Specifications.md), then read the relevant subsystem information kit in [`PS3/03_References/`](PS3/03_References/) before modelling.

## Supplied Resources

```text
PS3/
├── 01_Problem_Statement_3_Specifications.md
├── 02_Datasets/              # Door, ACV, Rail_Corrugation, and SHM data
├── 03_References/            # Authoritative subsystem information kits
└── 04_Example_Submission/    # Required prediction CSV schemas
```

Do not rename, reformat, or duplicate the supplied datasets. They remain tracked for this sprint. Generated features, temporary uploads, caches, checkpoints, and local prediction outputs should not be committed.

## Planned Solution Architecture

The following structure is the implementation target and may not exist until its owning branch is merged:

```text
backend/
├── app.py                    # FastAPI application
└── models/
    ├── door/
    ├── acv/
    ├── rail/
    └── shm/
frontend/                     # Vite React + TypeScript dashboard
tests/                        # Model, API, and submission-contract tests
outputs/                      # Generated locally; not committed
```

Every subsystem will expose the same batch interface:

```bash
python -m backend.models.<subsystem>.predict \
  --input <file-or-directory> \
  --output <prediction.csv>
```

The app will call `POST /api/predict/{subsystem}` with one uploaded file. The response contract contains `subsystem`, `output_filename`, `columns`, `rows`, `csv_text`, and a short `summary`. The React dashboard will provide subsystem selection, upload feedback, a result-specific visualization, a table, and CSV download.

## One-Day Team Plan

| Owner | Primary work | Integration responsibility |
| --- | --- | --- |
| Rail owner / ML coordinator | Streaming statistical and frequency features; class-balanced classifier | Rail label/confidence view |
| Door owner | Cycle segmentation and balanced status classifier | Segment timeline/table |
| SHM owner | Stress features and MAPE-focused regression | Damage result card |
| ACV owner / integration lead | Per-car anomaly ranking | FastAPI/React shell and ranked-car view |

Work in parallel, but freeze the shared request/response contract before model development. Each owner is responsible for training, validation, the prediction adapter, and their result component. Prioritize a complete baseline over prolonged tuning.

Suggested sequence:

1. Scaffold the backend, frontend, shared schemas, and mock responses.
2. Build four baselines in parallel and validate with each official metric.
3. Connect prediction adapters and subsystem visualizations.
4. Run the held-out inputs and validate every output schema.
5. Package `predictions.zip`, then record the app demo in under three minutes.

## Submission Gate

A subsystem is ready only when its CLI and app path both complete successfully and its CSV matches [`PS3/04_Example_Submission/`](PS3/04_Example_Submission/). Place the attempted `*_predictions.csv` files directly at the root of `predictions.zip`; do not include subfolders or raw datasets. The final submission also requires one app covering every attempted subsystem and a short end-to-end demo video.

## Optional LTA DataMall Enrichment

[`LTA_DataMall_API_User_Guide.pdf`](LTA_DataMall_API_User_Guide.pdf) remains available for optional operational context, such as route metadata, service incidents, or dashboard overlays. DataMall is not required for the four sensor-model pipelines and should not block the compulsory app or prediction outputs. External data should influence a model only when a reliable join key to the supplied PS3 observations can be demonstrated and documented.

Example request:

```bash
curl "https://datamall2.mytransport.sg/ltaodataservice/v3/BusArrival?BusStopCode=83139" \
  -H "AccountKey: ${LTA_ACCOUNT_KEY}"
```

Keep API keys in environment variables; never commit credentials or local `.env` files.

## Collaboration Rules

Use focused branches such as `model/door`, `model/rail`, or `app/frontend`, and keep commits small and imperative. Coordinate changes to shared API schemas with the integration lead. Record the validation split, metric, score, assumptions, and output-schema check in each pull request. Contributor-specific commands and conventions will live in `AGENTS.md` when the implementation scaffold is added.
