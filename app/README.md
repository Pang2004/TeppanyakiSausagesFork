# Fleet Diagnostic — Four Subsystems

React + TypeScript frontend and FastAPI backend for all four diagnostic models. Home links to Rail, Door, ACV and SHM. The supplied designs are adapted to desktop/mobile, with a shared Exit to Main Page control and real model outputs.

## Run the app

Requirements: Python 3.12 and Node.js 24 (Node is only needed to build or develop the frontend).

From this `app/` directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
npm ci --prefix frontend
npm run build --prefix frontend
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

If using the existing repository environment, from the repository root:

```bash
npm ci --prefix app/frontend
npm run build --prefix app/frontend
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m uvicorn backend.api:app --app-dir app --host 0.0.0.0 --port 8000
```

On Windows, activate with `.venv\Scripts\Activate.ps1`, set thread environment variables with PowerShell, and use `python` instead of `.venv/bin/python`.

Open **http://localhost:8000** on the computer. On a phone connected to the same Wi-Fi, open **http://<computer-LAN-IP>:8000**. Find the LAN IP in the computer's network settings or with `hostname -I` on Linux / `ipconfig` on Windows. Keep the server running and permit local-network access to port 8000 if the firewall prompts. Guest Wi-Fi/client isolation may prevent devices reaching one another.

The production build is served by Python; a second Node server is not needed. `/`, `/rail`, `/door`, `/acv` and `/shm` support direct reloads. Internet access is not needed after dependencies are installed and the frontend is built: fonts and the supplied background image are bundled locally.

This app has no accounts. It is prepared for a public Cloud Run demo but has not been deployed. See [deployment instructions](DEPLOYMENT.md) and the included Dockerfile/Makefile. The container uses a 30 MiB upload limit.

## Use Rail

1. Click the Rail card or illustrated track on Home.
2. Choose CSV files or a folder, or drag them onto the upload area. On browsers without directory support, use **Choose files**.
3. Analysis starts automatically. One file is uploaded and processed at a time.
4. Select a recording to inspect its predicted side; download `rail_predictions.csv`.

The model requires the official Rail schema: 10,000 rows and 129 named columns. A selected folder can include other files; those are ignored with a count. Filenames must be unique within a batch, including nested folders. Empty/oversized files are rejected; errors are never treated as a Normal prediction.

Default upload limit: **64 MiB per recording**. `RAIL_MAX_FILE_BYTES` overrides it. `RAIL_MODEL_PATH` can override the trusted server-side artifact path. Uploaded model files are not supported.

Stop aborts the active upload/request and prevents remaining files from being submitted. A model calculation that has already started may finish on the server before its temporary file is removed. Failed/stopped recordings can be retried. Partial downloads explicitly omit unfinished/failed recordings. Results remain while navigating in the same tab, but refreshing clears them. Uploaded files are held only in request-scoped temporary storage and removed when processing finishes, including errors.

The rail image shows the recording's predicted class, not exact defect locations or two independent side diagnoses. Model scores are not guarantees of correctness.

## What the app needs

The `.joblib` artifact alone is **not sufficient**. `backend/models/rail/` supplies schema checking and the exact feature extraction/prediction code, including wavelength features. `requirements.txt` pins the Python libraries used by the saved model. The runtime includes active artifacts and API/UI routes for all four models.

The app does **not** require `PS3/`, training CSVs, validation caches, or `Optional_Items/` to start or predict uploaded recordings.

## Development and verification

For frontend hot reload, run the Python server as above, then `npm run dev --prefix frontend`. Vite serves on port 5173 and proxies relative `/api` calls to Python on port 8000. Production and phone testing should normally use port 8000.

```bash
npm test --prefix frontend
npx --prefix frontend playwright install chromium webkit
# Start the built app on port 8000 first; run from app/frontend:
cd frontend
npm run test:e2e
```

On Linux, WebKit may need `npx playwright install-deps webkit` with administrator privileges. Browser tests use the repository's development datasets; the standalone packaged app does not include them. `APP_URL` can point the tests at a different local server. The full 68-file parity test runs once in desktop Chromium and saves the app-generated CSV in `app/exports/` only after it matches the validated output.

API endpoints:

- `GET /api/health`: readiness and file size limit.
- `POST /api/predict/rail`: one multipart field named `file`; returns `file_id`, `prediction`, `scores`, `side_energy`, `dominant_frequency`, `subsystem`, and `model_version`.
- Errors: JSON `error.code` and `error.message`; 413 for size, 422 for invalid recordings, 503 for model unavailable, 500 for unexpected inference failure.

## Door, ACV and SHM tabs

Each tab has its own batch, selection, readiness and results. Direct URLs `/door`, `/acv` and `/shm` work on refresh. The exit control always returns to `/`.

- **Door:** Use **Download selected stream submission CSV** for the official three-column submission without `file_id`. The existing download remains a multi-stream batch export. CSV current streams; select an Open/Close operation to inspect its classification, start/end clock times and duration. Exports include `file_id,start_time,end_time,prediction` to disambiguate multiple uploaded streams; this batch export has an extra source column compared with the official single-stream Door submission format.
- **ACV:** XLSX workbooks, including folder selection. All returned cars appear in ranked order with actual temperature and coverage diagnostics. Scores are not presented as probabilities; missing measurements show “Not available”. No work order is dispatched.
- **SHM:** CSV stress histories. Displays actual fatigue damage, cycle count, equivalent stress amplitude and maximum cycle range. Damage is not converted to remaining life. The battery built into the train displays D × 100% and saturates at 100%; the numeric output/export is never clipped.

Additional multipart endpoints: `POST /api/predict/door`, `/api/predict/acv`, `/api/predict/shm`. Each accepts a `file` field. `/api/health` reports separate readiness flags under `subsystems`. Optional server artifact overrides: `DOOR_MODEL_PATH`, `ACV_MODEL_PATH`, `SHM_MODEL_PATH`.

See [UI verification](UI_VERIFICATION.md) for verification coverage and browser limitations.

## Rail local-spectrum update

The active Rail artifact is `rail-pipeline-v3`, with local per-car wavelength energy and training-only side mirroring. Its production implementation exactly reproduced all 1,632 stored held-out predictions. Six-repeat exploratory macro F1 is 0.7609; Side I F1 is 0.5124. See the Rail methodology for selection limitations and the full-data inner comparison.

Rollback: set `RAIL_MODEL_PATH` to `backend/artifacts/rail_pipeline_baseline_v2.joblib` (relative to the server working directory), then restart the API. The existing v2 model and its prior results are preserved. The API reports the actually loaded artifact version.
