# TeppanyakiSausages — PS3 Fleet Diagnostic

A responsive React application for train-condition monitoring, backed by the validated Python models. **Home, Rail, Door, ACV and SHM are implemented**, with real model inference, independent upload batches and CSV exports.

## Start locally

Use the existing Python environment, or create Python 3.12 `.venv` and install `requirements-dev.txt`. Node.js 24 is used for the frontend build.

```bash
npm ci --prefix app/frontend
npm run build --prefix app/frontend
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m uvicorn backend.api:app --app-dir app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**. A phone on the same Wi-Fi can use **http://<computer-LAN-IP>:8000** while the server runs. The frontend and API are served from the same address; no public hosting is configured.

See [app/README.md](app/README.md) for first-time setup, Windows notes, phone access, uploads, configuration and browser tests.

## Repository layout

```text
app/
  frontend/                 React source, build configuration and browser tests
  backend/
    api.py                  Upload API and built frontend hosting
    models/                 Runtime parsing, feature extraction and prediction
    artifacts/              Active trusted model artifacts
  exports/                  Predictions downloaded through the app
  requirements.txt          Pinned runtime dependencies
Optional_Items/
  Door/
  ACV/
  Rail Corrugation/
  SHM/                      Each has methodology, code/<subsystem>_dev and reports
  UI_References/            Original HTML designs and screenshots
  write_up.md
PS3/                        Original supplied data/reference material; development only
scripts/                    Development launcher and submission packager
tests/                      API and packaging checks
```

Runtime modules remain importable as `backend.models.<subsystem>` with `app/` on the Python path. Training and evaluation live in optional packages named `rail_dev`, `door_dev`, `acv_dev`, and `shm_dev`. The launcher handles both paths and runs from the repository root:

```bash
.venv/bin/python scripts/model.py rail predict --input PS3/02_Datasets/Rail_Corrugation/Test --output /tmp/rail_predictions.csv
.venv/bin/python scripts/model.py rail train --help
.venv/bin/python scripts/model.py door validation --help
.venv/bin/python scripts/model.py shm validation --help
.venv/bin/python -m pytest -q
```

ACV validation is part of its `train` command. Training defaults save artifacts in `app/backend/artifacts` and reports in the corresponding optional subsystem's `code/outputs`. Shell-quote paths containing `Rail Corrugation`.

### Model methodologies

- [Rail Corrugation](Optional_Items/Rail%20Corrugation/METHODOLOGY.md)
- [Door](Optional_Items/Door/METHODOLOGY.md)
- [ACV](Optional_Items/ACV/METHODOLOGY.md)
- [SHM](Optional_Items/SHM/METHODOLOGY.md)

Artifacts alone are insufficient: the app also needs inference/feature code and compatible Python dependencies. It does not need training data or development reports. Door, ACV and SHM retain their existing trained parameters. Rail now uses the validated local-spectrum v3 integration; the previous v2 artifact is retained for rollback.

## PS3 packaging

The working repository retains the supplied `PS3/` materials for development; they are **never included in the submission package**. Do not submit a zip of the entire repository.

```bash
.venv/bin/python scripts/package_submission.py
# With the finished demo:
.venv/bin/python scripts/package_submission.py --output submission-final --demo-video /path/to/demo.mp4
```

The packager creates:

```text
TeppanyakiSausages/
  app/                      Self-contained source and built app
  Optional_Items/
    write_up.md
    Door/code/ and model/
    ACV/code/ and model/
    Rail Corrugation/code/ and model/
    SHM/code/ and model/
  predictions.zip           Existing app exports, flat CSV entries
  demo_video.mp4            Only when supplied
```

Use `--team-name` if the registered spelling differs. Existing output folders are not overwritten. Only app exports are packaged by default; old CLI-generated predictions remain development evidence. The packaging status explicitly reports missing predictions/video. A package made now is an incremental Rail demo, **not the final four-subsystem submission**: Door, ACV and SHM must be connected to this same app, their test predictions exported through it, and an end-to-end demo of at most three minutes recorded.

The original [PS3 specification](PS3/01_Problem_Statement_3_Specifications.md) is authoritative.
