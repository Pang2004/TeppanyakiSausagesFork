# TeppanyakiSausages — Fleet Diagnostic

One app for Rail corrugation, Door resistance, ACV ventilation and SHM fatigue damage. The included Rail v3 model uses the improved local-spectrum procedure (exploratory repeated-validation Side I F1 0.5124; macro F1 0.7609).

## Contents

- `app/`: standalone React/FastAPI app, model artifacts, built frontend, Dockerfile and Makefile.
- `predictions.zip`: app-generated CSVs for the supplied test inputs, with flat ZIP entries.
- `Optional_Items/`: methodology, development code and model copies.
- `MANIFEST.sha256`: hashes of packaged files (excluding this manifest itself).
- `demo_video.*`: required video of at most three minutes; see packaging status for whether supplied.

See `Optional_Items/packaging_status.md` for remaining items and [verification notes](app/PREPARATION_VERIFICATION.md) for completed checks and limitations. Do not treat a package missing the video as submission-complete. Regenerate predictions if organizers distribute replacement test inputs.

## Run and deploy

See [app setup](app/README.md) and [Google Cloud preparation](app/DEPLOYMENT.md). From this folder:

```bash
cd app
make container-build
make container-run
# In another terminal, from app/:
make container-smoke
```

Deploy only `app/` to Cloud Run. Optional materials, predictions, and the video are submission evidence, not runtime dependencies. No cloud deployment has been performed for this package.

## Create your separate GitHub repository

Extract this entire team folder to a location **outside** the original development repository. Open a terminal in the extracted `TeppanyakiSausages` folder, then:

```bash
git init -b main
git add .
git status
# Check that no credentials or raw datasets are staged.
git commit -m "Prepare fleet diagnostics submission and Cloud Run deployment"
```

Create an empty repository on GitHub with your chosen name and visibility; do not initialize its README or .gitignore. Copy its SSH URL, then:

```bash
git remote add origin git@github.com:YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

The supplied `.gitignore` retains model artifacts, lockfiles and `predictions.zip`. It excludes the reproducible frontend build, environments, dependencies and secrets. The downloadable submission includes the built frontend for local use; a fresh Git clone rebuilds it with `cd app && make install build` or with Docker. The original manifest describes the packaged snapshot; frontend build files may be absent or differ after a rebuild.

For the actual submission, send the complete team folder/archive, not only `app/`. Add the required demo video before final submission.
