# Google Cloud Run preparation

This package is prepared for a public judge/demo service. No Google Cloud resources have been created and no cloud credentials are included.

## Local prerequisites and verification

Install Docker Engine/Desktop and Make. The container uses Node 24 only during the frontend build and Python 3.12 at runtime. From this `app/` directory:

```bash
make container-build
make container-run
# In another terminal:
make container-smoke
```

Open http://localhost:8080 and upload a recording in each tab. The Dockerfile runs as a non-root user, starts one Uvicorn worker, and loads all four saved models. Docker rebuilds the frontend; it does not need a pre-existing `frontend/dist` directory. The runtime image excludes optional development code, data, references and exports.

The current preparation environment does not have Docker installed. Native packaged-app checks are documented in [preparation verification](PREPARATION_VERIFICATION.md); they do not substitute for a container build/run check. Run these container checks before cloud deployment. The initial 2 CPU / 2 GiB settings must be checked against your recordings and concurrent usage.

## Deploy later, when you choose

Use a billing-enabled Google Cloud project and install the Google Cloud CLI. The operator needs permission to build source, deploy Cloud Run, use the runtime service account and configure public access. Cloud Build and Artifact Registry store/build the image. Organization policies may prohibit unauthenticated services.

These commands are instructions only; they have not been executed. Run from the extracted repository's `app/` directory:

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION="asia-southeast1"
export SERVICE="fleet-diagnostic"
gcloud auth login
gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy "$SERVICE" \
  --source . \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --allow-unauthenticated \
  --cpu 2 --memory 2Gi \
  --concurrency 1 \
  --min-instances 0 --max-instances 2 \
  --timeout 180 \
  --port 8080
```

Cloud Run uses this directory's Dockerfile. Use the generated HTTPS service URL; all frontend requests use relative `/api` paths, so no separate frontend host or CORS configuration is needed. This app has no sign-in: public access allows visitors to run inference. Instance limits reduce scaling but are not a hard spending cap; use a project budget alert and monitor usage.

Verify `/api/health` reports all four subsystems ready and `rail-pipeline-v3`, run `make container-smoke BASE_URL=https://YOUR_SERVICE_URL`, then check real uploads/downloads and direct reloads of `/rail`, `/door`, `/acv` and `/shm` in a browser. Record request latency, errors and memory usage before changing resources. Keep the previous Cloud Run revision available for traffic rollback if a later update fails.

## Runtime behavior

- The server listens on `0.0.0.0` and Cloud Run's `PORT` environment variable (8080 locally).
- The container accepts up to 30 MiB per file, below Cloud Run's 32 MiB HTTP/1 request limit including multipart overhead. `/api/health` advertises this limit to the UI. Do not increase it past the platform limit without changing the upload architecture.
- Inference is serialized inside each instance. Concurrency 1 avoids a long queue of CPU-bound predictions inside one process. Numerical libraries use one thread.
- Uploaded files use request-scoped temporary storage and are removed after processing. Cloud Run temporary writes consume instance memory; results remain in the browser until cleared or refreshed. There is no database or bucket dependency.
- Minimum instances 0 allows scale-to-zero; the first request can experience model-loading delay. Browser request timeout is 180 seconds.
- Do not bake service-account keys or other credentials into the image. The app needs no cloud API credentials for inference.

References: [Source deployments](https://docs.cloud.google.com/run/docs/deploying-source-code), [container contract](https://docs.cloud.google.com/run/docs/container-contract), [request limits](https://docs.cloud.google.com/run/quotas).
