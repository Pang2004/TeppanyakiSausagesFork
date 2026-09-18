# Deployment and submission preparation verification

Verified on 2026-09-19. No Google Cloud resources or GitHub repository were created.

- Production TypeScript/Vite build passed.
- 10 frontend unit tests passed, including the official single-stream Door export.
- 15 API tests and 4 packaging tests passed. Packaging tests cover exclusion rules, manifest/archive creation, refresh backups, missing predictions and invalid output values.
- Browser-generated submission downloads passed for all supplied inputs: Rail 68 recordings, Door one stream (38 predicted operations), ACV one workbook, SHM 16 recordings.
- A server started from the packaged `app/` directory passed readiness, direct-route and 30 MiB upload-limit smoke checks. It used the existing Python environment for installed dependencies, but imported the packaged runtime and packaged artifacts.
- Four browser checks against that packaged server passed real inference and CSV downloads across Rail, Door, ACV and SHM.
- Archive inspection verified the single team root, four flat prediction ZIP entries, current artifact copies, file hashes, and exclusion of datasets, environments, credentials, deleted reference directories and browser outputs.
- Makefile container and packaging targets were inspected with dry runs.

Docker is not installed in this environment. The Docker image has **not** been built or run, and a fresh installation of the pinned Python dependencies has not been tested here. Run the container build/run/smoke targets before cloud deployment; native checks do not establish container compatibility or a measured cloud memory budget. No Safari verification is claimed.

The demo video (at most three minutes) is missing. The package is prepared for extraction and a new GitHub repository, but is not a complete competition submission until that video is supplied. Regenerate predictions if organizers supply replacement test inputs.
