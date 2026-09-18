PYTHON ?= python3
IMAGE ?= fleet-diagnostic:local
PORT ?= 8080
BASE_URL ?= http://127.0.0.1:$(PORT)
.PHONY: install build test serve container-build container-run container-smoke
install:
	$(PYTHON) -m pip install -r requirements.txt
	npm ci --prefix frontend
build:
	npm run build --prefix frontend
test:
	npm test --prefix frontend
serve:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAIL_MAX_FILE_BYTES=31457280 $(PYTHON) -m uvicorn backend.api:app --host 0.0.0.0 --port $(PORT) --workers 1
container-build:
	docker build -t $(IMAGE) .
container-run:
	docker run --rm --name fleet-diagnostic -p $(PORT):8080 --memory=2g --cpus=2 $(IMAGE)
container-smoke:
	$(PYTHON) scripts/smoke.py --url $(BASE_URL)
