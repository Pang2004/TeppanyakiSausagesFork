PYTHON ?= .venv/bin/python
APP_URL ?= http://127.0.0.1:8010
.PHONY: build test serve container-build container-run container-smoke predictions package
build:
	$(MAKE) -C app build
test:
	npm test --prefix app/frontend
	$(PYTHON) -m pytest tests/api tests/packaging -q
serve:
	$(MAKE) -C app serve PYTHON=$(abspath $(PYTHON))
container-build container-run container-smoke:
	$(MAKE) -C app $@ PYTHON=$(abspath $(PYTHON))
predictions: build
	cd app/frontend && APP_URL=$(APP_URL) npx playwright test e2e/submission.spec.ts --project=desktop-chromium
package: build
	$(PYTHON) scripts/package_submission.py --refresh --require-complete $(if $(VIDEO),--demo-video "$(VIDEO)",)
