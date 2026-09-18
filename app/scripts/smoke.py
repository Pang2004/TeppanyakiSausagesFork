"""Check a running app without datasets or third-party Python dependencies."""

import argparse
import json
from urllib.request import urlopen

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:8080")
args = parser.parse_args()
base = args.url.rstrip("/")
with urlopen(base + "/api/health", timeout=30) as response:
    health = json.load(response)
assert all(health["subsystems"].get(key) for key in ("rail", "door", "acv", "shm")), (
    health
)
assert health["model_version"] == "rail-pipeline-v3", health
assert health["maximum_file_bytes"] == 31457280, health
for route in ("/", "/rail", "/door", "/acv", "/shm"):
    with urlopen(base + route, timeout=30) as response:
        assert response.status == 200 and b"<html" in response.read().lower(), route
print("PASS: all routes, four model artifacts, Rail v3, and 30 MiB upload limit")
