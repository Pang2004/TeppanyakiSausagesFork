"""Manifest-validated development cache for runtime local spectra."""

import hashlib
import json
from pathlib import Path

import pandas as pd
from backend.models.rail import local
from backend.models.rail.local import (
    CONFIG,
    extract_local,
    integrate_band,
    spectral_rows,
)
from joblib import Parallel, delayed

__all__ = ["CONFIG", "extract_local", "integrate_band", "load_local", "spectral_rows"]


def load_local(paths, groups, cache, jobs=4):
    manifest = json.loads(
        json.dumps(
            {
                "config": CONFIG,
                "extractor_sha256": hashlib.sha256(
                    Path(local.__file__).read_bytes()
                ).hexdigest(),
                "recordings": dict(
                    zip([p.name for p in paths], groups.tolist(), strict=True)
                ),
            }
        )
    )
    manifest_path = cache.with_suffix(".manifest.json")
    if (
        cache.exists()
        and manifest_path.exists()
        and json.loads(manifest_path.read_text()) == manifest
    ):
        frame = pd.read_csv(cache)
        if frame.file_id.tolist() == [p.name for p in paths]:
            return frame
    rows = Parallel(n_jobs=jobs, prefer="threads", verbose=5)(
        delayed(extract_local)(p) for p in paths
    )
    frame = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return frame
