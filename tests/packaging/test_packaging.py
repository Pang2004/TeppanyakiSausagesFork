"""Submission layout and exclusion rules without copying real training data."""

import importlib.util
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "packager", Path(__file__).resolve().parents[2] / "scripts/package_submission.py"
)
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


def test_package_excludes_datasets_and_env_and_exports_flat_csv(tmp_path, monkeypatch):
    source = tmp_path / "repo"
    for folder in [
        "app/backend/artifacts",
        "app/frontend/dist",
        "app/frontend/node_modules",
        "app/.venv",
        "app/exports",
        "PS3/02_Datasets",
        "scripts",
        "Optional_Items",
    ]:
        (source / folder).mkdir(parents=True, exist_ok=True)
    (source / "app/frontend/dist/index.html").write_text("built UI")
    (source / "app/frontend/node_modules/ignored").write_text("do not package")
    (source / "app/.env").write_text("secret")
    (source / "PS3/02_Datasets/raw.csv").write_text("raw signals")
    (source / "scripts/model.py").write_text("launcher")
    (source / "Optional_Items/write_up.md").write_text("methodology")
    for key, label in packager.SUBSYSTEMS.items():
        (source / "Optional_Items" / label / "code").mkdir(parents=True)
        (source / f"app/backend/artifacts/{key}_pipeline.joblib").write_bytes(b"model")
    predictions = source / "app/exports"
    (predictions / "rail_predictions.csv").write_text(
        "file_id,prediction\nTest1.csv,Normal\n"
    )
    monkeypatch.setattr(packager, "ROOT", source)
    target = packager.package(tmp_path / "output", "Team", predictions, None)
    assert not (target / "PS3").exists()
    assert not (target / "app/.env").exists()
    assert not (target / "app/frontend/node_modules").exists()
    assert not (target / "app/.venv").exists()
    assert not (target / "app/exports").exists()
    assert (target / "app/frontend/dist/index.html").is_file()
    for label in packager.SUBSYSTEMS.values():
        assert (target / "Optional_Items" / label / "model").is_dir()
    with zipfile.ZipFile(target / "predictions.zip") as archive:
        assert archive.namelist() == ["rail_predictions.csv"]
    report = (target / "Optional_Items/packaging_status.md").read_text()
    assert "Demo video: missing" in report
    assert "door_predictions.csv" in report
    assert not list(target.glob("demo_video.*"))
    with pytest.raises(FileExistsError):
        packager.package(tmp_path / "output", "Team", predictions, None)
    with pytest.raises(ValueError, match="outside"):
        packager.package(source / "app", "Nested", predictions, None)
