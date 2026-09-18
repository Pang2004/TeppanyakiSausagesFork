"""Create the PS3 submission layout without copying raw datasets or environments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBSYSTEMS = {"door": "Door", "acv": "ACV", "rail": "Rail Corrugation", "shm": "SHM"}
EXCLUDED = {
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "test-results",
    "playwright-report",
    ".DS_Store",
    "exports",
    ".git",
    "UI_References",
}


def copy_tree(source: Path, destination: Path):
    def ignore(directory, names):
        return [
            name
            for name in names
            if name in EXCLUDED
            or name.endswith((".pyc", ".tsbuildinfo"))
            or name.startswith(".env")
        ]

    shutil.copytree(source, destination, ignore=ignore)


def door_timestamp(value: str):
    parts = [int(part) for part in value.split("-")]
    if len(parts) != 7:
        raise ValueError("Invalid Door timestamp")
    return datetime(*parts[:6], microsecond=parts[6] * 1000, tzinfo=timezone.utc)


def validate_predictions(path: Path, subsystem: str):
    columns = {
        "rail": ["file_id", "prediction"],
        "shm": ["file_id", "prediction"],
        "door": ["start_time", "end_time", "prediction"],
        "acv": ["file_id", "ranked_cars"],
    }[subsystem]
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != columns:
            raise ValueError(f"{path.name}: expected columns {columns}")
        rows = list(reader)
    if not rows or any(
        None in row or any(value is None or value == "" for value in row.values())
        for row in rows
    ):
        raise ValueError(f"{path.name}: empty or malformed predictions")
    if subsystem == "rail" and any(
        row["prediction"] not in ("Normal", "Side I", "Side II") for row in rows
    ):
        raise ValueError("Invalid Rail prediction label")
    if "file_id" in columns and len({row["file_id"] for row in rows}) != len(rows):
        raise ValueError(f"{path.name}: duplicate file IDs")

    if subsystem == "shm" and any(
        not math.isfinite(float(r["prediction"])) or float(r["prediction"]) < 0
        for r in rows
    ):
        raise ValueError("SHM predictions must be finite nonnegative numbers")
    if subsystem == "door":
        for row in rows:
            start = door_timestamp(row["start_time"])
            end = door_timestamp(row["end_time"])
            if end <= start or row["prediction"] not in (
                "Normal",
                "Abnormal resistance",
            ):
                raise ValueError("Invalid Door segment")
    if subsystem == "acv":
        for row in rows:
            cars = row["ranked_cars"].split("|")
            if not all(cars) or len(cars) != len(set(cars)):
                raise ValueError("Invalid ACV ranking")


def validate_coverage(predictions: Path):
    """Validate the official input inventory and browser-export evidence."""
    datasets = ROOT / "PS3/02_Datasets"
    folders = {"rail": "Rail_Corrugation/Test", "acv": "ACV/Test", "shm": "SHM/Test"}
    for key in SUBSYSTEMS:
        path = predictions / f"{key}_predictions.csv"
        validate_predictions(path, key)
        expected = (
            {"Test.csv"}
            if key == "door"
            else {
                p.name
                for p in (datasets / folders[key]).glob(
                    "*.xlsx" if key == "acv" else "*.csv"
                )
            }
        )
        if not expected:
            raise ValueError(f"No official inputs found for {key}")
        evidence = json.loads((predictions / f"{key}_provenance.json").read_text())
        if (
            evidence.get("source") != "browser submission download"
            or set(evidence["inputs"]) != expected
        ):
            raise ValueError(
                f"{key}: browser export provenance does not cover official inputs"
            )
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if key != "door" and {row["file_id"] for row in rows} != expected:
            raise ValueError(f"{key}: predictions do not cover official inputs")


def package(
    destination: Path,
    team_name: str,
    predictions: Path,
    video: Path | None,
    *,
    refresh: bool = False,
    require_complete: bool = False,
):
    if (
        not team_name
        or Path(team_name).name != team_name
        or team_name in (".", "..")
        or "\\" in team_name
    ):
        raise ValueError("Team name must be one folder name.")
    target = destination.resolve() / team_name
    if target.is_relative_to(ROOT / "app") or target.is_relative_to(
        ROOT / "Optional_Items"
    ):
        raise ValueError("Package output must be outside app/ and Optional_Items/.")
    if target.exists() and not refresh:
        raise FileExistsError(
            f"{target} already exists; choose a new --output directory."
        )
    if not (ROOT / "app/frontend/dist/index.html").is_file():
        raise ValueError(
            "Build the frontend first: npm run build --prefix app/frontend"
        )
    selected = []
    missing = []
    for key in SUBSYSTEMS:
        path = predictions / f"{key}_predictions.csv"
        if path.is_file():
            validate_predictions(path, key)
            selected.append(path)
        else:
            missing.append(path.name)
    if video and (
        not video.is_file()
        or video.suffix.lower() not in (".mp4", ".mov", ".webm", ".mkv")
    ):
        raise ValueError("--demo-video must name an existing video file.")
    if require_complete:
        if missing:
            raise ValueError("Missing predictions: " + ", ".join(missing))
        validate_coverage(predictions)
    if target.exists():
        archive_root = destination.resolve() / "archive"
        archive_root.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target.rename(archive_root / f"{team_name}-{stamp}")
        old_zip = destination.resolve() / f"{team_name}.zip"
        if old_zip.exists():
            old_zip.rename(archive_root / f"{team_name}-{stamp}.zip")
    target.mkdir(parents=True)
    copy_tree(ROOT / "app", target / "app")
    templates = ROOT / "scripts/submission_templates"
    for source, name in [("README.md", "README.md"), ("gitignore", ".gitignore")]:
        shutil.copy2(templates / source, target / name)
    copy_tree(ROOT / "Optional_Items", target / "Optional_Items")
    for key, name in SUBSYSTEMS.items():
        model_dir = target / "Optional_Items" / name / "model"
        model_dir.mkdir(exist_ok=True)
        shutil.copy2(ROOT / f"app/backend/artifacts/{key}_pipeline.joblib", model_dir)
    tools = target / "Optional_Items/tools"
    tools.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "scripts/model.py", tools)
    # The app is self-contained. Optional code imports its runtime from app/.
    shutil.copy2(
        ROOT / "Optional_Items/write_up.md", target / "Optional_Items/write_up.md"
    )
    if selected:
        with zipfile.ZipFile(
            target / "predictions.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for path in selected:
                archive.write(path, path.name)
    if video:
        shutil.copy2(video, target / f"demo_video{video.suffix.lower()}")
    report = [
        "# Packaging status",
        "",
        "The app includes Home and all four diagnostic tabs. Check the remaining submission items below.",
        "",
        "Included predictions: "
        + (", ".join(path.name for path in selected) or "none"),
        "Missing predictions: " + (", ".join(missing) or "none"),
        "Demo video: "
        + ("included; verify duration is at most 3 minutes" if video else "missing"),
        "",
        "All four interfaces are implemented. Raw PS3 data and example submission files are excluded.",
        "Input coverage: "
        + (
            "validated against supplied test inputs and browser provenance"
            if require_complete
            else "not checked; incremental package"
        ),
        "Container verification: not performed here; Docker is unavailable. Run app/Makefile container targets before deployment.",
        "Cloud deployment and GitHub publication: not performed.",
        "Prediction files must come from the app, cover the official test inputs, and be checked for completeness before final submission.",
    ]
    (target / "Optional_Items/packaging_status.md").write_text("\n".join(report) + "\n")
    provenance = target / "Optional_Items/export_provenance"
    provenance.mkdir(exist_ok=True)
    for path in predictions.glob("*_provenance.json"):
        shutil.copy2(path, provenance / path.name)
    entries = []
    for path in sorted(target.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append(f"{digest}  {path.relative_to(target).as_posix()}")
    (target / "MANIFEST.sha256").write_text("\n".join(entries) + "\n")
    with zipfile.ZipFile(
        destination.resolve() / f"{team_name}.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(destination.resolve()))
    print("\n".join(report))
    print(f"Packaged at {target}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "submission")
    parser.add_argument("--team-name", default="TeppanyakiSausages")
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "app/exports")
    parser.add_argument("--demo-video", type=Path)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Archive the previous package before replacing it",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require all four exports and official input coverage",
    )
    args = parser.parse_args()
    package(
        args.output,
        args.team_name,
        args.predictions_dir,
        args.demo_video,
        refresh=args.refresh,
        require_complete=args.require_complete,
    )


if __name__ == "__main__":
    main()
