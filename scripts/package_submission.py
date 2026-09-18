"""Create the PS3 submission layout without copying raw datasets or environments."""

from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
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


def package(destination: Path, team_name: str, predictions: Path, video: Path | None):
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
    if target.exists():
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
    target.mkdir(parents=True)
    copy_tree(ROOT / "app", target / "app")
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
        "Door, ACV and SHM interfaces remain to be implemented. Raw PS3 data and example submission files are excluded.",
        "Prediction files must come from the app, cover the official test inputs, and be checked for completeness before final submission.",
    ]
    (target / "Optional_Items/packaging_status.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"Packaged at {target}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "submission")
    parser.add_argument("--team-name", default="TeppanyakiSausages")
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "app/exports")
    parser.add_argument("--demo-video", type=Path)
    args = parser.parse_args()
    package(args.output, args.team_name, args.predictions_dir, args.demo_video)


if __name__ == "__main__":
    main()
