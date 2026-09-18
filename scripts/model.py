"""Run a model's prediction, training or validation command from the repo root."""

import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "app/backend").is_dir()
)
SUBSYSTEMS = {"rail": "Rail Corrugation", "door": "Door", "acv": "ACV", "shm": "SHM"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subsystem", choices=SUBSYSTEMS)
    parser.add_argument("command", choices=("predict", "train", "validation"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "app"))
    sys.path.insert(
        0, str(ROOT / "Optional_Items" / SUBSYSTEMS[args.subsystem] / "code")
    )
    module = (
        f"backend.models.{args.subsystem}.predict"
        if args.command == "predict"
        else f"{args.subsystem}_dev.{args.command}"
    )
    if args.subsystem == "rail" and args.command == "train":
        module = "rail_dev.train_local"
    if args.subsystem == "acv" and args.command == "validation":
        parser.error("ACV validation is performed by its train command.")
    sys.argv = [module, *args.arguments]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
