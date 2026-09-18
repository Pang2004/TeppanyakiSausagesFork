"""Make the optional development packages available to repository tests."""

import sys
from pathlib import Path

for code_dir in (Path(__file__).parent / "Optional_Items").glob("*/code"):
    sys.path.insert(0, str(code_dir))
