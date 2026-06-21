#!/usr/bin/env python3
"""CLI wrapper for Thoth synchronized minute capture."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backend.minute_collector import main


if __name__ == "__main__":
    raise SystemExit(main())
