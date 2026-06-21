#!/usr/bin/env python3
"""Continuous synchronized collector for Thoth RPi."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


THOTH_ROOT = Path(os.environ.get("THOTH_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_CAPTURE_SCRIPT = Path(os.environ.get("THOTH_CAPTURE_SCRIPT", str(THOTH_ROOT / "capture_dreamhat_minute.py")))
DEFAULT_PYTHON = os.environ.get("THOTH_CAPTURE_PYTHON", sys.executable)

sys.path.insert(0, str(THOTH_ROOT / "src"))

from backend.capture_manager import cleanup_old_minutes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synchronized Thoth minute capture continuously.")
    parser.add_argument("--capture-script", default=str(DEFAULT_CAPTURE_SCRIPT), help="Path to capture_dreamhat_minute.py")
    parser.add_argument("--python", default=DEFAULT_PYTHON, help="Python interpreter used to run the capture script")
    parser.add_argument("--keep-minutes", type=int, default=100, help="Number of minute folders to keep locally")
    return parser.parse_args()


def next_minute_boundary(now: datetime | None = None) -> datetime:
    current = (now or datetime.now().astimezone()).replace(second=0, microsecond=0)
    if now and now.second == 0 and now.microsecond == 0:
        return current
    return current + timedelta(minutes=1)


def sleep_until(target: datetime) -> None:
    while True:
        remaining = target.timestamp() - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.25))


def run_capture(python: str, capture_script: str) -> int:
    labels = [label.strip() for label in os.environ.get("THOTH_MINUTE_LABELS", "").split(",") if label.strip()]
    cmd = [
        python,
        capture_script,
        "--start-now",
        "--duration",
        "59.5",
    ]
    for label in labels:
        cmd.extend(["--label", label])
    proc = subprocess.run(cmd)
    return proc.returncode


def main() -> int:
    args = parse_args()
    capture_script = str(Path(args.capture_script).expanduser())
    if not Path(capture_script).exists():
        print(f"Capture script not found: {capture_script}", file=sys.stderr)
        return 1

    print(f"Thoth collector using {capture_script}")
    print(f"Keeping the latest {args.keep_minutes} minute folders")

    while True:
        target = next_minute_boundary()
        print(f"Waiting for {target.isoformat(timespec='seconds')}")
        sleep_until(target)

        started = datetime.now().astimezone().isoformat(timespec="seconds")
        print(f"Starting capture minute at {started}")
        rc = run_capture(args.python, capture_script)
        if rc != 0:
            print(f"Capture exited with code {rc}", file=sys.stderr)

        cleanup_old_minutes(args.keep_minutes)


if __name__ == "__main__":
    raise SystemExit(main())
