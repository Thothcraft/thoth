#!/usr/bin/env python3
"""Continuous synchronized collector for Thoth RPi."""

from __future__ import annotations

import argparse
import json
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


CAPTURE_SETTINGS_PATH = Path(os.environ.get("THOTH_CAPTURE_SETTINGS", str(THOTH_ROOT / "data" / "config" / "capture_settings.json")))
DEFAULT_CAPTURE_SETTINGS = {
    "labels": [],
    "sensors": {
        "usb_camera": True,
        "dreamhat_radar": True,
        "esp32_csi": True,
        "sense_hat": True,
    },
}
SENSOR_FLAGS = {
    "usb_camera": "--no-camera",
    "dreamhat_radar": "--no-radar",
    "esp32_csi": "--no-csi",
    "sense_hat": "--no-sensehat",
}


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


def load_capture_settings() -> dict:
    settings = {
        "labels": [label.strip() for label in os.environ.get("THOTH_MINUTE_LABELS", "").split(",") if label.strip()],
        "sensors": dict(DEFAULT_CAPTURE_SETTINGS["sensors"]),
    }
    try:
        if CAPTURE_SETTINGS_PATH.exists():
            loaded = json.loads(CAPTURE_SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                labels = loaded.get("labels")
                if isinstance(labels, str):
                    settings["labels"] = [item.strip() for item in labels.split(",") if item.strip()]
                elif isinstance(labels, list):
                    settings["labels"] = [str(item).strip() for item in labels if str(item).strip()]
                sensor_updates = loaded.get("sensors")
                if isinstance(sensor_updates, dict):
                    for key in SENSOR_FLAGS:
                        if key in sensor_updates:
                            settings["sensors"][key] = bool(sensor_updates[key])
    except Exception as exc:
        print(f"Unable to load capture settings from {CAPTURE_SETTINGS_PATH}: {exc}", file=sys.stderr)
    return settings


def run_capture(python: str, capture_script: str) -> int:
    settings = load_capture_settings()
    cmd = [
        python,
        capture_script,
        "--start-now",
        "--duration",
        "59.5",
    ]
    for label in settings["labels"]:
        cmd.extend(["--label", label])
    for sensor, flag in SENSOR_FLAGS.items():
        if settings["sensors"].get(sensor) is False:
            cmd.append(flag)
    print(
        "Capture settings: labels=%s sensors=%s"
        % (settings["labels"] or ["unlabeled"], settings["sensors"]),
        flush=True,
    )
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

    target = next_minute_boundary()
    while True:
        print(f"Waiting for {target.isoformat(timespec='seconds')}")
        sleep_until(target)

        started = datetime.now().astimezone().isoformat(timespec="seconds")
        print(f"Starting capture minute at {started}")
        rc = run_capture(args.python, capture_script)
        if rc != 0:
            print(f"Capture exited with code {rc}", file=sys.stderr)

        cleanup_old_minutes(args.keep_minutes)
        target = target + timedelta(minutes=1)
        if target.timestamp() + 60 < time.time():
            target = next_minute_boundary()


if __name__ == "__main__":
    raise SystemExit(main())
