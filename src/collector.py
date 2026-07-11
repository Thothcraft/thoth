#!/usr/bin/env python3
"""Continuously run synchronized one-minute Raspberry Pi captures."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import signal
from datetime import datetime, timedelta
from pathlib import Path


THOTH_ROOT = Path(os.environ.get("THOTH_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_CAPTURE_SCRIPT = Path(
    os.environ.get("THOTH_CAPTURE_SCRIPT", THOTH_ROOT / "src" / "backend" / "minute_collector.py")
)
DEFAULT_PYTHON = os.environ.get("THOTH_CAPTURE_PYTHON", sys.executable)
CAPTURE_SETTINGS_PATH = Path(
    os.environ.get("THOTH_CAPTURE_SETTINGS", THOTH_ROOT / "config" / "capture_settings.json")
)
PAUSE_PATH = Path(os.environ.get("THOTH_COLLECTOR_PAUSE", THOTH_ROOT / "config" / "collector.pause"))
active_capture: subprocess.Popen | None = None
shutdown_requested = False

sys.path.insert(0, str(THOTH_ROOT / "src"))
from backend.capture_manager import cleanup_old_minutes  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Run synchronized Thoth minute captures continuously.")
    parser.add_argument("--capture-script", default=str(DEFAULT_CAPTURE_SCRIPT))
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--keep-minutes", type=int, default=300)
    return parser.parse_args()


def next_minute_boundary(now: datetime | None = None) -> datetime:
    current_time = now or datetime.now().astimezone()
    current = current_time.replace(second=0, microsecond=0)
    if current_time.second == 0 and current_time.microsecond == 0:
        return current
    return current + timedelta(minutes=1)


def sleep_until(target: datetime) -> None:
    while True:
        remaining = target.timestamp() - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.25))


def requested_pause() -> bool:
    return PAUSE_PATH.exists()


def handle_shutdown(_signum, _frame) -> None:
    global shutdown_requested
    shutdown_requested = True
    if active_capture and active_capture.poll() is None:
        active_capture.terminate()


def load_capture_settings() -> dict:
    settings = {
        "labels": [item.strip() for item in os.environ.get("THOTH_MINUTE_LABELS", "").split(",") if item.strip()],
        "sensors": dict(DEFAULT_CAPTURE_SETTINGS["sensors"]),
    }
    try:
        loaded = json.loads(CAPTURE_SETTINGS_PATH.read_text(encoding="utf-8"))
        labels = loaded.get("labels")
        if isinstance(labels, str):
            settings["labels"] = [item.strip() for item in labels.split(",") if item.strip()]
        elif isinstance(labels, list):
            settings["labels"] = [str(item).strip() for item in labels if str(item).strip()]
        for key, value in (loaded.get("sensors") or {}).items():
            if key in SENSOR_FLAGS:
                settings["sensors"][key] = bool(value)
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"Unable to load capture settings: {exc}", file=sys.stderr)
    return settings


def run_capture(python: str, capture_script: str) -> int:
    global active_capture
    settings = load_capture_settings()
    command = [python, capture_script, "--start-now", "--duration", "59.5"]
    for label in settings["labels"]:
        command.extend(["--label", label])
    for sensor, flag in SENSOR_FLAGS.items():
        if settings["sensors"].get(sensor) is False:
            command.append(flag)
    print(f"Capture settings: labels={settings['labels'] or ['unlabeled']} sensors={settings['sensors']}", flush=True)
    active_capture = subprocess.Popen(command, start_new_session=True)
    while active_capture.poll() is None:
        if shutdown_requested or requested_pause():
            active_capture.terminate()
            try:
                active_capture.wait(timeout=10)
            except subprocess.TimeoutExpired:
                active_capture.kill()
            break
        time.sleep(0.25)
    result = active_capture.returncode or 0
    active_capture = None
    return result


def main() -> int:
    args = parse_args()
    capture_script = str(Path(args.capture_script).expanduser())
    if not Path(capture_script).exists():
        print(f"Capture script not found: {capture_script}", file=sys.stderr)
        return 1

    print(f"Thoth collector using {capture_script}")
    print(f"Keeping the latest {args.keep_minutes} minute folders")
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    # Start the first capture immediately so the dashboard shows activity as soon
    # as collection is enabled. Subsequent captures still advance on minute
    # boundaries to preserve the existing naming and retention behavior.
    target = datetime.now().astimezone()
    while not shutdown_requested:
        if requested_pause():
            time.sleep(0.5)
            target = next_minute_boundary()
            continue
        print(f"Waiting for {target.isoformat(timespec='seconds')}")
        sleep_until(target)
        if requested_pause() or shutdown_requested:
            continue
        print(f"Starting capture minute at {datetime.now().astimezone().isoformat(timespec='seconds')}")
        result = run_capture(args.python, capture_script)
        if result != 0:
            print(f"Capture exited with code {result}", file=sys.stderr)
        cleanup_old_minutes(args.keep_minutes)
        target += timedelta(minutes=1)
        if target.timestamp() + 60 < time.time():
            target = next_minute_boundary()


if __name__ == "__main__":
    raise SystemExit(main())
