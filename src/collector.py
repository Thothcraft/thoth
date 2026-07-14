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
PREPARE_LEAD_SECONDS = 5.0
CAPTURE_SETTINGS_PATH = Path(
    os.environ.get("THOTH_CAPTURE_SETTINGS", THOTH_ROOT / "config" / "capture_settings.json")
)
PAUSE_PATH = Path(os.environ.get("THOTH_COLLECTOR_PAUSE", THOTH_ROOT / "config" / "collector.pause"))
active_captures: list[subprocess.Popen] = []
shutdown_requested = False

sys.path.insert(0, str(THOTH_ROOT / "src"))


DEFAULT_CAPTURE_SETTINGS = {
    "labels": [],
    "chunk_seconds": 10.0,
    "system_mode": "balanced",
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
    terminate_captures()


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
        settings["chunk_seconds"] = min(30.0, max(2.0, float(loaded.get("chunk_seconds", 10.0))))
        mode = str(loaded.get("system_mode", "balanced")).strip().lower()
        settings["system_mode"] = mode if mode in {"responsive", "balanced", "precision"} else "balanced"
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"Unable to load capture settings: {exc}", file=sys.stderr)
    return settings


def terminate_captures() -> None:
    for capture in list(active_captures):
        if capture.poll() is None:
            capture.terminate()
    deadline = time.monotonic() + 10
    for capture in list(active_captures):
        if capture.poll() is not None:
            continue
        try:
            capture.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            capture.kill()
    reap_captures()


def reap_captures() -> None:
    for capture in list(active_captures):
        result = capture.poll()
        if result is None:
            continue
        active_captures.remove(capture)
        if result != 0:
            print(f"Capture exited with code {result}", file=sys.stderr)


def start_capture(python: str, capture_script: str, target: datetime) -> subprocess.Popen:
    settings = load_capture_settings()
    command = [
        python, capture_script, "--duration", "59.5",
        "--chunk-seconds", str(settings["chunk_seconds"]),
        "--scheduled-start", target.isoformat(),
    ]
    preferred_csi = os.environ.get("THOTH_CSI_PORT")
    if not preferred_csi:
        preferred_csi = next((
            str(path) for pattern in ("ttyACM*", "ttyUSB*")
            for path in sorted(Path("/dev").glob(pattern))
        ), None)
    if preferred_csi:
        command.extend(["--csi-port", preferred_csi])
    for label in settings["labels"]:
        command.extend(["--label", label])
    for sensor, flag in SENSOR_FLAGS.items():
        if settings["sensors"].get(sensor) is False:
            command.append(flag)
    print(
        f"Capture settings for {target.isoformat(timespec='seconds')}: "
        f"labels={settings['labels'] or ['unlabeled']} sensors={settings['sensors']}",
        flush=True,
    )
    capture = subprocess.Popen(command, start_new_session=True)
    active_captures.append(capture)
    return capture


def main() -> int:
    args = parse_args()
    capture_script = str(Path(args.capture_script).expanduser())
    if not Path(capture_script).exists():
        print(f"Capture script not found: {capture_script}", file=sys.stderr)
        return 1

    print(f"Thoth collector using {capture_script}")
    print(f"Retention limit is {args.keep_minutes} minute folders (enforced by the dashboard cleanup job)")
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    # Always start on an actual wall-clock minute boundary. Starting a full
    # minute immediately after process boot permanently offsets every folder.
    target = next_minute_boundary()
    if target.timestamp() - time.time() < PREPARE_LEAD_SECONDS:
        target += timedelta(minutes=1)
    while not shutdown_requested:
        if requested_pause():
            terminate_captures()
            time.sleep(0.5)
            target = next_minute_boundary()
            if target.timestamp() - time.time() < PREPARE_LEAD_SECONDS:
                target += timedelta(minutes=1)
            continue
        prepare_at = target - timedelta(seconds=PREPARE_LEAD_SECONDS)
        print(f"Preparing capture for {target.isoformat(timespec='seconds')}")
        while not shutdown_requested and not requested_pause():
            reap_captures()
            remaining = prepare_at.timestamp() - time.time()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.25))
        if requested_pause() or shutdown_requested:
            continue
        # The child imports and configures during the lead time, then sleeps on
        # its explicit target. At most two children coexist for a few seconds:
        # one finishing the prior minute and one waiting for the next boundary.
        reap_captures()
        start_capture(args.python, capture_script, target)
        target += timedelta(minutes=1)
        if target.timestamp() + 5 < time.time():
            target = next_minute_boundary()
            if target.timestamp() - time.time() < PREPARE_LEAD_SECONDS:
                target += timedelta(minutes=1)

    terminate_captures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
