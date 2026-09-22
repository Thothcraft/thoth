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
# Finish physical acquisition before the next wall-clock boundary. This gives
# serial and GPIO drivers time to release their devices while CPU-heavy radar
# analysis from the completed minute may continue in the old worker.
CAPTURE_DURATION_SECONDS = 58.0
CAPTURE_SETTINGS_PATH = Path(
    os.environ.get("THOTH_CAPTURE_SETTINGS", THOTH_ROOT / "config" / "capture_settings.json")
)
PAUSE_PATH = Path(os.environ.get("THOTH_COLLECTOR_PAUSE", THOTH_ROOT / "config" / "collector.pause"))
# While the Sensor Lab page is open it heartbeats this file; the collector
# then pauses minute collection + model inference and runs a dedicated
# live-streaming child so the selected sensor gets the full device.
LIVE_SESSION_PATH = Path(os.environ.get("THOTH_LIVE_SESSION", THOTH_ROOT / "config" / "live_session.json"))
LIVE_SESSION_TTL_SECONDS = 15.0
# Each entry is (process, monotonic start time). A minute collector should
# finish within ~CAPTURE_DURATION_SECONDS; one that far overruns is stuck and
# must not be allowed to pile up — on a slow board, unbounded overlapping
# collectors saturate the CPU until the app and even SSH stop responding.
active_captures: list[tuple[subprocess.Popen, float]] = []
MAX_CONCURRENT_CAPTURES = 2
MAX_CAPTURE_AGE_SECONDS = 150.0
live_capture: subprocess.Popen | None = None
live_capture_started_at = 0.0
live_capture_sensor = ""
LIVE_RESTART_MIN_SECONDS = 5.0
shutdown_requested = False

sys.path.insert(0, str(THOTH_ROOT / "src"))
from backend.capture_manager import cleanup_old_minutes, disk_percent_used


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
    parser.add_argument("--max-disk-percent", type=float, default=95.0)
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


def live_session_active() -> bool:
    """True while the Sensor Lab heartbeat is fresh."""
    try:
        data = json.loads(LIVE_SESSION_PATH.read_text(encoding="utf-8"))
        return time.time() - float(data.get("ts") or 0) < LIVE_SESSION_TTL_SECONDS
    except Exception:
        return False


def handle_shutdown(_signum, _frame) -> None:
    global shutdown_requested
    shutdown_requested = True
    terminate_captures()
    terminate_live_capture()


def load_capture_settings() -> dict:
    settings = {
        "labels": [item.strip() for item in os.environ.get("THOTH_MINUTE_LABELS", "").split(",") if item.strip()],
        "chunk_seconds": DEFAULT_CAPTURE_SETTINGS["chunk_seconds"],
        "system_mode": DEFAULT_CAPTURE_SETTINGS["system_mode"],
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
    for capture, _started in list(active_captures):
        if capture.poll() is None:
            capture.terminate()
    deadline = time.monotonic() + 10
    for capture, _started in list(active_captures):
        if capture.poll() is not None:
            continue
        try:
            capture.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            capture.kill()
    reap_captures()


def terminate_live_capture() -> None:
    global live_capture
    if live_capture is None:
        return
    if live_capture.poll() is None:
        live_capture.terminate()
        try:
            live_capture.wait(timeout=10)
        except subprocess.TimeoutExpired:
            live_capture.kill()
    live_capture = None


# Live-session sensor names → capture-settings keys. The Sensor Lab page
# heartbeats which tab is open; live mode must stream that sensor even when
# minute collection has it disabled (live frames are transient and pruned).
LIVE_SENSOR_KEYS = {
    "camera": "usb_camera",
    "radar": "dreamhat_radar",
    "csi": "esp32_csi",
    "sensehat": "sense_hat",
}


def live_session_sensor() -> str:
    """The sensor tab the Sensor Lab page is currently viewing."""
    try:
        data = json.loads(LIVE_SESSION_PATH.read_text(encoding="utf-8"))
        return str(data.get("sensor") or "")
    except Exception:
        return ""


def start_live_capture(python: str, capture_script: str) -> subprocess.Popen:
    """Spawn the dedicated live-streaming worker (no minute collection)."""
    settings = load_capture_settings()
    wanted = LIVE_SENSOR_KEYS.get(live_session_sensor())
    command = [python, capture_script, "--live-only"]
    for sensor, flag in SENSOR_FLAGS.items():
        if settings["sensors"].get(sensor) is False and sensor != wanted:
            command.append(flag)
    print("Live session active: streaming sensors at full rate (collection paused)", flush=True)
    return subprocess.Popen(command, start_new_session=True)


def reap_captures() -> None:
    for entry in list(active_captures):
        capture, _started = entry
        result = capture.poll()
        if result is None:
            continue
        active_captures.remove(entry)
        if result != 0:
            print(f"Capture exited with code {result}", file=sys.stderr)


def kill_overrunning_captures() -> None:
    """Terminate minute collectors that have far overrun their minute.

    A stuck collector would otherwise run forever while new ones keep spawning,
    saturating the CPU. Anything older than MAX_CAPTURE_AGE_SECONDS is killed.
    """
    now = time.monotonic()
    for capture, started in list(active_captures):
        if capture.poll() is not None:
            continue
        if now - started > MAX_CAPTURE_AGE_SECONDS:
            print(
                f"Capture overran {MAX_CAPTURE_AGE_SECONDS:.0f}s - terminating stuck collector",
                file=sys.stderr, flush=True,
            )
            capture.terminate()


def start_capture(python: str, capture_script: str, target: datetime) -> subprocess.Popen:
    settings = load_capture_settings()
    command = [
        python, capture_script, "--duration", str(CAPTURE_DURATION_SECONDS),
        "--chunk-seconds", str(settings["chunk_seconds"]),
        "--scheduled-start", target.isoformat(),
    ]
    # Explicit ports may be comma-separated.  With no override the minute
    # collector performs its own ESP32-aware multi-device discovery.
    preferred_csi = [
        item.strip()
        for item in os.environ.get("THOTH_CSI_PORT", "").split(",")
        if item.strip()
    ]
    for port in preferred_csi:
        command.extend(["--csi-port", port])
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
    active_captures.append((capture, time.monotonic()))
    return capture


def main() -> int:
    global live_capture, live_capture_started_at, live_capture_sensor
    args = parse_args()
    capture_script = str(Path(args.capture_script).expanduser())
    if not Path(capture_script).exists():
        print(f"Capture script not found: {capture_script}", file=sys.stderr)
        return 1

    print(f"Thoth collector using {capture_script}")
    print(f"Capture retention uses up to {args.max_disk_percent:.1f}% of disk space")
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    # Always start on an actual wall-clock minute boundary. Starting a full
    # minute immediately after process boot permanently offsets every folder.
    target = next_minute_boundary()
    if target.timestamp() - time.time() < PREPARE_LEAD_SECONDS:
        target += timedelta(minutes=1)
    while not shutdown_requested:
        if live_session_active():
            # Sensor Lab open: stop minute collection + models, stream sensors.
            terminate_captures()
            wanted_sensor = live_session_sensor()
            if live_capture is not None and live_capture.poll() is None \
                    and wanted_sensor != live_capture_sensor \
                    and time.monotonic() - live_capture_started_at >= LIVE_RESTART_MIN_SECONDS:
                # Tab switched — restart so the newly viewed sensor is
                # streamed even if capture settings disable it.
                terminate_live_capture()
            if live_capture is None or live_capture.poll() is not None:
                if time.monotonic() - live_capture_started_at >= LIVE_RESTART_MIN_SECONDS:
                    terminate_live_capture()
                    live_capture = start_live_capture(args.python, capture_script)
                    live_capture_started_at = time.monotonic()
                    live_capture_sensor = wanted_sensor
            time.sleep(0.5)
            target = next_minute_boundary()
            if target.timestamp() - time.time() < PREPARE_LEAD_SECONDS:
                target += timedelta(minutes=1)
            continue
        if live_capture is not None:
            # Session ended — resume normal minute collection.
            print("Live session ended: resuming minute collection", flush=True)
            terminate_live_capture()
            live_capture_sensor = ""
            target = next_minute_boundary()
            if target.timestamp() - time.time() < PREPARE_LEAD_SECONDS:
                target += timedelta(minutes=1)
            continue
        if requested_pause():
            terminate_captures()
            time.sleep(0.5)
            target = next_minute_boundary()
            if target.timestamp() - time.time() < PREPARE_LEAD_SECONDS:
                target += timedelta(minutes=1)
            continue
        prepare_at = target - timedelta(seconds=PREPARE_LEAD_SECONDS)
        print(f"Preparing capture for {target.isoformat(timespec='seconds')}")
        while not shutdown_requested and not requested_pause() and not live_session_active():
            reap_captures()
            remaining = prepare_at.timestamp() - time.time()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.25))
        if requested_pause() or shutdown_requested or live_session_active():
            continue
        # The child imports and configures during the lead time, then sleeps on
        # its explicit target. At most two children coexist for a few seconds:
        # one finishing the prior minute and one waiting for the next boundary.
        reap_captures()
        kill_overrunning_captures()
        cleanup_old_minutes(max_disk_percent=args.max_disk_percent)
        if disk_percent_used() >= args.max_disk_percent:
            # Stop collection rather than fill the card. The cleanup above only
            # reclaims capture folders; if the disk is still over the limit the
            # pressure is from non-capture data and starting a new minute would
            # just write into an already-full filesystem.
            print(
                f"Disk usage >= {args.max_disk_percent:.1f}% - skipping capture for "
                f"{target.isoformat(timespec='seconds')} (free space or raise --max-disk-percent)",
                flush=True,
            )
        elif len(active_captures) >= MAX_CONCURRENT_CAPTURES:
            # A previous minute is still running. Spawning another would pile up
            # collectors and saturate a slow board — skip this minute instead.
            print(
                f"Skipping capture for {target.isoformat(timespec='seconds')}: "
                f"{len(active_captures)} collector(s) still running",
                flush=True,
            )
        else:
            start_capture(args.python, capture_script, target)
        target += timedelta(minutes=1)
        if target.timestamp() + 5 < time.time():
            target = next_minute_boundary()
            if target.timestamp() - time.time() < PREPARE_LEAD_SECONDS:
                target += timedelta(minutes=1)

    terminate_captures()
    terminate_live_capture()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
