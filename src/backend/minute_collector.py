#!/usr/bin/env python3
"""Capture one synchronized minute of Thoth sensor data."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.config import Config  # type: ignore
    from backend.radar_analysis import (  # type: ignore
        PersistentTargetIdentity,
        StreamingChunkAnalyzer,
        compile_minute_xy_payload,
        create_signal_processor,
        load_room_config,
        occupancy_label,
    )
else:
    from .config import Config
    from .radar_analysis import (
        PersistentTargetIdentity,
        StreamingChunkAnalyzer,
        compile_minute_xy_payload,
        create_signal_processor,
        load_room_config,
        occupancy_label,
    )

THOTH_ROOT = Path(__file__).resolve().parents[2]
MMW_RELEASE = THOTH_ROOT / "WS" / "MMW-HAT" / "MMW-HAT-Release"
RADAR_CFG = MMW_RELEASE / "radar_config" / "config_3rx_3m"
DATA_ROOT = Path(Config.CAPTURE_DATA_DIR).expanduser()
CAPTURE_SETTINGS_PATH = Path(Config.CONFIG_DIR).expanduser() / "capture_settings.json"
CSI_HEADER = "type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data"
RADAR_FRAMES_PER_CHUNK = 10
# A minute contains at most about sixty 10-frame chunks. The dedicated live
# worker now owns freshness, so archival jobs can be buffered for the whole
# minute instead of discarding a valid saved chunk during a transient CPU spike.
MAX_PENDING_ANALYSIS_CHUNKS = 64
LIVE_VISUALIZATION_INTERVAL_SECONDS = 0.16

sys.path.insert(0, str(MMW_RELEASE))

MAX_PENDING_ANALYSIS_FRAMES_PER_CHUNK = 1


def enqueue_latest_chunk_frame(
    analysis_queue: queue.Queue[Any],
    entry: dict[str, Any],
    frame: bytes,
    captured_at: float,
    *metadata: Any,
) -> bool:
    """Queue a frame while retaining the newest pending samples for its chunk.

    Signal processing must never back-pressure radar capture. When processing
    falls behind, replacing the oldest still-pending frame from this chunk
    preserves current motion instead of keeping stale empty-room frames until
    the following minute. Start/end control messages are never removed.

    Returns True when an older pending frame was replaced.
    """
    replaced = False
    with analysis_queue.mutex:
        matching_indexes = [
            index
            for index, item in enumerate(analysis_queue.queue)
            if isinstance(item, tuple)
            and len(item) >= 2
            and item[0] == "frame"
            and item[1] is entry
        ]
        if len(matching_indexes) >= MAX_PENDING_ANALYSIS_FRAMES_PER_CHUNK:
            del analysis_queue.queue[matching_indexes[0]]
            analysis_queue.unfinished_tasks = max(0, analysis_queue.unfinished_tasks - 1)
            analysis_queue.not_full.notify()
            replaced = True
    analysis_queue.put_nowait(("frame", entry, frame, captured_at, *metadata))
    return replaced


def enqueue_analysis_chunk(
    analysis_queue: queue.Queue[Any],
    job: tuple[Any, ...],
) -> list[dict[str, Any]]:
    """Queue one exact 10-frame archival job.

    The production queue holds a complete minute, so transient CPU pressure
    cannot remove captured chunk statistics. The replacement branch remains a
    final memory bound for callers that intentionally provide a smaller queue.
    """
    if len(job) < 4 or job[0] != "chunk" or len(job[3]) != RADAR_FRAMES_PER_CHUNK:
        raise ValueError(
            f"radar analysis chunks require exactly {RADAR_FRAMES_PER_CHUNK} frames"
        )

    dropped: list[dict[str, Any]] = []
    while True:
        try:
            analysis_queue.put_nowait(job)
            return dropped
        except queue.Full:
            try:
                stale = analysis_queue.get_nowait()
            except queue.Empty:
                continue
            try:
                if (
                    isinstance(stale, tuple)
                    and len(stale) >= 2
                    and stale[0] == "chunk"
                    and isinstance(stale[1], dict)
                ):
                    dropped.append(stale[1])
            finally:
                analysis_queue.task_done()


def live_chunk_statistics(analyzer: StreamingChunkAnalyzer) -> dict[str, Any]:
    """Build the partial chunk result published while analysis is running."""
    evaluated = analyzer.evaluated_frames
    detected = analyzer.detected_frames
    label = occupancy_label(
        detected, evaluated, (100.0 / evaluated) if evaluated else 100.0,
    )
    classification = "green" if label == "occupied" else "red"
    return {
        "status": "collecting",
        "detected_frames": detected,
        "evaluated_frames": evaluated,
        "ratio": detected / evaluated if evaluated else 0.0,
        "classification": classification,
        "occupied": label == "occupied",
        "location": list(analyzer.last_position),
        "score": analyzer.last_score,
        "people_count": len(analyzer.last_targets),
        "targets": analyzer.last_targets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one synchronized Thoth sensor minute into the configured data folder."
    )
    parser.add_argument("--duration", type=float, default=59.5, help="Capture duration in seconds.")
    parser.add_argument("--camera", default=None, help="USB camera device, for example /dev/video0.")
    parser.add_argument("--csi-port", default="auto", help="ESP32 CSI receiver serial port, or 'auto'.")
    parser.add_argument("--csi-baud", type=int, default=115200, help="ESP32 CSI receiver baud rate.")
    parser.add_argument(
        "--csi-detect-seconds",
        type=float,
        default=0.0,
        help="Optional seconds to probe serial ports for CSI_DATA in auto mode. Default avoids pre-opening the ESP32.",
    )
    parser.add_argument("--no-camera", action="store_true", help="Skip USB camera capture.")
    parser.add_argument("--no-radar", action="store_true", help="Skip mmWave radar capture.")
    parser.add_argument("--no-csi", action="store_true", help="Skip ESP32 CSI serial capture.")
    parser.add_argument("--no-sensehat", action="store_true", help="Skip Sense HAT capture.")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=10.0,
        help="Radar capture and analysis chunk size in seconds.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Preset one or more labels for the captured minute. Repeat the flag to add multiple labels.",
    )
    parser.add_argument(
        "--start-now",
        action="store_true",
        help="Start immediately and name the folder from the current real-clock minute.",
    )
    parser.add_argument(
        "--scheduled-start",
        default=None,
        help="Exact ISO-8601 wall-clock minute assigned by the continuous supervisor.",
    )
    return parser.parse_args()


def normalize_labels(labels: object) -> list[str]:
    if isinstance(labels, str):
        items = labels.split(",")
    elif isinstance(labels, list):
        items = labels
    else:
        items = []

    cleaned: list[str] = []
    for item in items:
        label = str(item or "").strip().replace("/", "_").replace("\\", "_")
        label = " ".join(label.split())
        if label and label not in cleaned:
            cleaned.append(label)
    return cleaned


def output_dir_for_minute(folder_name: str, labels: list[str]) -> Path:
    root = DATA_ROOT
    if labels:
        return root / labels[0] / folder_name
    return root / folder_name


def minute_start(start_now: bool, scheduled_start: str | None = None) -> dt.datetime:
    if scheduled_start:
        scheduled = dt.datetime.fromisoformat(scheduled_start)
        if scheduled.tzinfo is None:
            scheduled = scheduled.replace(tzinfo=dt.datetime.now().astimezone().tzinfo)
        return scheduled
    now = dt.datetime.now().astimezone()
    current_minute = now.replace(second=0, microsecond=0)
    if start_now:
        return current_minute
    if now.second == 0 and now.microsecond < 250_000:
        return current_minute
    return current_minute + dt.timedelta(minutes=1)


def sleep_until(target: dt.datetime) -> None:
    while True:
        remaining = target.timestamp() - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.25))


def iso_now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with open(temporary, "w", encoding="utf-8") as fd:
        json.dump(payload, fd, indent=2)
    temporary.replace(path)


def load_processing_settings() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "labels": [],
        "radar_detection_threshold_db": 8.0,
        "auto_occupancy_label_enabled": True,
        "system_mode": "balanced",
        "prediction_label_style": "occupancy",
        "people_count_label_enabled": False,
        "sleep_study_enabled": False,
        "revision": 0,
        "updated_at": None,
    }
    loaded: dict[str, Any] = {}
    try:
        parsed = json.loads(CAPTURE_SETTINGS_PATH.read_text(encoding="utf-8"))
        loaded = parsed if isinstance(parsed, dict) else {}
        defaults.update({key: loaded[key] for key in defaults if key in loaded})
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"Unable to load processing settings: {exc}", file=sys.stderr)
    if "radar_detection_threshold_db" not in loaded:
        legacy_normalized = loaded.get("radar_detection_threshold_normalized")
        if legacy_normalized is not None:
            defaults["radar_detection_threshold_db"] = float(legacy_normalized) * 10.0
    defaults["radar_detection_threshold_db"] = min(
        30.0, max(0.0, float(defaults["radar_detection_threshold_db"]))
    )
    defaults["auto_occupancy_label_enabled"] = bool(defaults["auto_occupancy_label_enabled"])
    mode = str(defaults.get("system_mode") or "balanced").strip().lower()
    defaults["system_mode"] = mode if mode in {"responsive", "balanced", "precision"} else "balanced"
    style = str(defaults.get("prediction_label_style") or "occupancy").strip().lower()
    defaults["prediction_label_style"] = style if style in {"occupancy", "presence"} else "occupancy"
    defaults["people_count_label_enabled"] = bool(defaults.get("people_count_label_enabled"))
    defaults["sleep_study_enabled"] = bool(defaults.get("sleep_study_enabled"))
    defaults["labels"] = normalize_labels(defaults.get("labels"))
    return defaults


def prediction_label_for(label: str, style: str) -> str:
    if style == "presence":
        return "present" if label == "occupied" else "absent"
    return "occupied" if label == "occupied" else "empty"


def annotate_chunk_result(
    result: dict[str, Any], settings: dict[str, Any], room: dict[str, Any],
    preset_labels: list[str], minute: str, expected_chunks: int,
    previous_frames: int,
) -> dict[str, Any]:
    occupancy = result.get("occupancy") or {}
    raw_label = str(occupancy.get("label") or "empty")
    classification = str(occupancy.get("classification") or ("green" if raw_label == "occupied" else "red"))
    targets = result.get("targets") if isinstance(result.get("targets"), list) else []
    frames = result.get("frames") if isinstance(result.get("frames"), list) else []
    evaluated_frames = int(occupancy.get("evaluated_frames") or len(frames))
    dwell_threshold = min(100.0, max(0.0, float(occupancy.get("threshold_percent") or 50.0)))
    target_stats: dict[int, dict[str, Any]] = {}
    people_count = 0

    def zones_at(position: object) -> list[str]:
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            return []
        tx, ty = float(position[0]), float(position[1])
        matched: list[str] = []
        for zone in room.get("zones") or []:
            if not isinstance(zone, dict):
                continue
            x, y = float(zone.get("x") or 0), float(zone.get("y") or 0)
            width, depth = float(zone.get("width") or 1), float(zone.get("depth") or 1)
            if x <= tx <= x + width and y <= ty <= y + depth:
                label = str(zone.get("label") or zone.get("id") or "zone").strip()
                if label and label not in matched:
                    matched.append(label)
        return matched

    for frame in frames:
        frame_targets = frame.get("targets") if isinstance(frame, dict) and isinstance(frame.get("targets"), list) else []
        people_count = max(people_count, len(frame_targets))
        for target in frame_targets:
            if not isinstance(target, dict):
                continue
            target_id = int(target.get("id") or 0)
            stats = target_stats.setdefault(target_id, {"target_id": target_id, "present_frames": 0, "zone_frames": {}})
            stats["present_frames"] += 1
            for zone_label in zones_at(target.get("position")):
                stats["zone_frames"][zone_label] = int(stats["zone_frames"].get(zone_label) or 0) + 1

    if not frames and targets:
        people_count = len(targets)
        for target in targets:
            if not isinstance(target, dict):
                continue
            target_id = int(target.get("id") or 0)
            stats = target_stats.setdefault(target_id, {"target_id": target_id, "present_frames": evaluated_frames, "zone_frames": {}})
            for zone_label in zones_at(target.get("position")):
                stats["zone_frames"][zone_label] = evaluated_frames

    occupied_zones: list[str] = []
    activity_targets: list[dict[str, Any]] = []
    for target_id, stats in target_stats.items():
        qualified = [
            zone_label for zone_label, count in stats["zone_frames"].items()
            if evaluated_frames > 0 and int(count) * 100.0 >= dwell_threshold * evaluated_frames
        ]
        for zone_label in qualified:
            if zone_label not in occupied_zones:
                occupied_zones.append(zone_label)
        activity_targets.append({
            "target_id": target_id,
            "present_frames": int(stats["present_frames"]),
            "evaluated_frames": evaluated_frames,
            "zone_frames": dict(stats["zone_frames"]),
            "zones": qualified,
        })
    for target in targets:
        if isinstance(target, dict):
            activity = next((item for item in activity_targets if item["target_id"] == int(target.get("id") or 0)), None)
            target["zones"] = list((activity or {}).get("zones") or [])

    activity_labels = ["present", "occupied"] if classification == "green" else ["absent", "empty"]
    activity_labels.extend(f"zone:{label}" for label in occupied_zones)
    labels = list(dict.fromkeys(preset_labels))
    if settings.get("auto_occupancy_label_enabled"):
        labels.append(prediction_label_for(raw_label, str(settings.get("prediction_label_style") or "occupancy")))
    if settings.get("people_count_label_enabled"):
        labels.append(f"people_count:{people_count}")
    labels.extend(activity_labels)

    chunk_index = int(result.get("chunk_index") or 0)
    result.update({
        "settings_revision": int(settings.get("revision") or 0),
        "settings_snapshot": {
            "revision": int(settings.get("revision") or 0),
            "system_mode": str(settings.get("system_mode") or "balanced"),
            "radar_detection_threshold_db": float(
                settings.get("radar_detection_threshold_db") or 8.0
            ),
            "chunk_frames": RADAR_FRAMES_PER_CHUNK,
        },
        "labels": list(dict.fromkeys(labels)),
        "zones": occupied_zones,
        "people_count": people_count,
        "activity_labels": list(dict.fromkeys(activity_labels)),
        "activity": {
            "state": "occupied" if classification == "green" else "empty",
            "labels": list(dict.fromkeys(activity_labels)),
            "zones": occupied_zones,
            "targets": activity_targets,
            "dwell_threshold_percent": dwell_threshold,
        },
        "join": {
            "schema_version": 2,
            "minute": minute,
            "chunk_id": f"{minute}:{chunk_index:02d}",
            "chunk_index": chunk_index,
            "expected_chunks": expected_chunks,
            "previous_chunk_id": f"{minute}:{chunk_index - 1:02d}" if chunk_index else None,
            "next_chunk_id": f"{minute}:{chunk_index + 1:02d}" if chunk_index + 1 < expected_chunks else None,
            "start_offset_seconds": round(chunk_index * float(result.get("chunk_seconds") or 0.0), 3),
            "duration_seconds": float(result.get("chunk_seconds") or 0.0),
            "frame_start": previous_frames,
            "frame_count": evaluated_frames,
            "frame_end_exclusive": previous_frames + evaluated_frames,
            "source_files": {
                "radar_bin": Path(str(result.get("bin_path") or "")).name,
                "camera_image": Path(str(result.get("camera_path") or "")).name or None,
            },
        },
    })
    return result


def summarize_minute_results(
    chunks: list[dict[str, Any]], settings: dict[str, Any], preset_labels: list[str]
) -> dict[str, Any]:
    occupied_chunks = sum((chunk.get("occupancy") or {}).get("label") == "occupied" for chunk in chunks)
    vote_required = 1
    label = "occupied" if occupied_chunks > 0 else "empty"
    detected_frames = sum(int((chunk.get("occupancy") or {}).get("detected_frames") or 0) for chunk in chunks)
    evaluated_frames = sum(int((chunk.get("occupancy") or {}).get("evaluated_frames") or 0) for chunk in chunks)
    ratio = detected_frames / evaluated_frames if evaluated_frames else 0.0
    classification = "green" if label == "occupied" else "red"
    people_count = max((int(chunk.get("people_count") or 0) for chunk in chunks), default=0)
    labels = list(dict.fromkeys(preset_labels))
    if settings.get("auto_occupancy_label_enabled"):
        labels.append(prediction_label_for(label, str(settings.get("prediction_label_style") or "occupancy")))
    if settings.get("people_count_label_enabled"):
        labels.append(f"people_count:{people_count}")
    occupied_zones = list(dict.fromkeys(
        str(zone) for chunk in chunks for zone in (chunk.get("zones") or []) if str(zone).strip()
    ))
    activity_labels = ["present", "occupied"] if label == "occupied" else ["absent", "empty"]
    activity_labels.extend(f"zone:{zone}" for zone in occupied_zones)
    labels.extend(activity_labels)
    latest = chunks[-1] if chunks else {}
    return {
        "occupancy": {
            "label": label,
            "classification": classification,
            "occupied_chunks": occupied_chunks,
            "evaluated_chunks": len(chunks),
            "vote_required_chunks": vote_required,
            "detected_frames": detected_frames,
            "evaluated_frames": evaluated_frames,
            "ratio": ratio,
            "threshold_db": float((latest.get("occupancy") or {}).get("threshold_db") or 8.0),
        },
        "labels": list(dict.fromkeys(labels)),
        "zones": occupied_zones,
        "activity_labels": list(dict.fromkeys(activity_labels)),
        "activity": {
            "state": label,
            "labels": list(dict.fromkeys(activity_labels)),
            "zones": occupied_zones,
            "occupied_chunks": occupied_chunks,
            "evaluated_chunks": len(chunks),
            "vote_required_chunks": vote_required,
        },
        "people_count": people_count,
        "targets": latest.get("targets") or [],
        "location": latest.get("location"),
        "score": latest.get("score"),
    }


def find_camera(requested: str | None) -> str | None:
    if requested:
        return requested

    devices = sorted(
        glob.glob("/dev/video*"),
        key=lambda path: int(Path(path).name.removeprefix("video")),
    )
    usb_devices = []
    for device in devices:
        sysfs_device = Path("/sys/class/video4linux") / Path(device).name / "device"
        try:
            resolved = sysfs_device.resolve(strict=True)
        except OSError:
            continue
        if any((parent / "idVendor").exists() for parent in (resolved, *resolved.parents)):
            usb_devices.append(device)

    v4l2_ctl = shutil.which("v4l2-ctl")
    if v4l2_ctl:
        for device in usb_devices:
            try:
                probe = subprocess.run(
                    [v4l2_ctl, "--device", device, "--all"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=3,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            is_usb_camera = "Driver name      : uvcvideo" in probe.stdout or "Bus info         : usb-" in probe.stdout
            is_capture_node = "Video Capture" in probe.stdout and "Video Output" not in probe.stdout
            if probe.returncode == 0 and is_usb_camera and is_capture_node:
                return device
    return None


def serial_candidates() -> list[str]:
    try:
        import serial.tools.list_ports as list_ports

        ports = [port.device for port in list_ports.comports()]
    except Exception:
        ports = []

    globbed = glob.glob("/dev/serial/by-id/*") + glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
    candidates = []
    for port in ports + globbed:
        resolved = str(Path(port).resolve()) if port.startswith("/dev/serial/by-id/") else port
        if resolved not in candidates:
            candidates.append(resolved)
    return sorted(candidates)


def open_serial_without_reset(port: str, baud: int, timeout: float):
    import serial

    connection = serial.Serial()
    connection.port = port
    connection.baudrate = baud
    connection.timeout = timeout
    # ESP32-C6 USB Serial/JTAG reboots when DTR is deasserted as the port opens.
    # Keep DTR asserted and RTS deasserted so rotating minute files does not
    # reset the CSI receiver firmware.
    connection.dtr = True
    connection.rts = False
    connection.open()
    return connection


def probe_csi_port(port: str, baud: int, timeout_s: float) -> bool:
    try:
        import serial

        deadline = time.monotonic() + timeout_s
        with open_serial_without_reset(port, baud, 0.1) as ser:
            while time.monotonic() < deadline:
                line = ser.readline()
                if not line:
                    continue
                text = line.decode("utf-8", errors="ignore").strip()
                if text.startswith("CSI_DATA,"):
                    return True
    except Exception:
        return False
    return False


def find_csi_port(requested: str, baud: int, detect_seconds: float) -> tuple[str | None, list[str]]:
    candidates = serial_candidates()
    if requested and requested != "auto":
        return requested, candidates

    try:
        import serial.tools.list_ports as list_ports

        for port in list_ports.comports():
            manufacturer = (port.manufacturer or "").lower()
            description = (port.description or "").lower()
            if port.vid == 0x303A or "espressif" in manufacturer or "espressif" in description:
                return port.device, candidates
    except Exception:
        pass

    if detect_seconds <= 0:
        return (candidates[0] if candidates else None), candidates

    per_port_timeout = max(0.2, detect_seconds / max(1, len(candidates)))
    for port in candidates:
        if probe_csi_port(port, baud, per_port_timeout):
            return port, candidates
    return (candidates[0] if candidates else None), candidates


def collect_csi(
    port: str,
    baud: int,
    output_file: Path,
    stop_event: threading.Event,
) -> None:
    try:
        import serial
    except Exception as exc:
        with open(output_file.with_suffix(".error.json"), "w", encoding="utf-8") as fd:
            json.dump({"timestamp": iso_now(), "error": f"pyserial import failed: {exc}"}, fd, indent=2)
        return

    try:
        with (
            open_serial_without_reset(port, baud, 0.05) as ser,
            open(output_file, "w", encoding="utf-8", newline="", buffering=1) as output_fd,
        ):
            writer = csv.writer(output_fd)
            writer.writerow(["host_timestamp", "monotonic_ns", "serial_port", "raw_csi_line"])

            if ser.in_waiting:
                ser.read(ser.in_waiting)

            while not stop_event.is_set():
                line = ser.readline()
                if not line:
                    continue
                host_timestamp = iso_now()
                monotonic_ns = time.monotonic_ns()
                text = line.decode("utf-8", errors="ignore").strip()
                if not text:
                    continue
                if text.startswith("CSI_DATA,"):
                    writer.writerow([host_timestamp, monotonic_ns, port, text])
    except Exception as exc:
        with open(output_file.with_suffix(".error.json"), "w", encoding="utf-8") as fd:
            json.dump({"timestamp": iso_now(), "port": port, "baud": baud, "error": str(exc)}, fd, indent=2)


def collect_sensehat(output_file: Path, stop_event: threading.Event, interval: float = 0.2) -> None:
    try:
        from sense_hat import SenseHat
    except Exception as exc:
        output_file.with_suffix(".error.json").write_text(
            json.dumps({"timestamp": iso_now(), "error": f"sense_hat import failed: {exc}"}, indent=2),
            encoding="utf-8",
        )
        return

    try:
        sense = SenseHat()
        with open(output_file, "w", encoding="utf-8", buffering=1) as fd:
            while not stop_event.is_set():
                row = {
                    "host_timestamp": iso_now(),
                    "monotonic_ns": time.monotonic_ns(),
                    "temperature_c": sense.get_temperature(),
                    "humidity_percent": sense.get_humidity(),
                    "pressure_mbar": sense.get_pressure(),
                    "acceleration": sense.get_accelerometer_raw(),
                    "gyroscope": sense.get_gyroscope_raw(),
                    "compass": sense.get_compass_raw(),
                    "orientation": sense.get_orientation(),
                }
                fd.write(json.dumps(row, separators=(",", ":")) + "\n")
                time.sleep(max(0.05, interval))
    except Exception as exc:
        output_file.with_suffix(".error.json").write_text(
            json.dumps({"timestamp": iso_now(), "error": str(exc)}, indent=2),
            encoding="utf-8",
        )


def capture_camera_image(camera: str, output_file: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found in PATH.")

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "v4l2",
        "-i",
        camera,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_file),
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=5,
        check=False,
    )
    if result.returncode != 0 or not output_file.exists() or output_file.stat().st_size == 0:
        output_file.unlink(missing_ok=True)
        detail = result.stderr.decode("utf-8", errors="replace")[-300:]
        raise RuntimeError(f"camera snapshot failed ({result.returncode}): {detail}")


def start_radar_capture(output_prefix: Path | None = None) -> Any:
    if not Path("/dev/spidev0.0").exists():
        raise RuntimeError("/dev/spidev0.0 is missing; enable SPI and reboot the Raspberry Pi.")

    from utility.BGT60TR13C import BGT60TR13C, RET_VAL_OK
    from utility.helper import calculate_frame_size, find_register_config_in_directory, find_setting_in_directory

    bgt60tr13c = None
    try:
        bgt60tr13c = BGT60TR13C(
            spi_speed=50_000_000,
            save_to_file=str(output_prefix) if output_prefix is not None else None,
        )
        if bgt60tr13c.check_chip_id() != RET_VAL_OK:
            raise RuntimeError("BGT60TR13C chip ID check failed.")

        reg_file = find_register_config_in_directory(str(RADAR_CFG))
        setting_file = find_setting_in_directory(str(RADAR_CFG))
        bgt60tr13c.load_register_config_file(reg_file)

        with open(setting_file, "r", encoding="utf-8") as fd:
            setting_data = json.load(fd)

        frame_size = calculate_frame_size(setting_data)
        bgt60tr13c.set_fifo_parameters(frame_size, 4096, 2048)
        if bgt60tr13c.start() != RET_VAL_OK:
            raise RuntimeError("BGT60TR13C failed to start.")

        return bgt60tr13c
    except Exception:
        stop_radar_capture(bgt60tr13c)
        raise


def stop_radar_capture(radar: Any | None) -> None:
    if radar is not None:
        radar.stop()


def chown_to_invoking_user(path: Path) -> None:
    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")
    if not sudo_uid or not sudo_gid:
        return
    uid = int(sudo_uid)
    gid = int(sudo_gid)
    for root, dirs, files in os.walk(path):
        os.chown(root, uid, gid)
        for name in dirs:
            os.chown(os.path.join(root, name), uid, gid)
        for name in files:
            os.chown(os.path.join(root, name), uid, gid)


def main() -> int:
    args = parse_args()
    preset_labels = normalize_labels(args.label or [])
    chunk_seconds = 1.0
    expected_chunks = max(1, int(math.floor(float(args.duration))))
    target_start = minute_start(args.start_now, args.scheduled_start)
    folder_name = target_start.strftime("%Y%m%d_%H%M")
    output_dir = output_dir_for_minute(folder_name, preset_labels)
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest: dict[str, Any] = {
        "schema": "thoth-minute-manifest/v4",
        "collection_unit": "minute",
        "sample_unit": "one-second synchronized sensor window",
        "folder_minute": folder_name,
        "scheduled_start": target_start.isoformat(timespec="seconds"),
        "duration_seconds": args.duration,
        "chunk_seconds": chunk_seconds,
        "chunk_frames": RADAR_FRAMES_PER_CHUNK,
        "expected_chunks": expected_chunks,
        "labels": preset_labels or ["collecting"],
        "outputs": {},
        "assets": [],
        "errors": [],
        "warnings": [],
        "primary_label": preset_labels[0] if preset_labels else "collecting",
        "relative_path": str(output_dir.relative_to(DATA_ROOT)),
        "sensors_enabled": {
            "usb_camera": not args.no_camera,
            "dreamhat_radar": not args.no_radar,
            "esp32_csi": not args.no_csi,
            "sense_hat": not args.no_sensehat,
        },
    }

    csi_port = None
    csi_candidates: list[str] = []
    if not args.no_csi:
        csi_port, csi_candidates = find_csi_port(args.csi_port, args.csi_baud, args.csi_detect_seconds)
        if csi_port is None:
            manifest["errors"].append("No ESP32 CSI serial device found.")

    print(f"Output folder: {output_dir}")
    print(f"Waiting for real-clock minute: {target_start.isoformat(timespec='seconds')}")
    sleep_until(target_start)

    stop_at = time.monotonic() + args.duration
    capture_started = iso_now()
    print(f"Capture started: {capture_started}")

    csi_stop = threading.Event()
    csi_thread: threading.Thread | None = None
    camera: str | None = None
    camera_queue: queue.Queue[Any] = queue.Queue()
    camera_thread: threading.Thread | None = None
    radar_reader_thread: threading.Thread | None = None
    radar_analysis_thread: threading.Thread | None = None
    radar_live_thread: threading.Thread | None = None
    radar_upload_thread: threading.Thread | None = None
    analysis_queue: queue.Queue[Any] = queue.Queue(maxsize=MAX_PENDING_ANALYSIS_CHUNKS)
    live_analysis_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
    live_queue_key: dict[str, Any] = {"stream": folder_name}
    upload_queue: queue.Queue[Any] = queue.Queue()
    radar_chunk_results: list[dict[str, Any]] = []
    publish_lock = threading.Lock()
    room_config = load_room_config()

    def merge_home_assistant_status() -> None:
        status_path = output_dir / ".home_assistant_status.json"
        try:
            statuses = json.loads(status_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return
        chunks = (((manifest.get("outputs") or {}).get("radar") or {}).get("chunks") or [])
        minute_status = statuses.get("minute") if isinstance(statuses, dict) else None
        if isinstance(minute_status, dict):
            manifest["home_assistant"] = minute_status
        for entry in chunks:
            status = statuses.get(str(entry.get("chunk_index"))) if isinstance(statuses, dict) else None
            if isinstance(status, dict):
                entry["home_assistant"] = status

    def write_live_manifest() -> None:
        merge_home_assistant_status()
        assets: list[dict[str, Any]] = []
        for entry in radar_chunk_results:
            index = int(entry.get("chunk_index") or 0)
            result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
            occupancy = result.get("occupancy") if isinstance(result.get("occupancy"), dict) else {}
            common = {
                "second_index": index,
                "started_at": entry.get("started"),
                "finished_at": entry.get("finished_capture"),
                "duration_seconds": entry.get("chunk_seconds"),
                "labels": result.get("labels") or entry.get("labels") or [],
                "prediction": occupancy.get("label") or entry.get("status"),
                "properties": {
                    "frame_count": entry.get("chunk_frames"),
                    "detected_frames": occupancy.get("detected_frames"),
                    "evaluated_frames": occupancy.get("evaluated_frames"),
                    "ratio": occupancy.get("ratio"),
                    "people_count": result.get("people_count", entry.get("people_count", 0)),
                },
            }
            radar_name = Path(str(entry.get("bin_path") or "")).name
            if radar_name:
                assets.append({**common, "sensor": "radar", "filename": radar_name, "content_type": "application/octet-stream"})
            camera_name = Path(str(entry.get("camera_path") or "")).name
            if camera_name:
                assets.append({**common, "sensor": "camera", "filename": camera_name, "content_type": "image/jpeg"})
        csi_output = (manifest.get("outputs") or {}).get("wifi_csi")
        if isinstance(csi_output, dict) and csi_output.get("path"):
            assets.append({
                "sensor": "wifi_csi",
                "filename": Path(str(csi_output["path"])).name,
                "content_type": "text/csv",
                "started_at": csi_output.get("started"),
                "duration_seconds": args.duration,
                "coverage": "continuous minute stream",
                "labels": manifest.get("labels") or [],
                "properties": {"baud": csi_output.get("baud"), "device": csi_output.get("device")},
            })
        manifest["assets"] = assets
        snapshot = dict(manifest)
        snapshot["capture_started"] = capture_started
        snapshot["status"] = "collecting"
        outputs = dict(manifest.get("outputs") or {})
        radar_output = outputs.get("radar")
        if isinstance(radar_output, dict):
            radar_snapshot = dict(radar_output)
            radar_snapshot["chunks"] = [
                {key: value for key, value in entry.items() if key != "result"}
                for entry in (radar_output.get("chunks") or [])
            ]
            outputs["radar"] = radar_snapshot
        snapshot["outputs"] = outputs
        write_json_atomic(output_dir / "manifest.json", snapshot)

    def publish_radar_results() -> None:
        with publish_lock:
            completed = [entry["result"] for entry in radar_chunk_results if isinstance(entry.get("result"), dict)]
            if not completed:
                return
            active_settings = next((
                item.get("settings") for item in radar_chunk_results
                if isinstance(item.get("settings"), dict)
            ), load_processing_settings())
            minute_summary = summarize_minute_results(completed, active_settings, preset_labels)
            manifest["labels"] = minute_summary["labels"]
            manifest["primary_label"] = minute_summary["labels"][0]
            manifest["minute_summary"] = minute_summary
            manifest["auto_occupancy_label"] = minute_summary["occupancy"]
            # Keep the live artifact bounded to one native 10-frame chunk.
            # Rebuilding and serving the entire minute here produced
            # multi-megabyte responses every second and made analysis fall
            # progressively behind the sensor.  The complete minute playback
            # is written once during finalization below.
            xy_payload = compile_minute_xy_payload([completed[-1]])
            xy_payload["z"] = []
            xy_payload["occupancy"] = minute_summary["occupancy"]
            xy_payload["frame_count"] = sum(
                int((item.get("occupancy") or {}).get("evaluated_frames") or 0)
                for item in completed
            )
            xy_payload["sample_count"] = len(xy_payload.get("frames") or [])
            xy_payload["stream_window_frames"] = RADAR_FRAMES_PER_CHUNK
            xy_payload["live"] = True
            write_json_atomic(output_dir / "xy-tracking.json", xy_payload)
            manifest["outputs"]["radar"]["xy_tracking"] = str(output_dir / "xy-tracking.json")
            write_live_manifest()

    def upload_live_chunk(index: int) -> None:
        entry = next((
            item for item in radar_chunk_results
            if int(item.get("chunk_index", -1)) == index
        ), None)
        if not entry:
            return
        result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
        occupancy = result.get("occupancy") if isinstance(result.get("occupancy"), dict) else {}
        payload = json.dumps({
            "minute": folder_name,
            "chunk_index": index,
            "chunk_frames": RADAR_FRAMES_PER_CHUNK,
            "status": entry.get("status") or "loading",
            "occupancy": occupancy,
            "location": result.get("location"),
            "score": result.get("score"),
            "people_count": result.get("people_count", 0),
            "targets": result.get("targets") or [],
            "labels": result.get("labels") or [],
            "activity_labels": result.get("activity_labels") or [],
            "xy_map": result.get("xy_map") or {},
            "camera_filename": Path(str(entry.get("camera_path") or "")).name or None,
            "captured_at": entry.get("finished") or entry.get("started") or iso_now(),
        }, separators=(",", ":")).encode("utf-8")
        try:
            request = urllib.request.Request(
                "http://127.0.0.1:5000/api/internal/capture-chunk",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5.0):
                pass
        except Exception as exc:
            print(f"Live timeline update {index} deferred: {exc}", file=sys.stderr)

    def enqueue_home_assistant(entry: dict[str, Any], occupancy: dict[str, Any], result: dict[str, Any], scope: str = "chunk") -> None:
        payload = json.dumps({
            "minute": folder_name,
            "scope": scope,
            "chunk_index": entry.get("chunk_index"),
            "occupancy": occupancy,
            "location": result.get("location"),
            "confidence": result.get("score"),
            "targets": result.get("targets") or [],
            "people_count": result.get("people_count"),
            "labels": result.get("labels") or [],
            "activity_labels": result.get("activity_labels") or [],
            "activity": result.get("activity"),
            "timestamp": entry.get("finished") or iso_now(),
        }).encode("utf-8")
        try:
            request = urllib.request.Request(
                "http://127.0.0.1:5000/api/internal/home-assistant/publish",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=1.0):
                pass
            entry["home_assistant"] = {"success": True, "status": "queued", "updated_at": iso_now()}
        except Exception as exc:
            entry["home_assistant"] = {"success": False, "status": "queue_error", "error": str(exc), "updated_at": iso_now()}

    def run_upload_worker() -> None:
        while True:
            index = upload_queue.get()
            try:
                if index is None:
                    return
                upload_live_chunk(int(index))
            finally:
                upload_queue.task_done()

    def run_camera_worker() -> None:
        while True:
            item = camera_queue.get()
            try:
                if item is None:
                    return
                entry = item
                if not camera:
                    continue
                stamp = str(entry.get("started") or iso_now()).replace("-", "").replace(":", "").replace("T", "_")
                stamp = stamp.replace("+", "_").replace(".", "_")[:19]
                image_path = output_dir / f"camera_{int(entry['chunk_index']):03d}_{stamp}.jpg"
                try:
                    capture_camera_image(camera, image_path)
                    entry["camera_path"] = str(image_path)
                    with publish_lock:
                        write_live_manifest()
                    if isinstance(entry.get("result"), dict):
                        upload_queue.put(int(entry["chunk_index"]))
                except Exception as exc:
                    entry["camera_error"] = str(exc)
            finally:
                camera_queue.task_done()

    def run_analysis_worker() -> None:
        try:
            processor = create_signal_processor()
        except Exception as init_exc:
            while True:
                failed_job = analysis_queue.get()
                try:
                    if failed_job is None:
                        return
                    failed_entry = failed_job[1]
                    failed_entry.update({
                        "status": "empty",
                        "classification": "red",
                        "occupied": False,
                        "data_quality": "analysis_error",
                        "error": str(init_exc),
                        "finished": iso_now(),
                    })
                    with publish_lock:
                        write_live_manifest()
                finally:
                    analysis_queue.task_done()
        identity: PersistentTargetIdentity | None = None
        while True:
            job = analysis_queue.get()
            entry: dict[str, Any] | None = None
            analyzer: StreamingChunkAnalyzer | None = None
            try:
                if job is None:
                    return
                _kind, entry, settings_snapshot, frames, captured_at = job
                mode = str(settings_snapshot.get("system_mode") or "balanced")
                if identity is None:
                    identity = PersistentTargetIdentity(mode=mode)
                if hasattr(processor, "set_system_mode"):
                    processor.set_system_mode(mode)
                analyzer = StreamingChunkAnalyzer(
                    processor,
                    None,
                    int(entry["chunk_index"]),
                    float(entry["chunk_seconds"]),
                    room_config,
                    0.45,
                    0.0,
                    0.0,
                    0.01,
                    identity=identity,
                    live_state_path=None,
                    radar_detection_threshold_db=float(
                        settings_snapshot.get("radar_detection_threshold_db") or 8.0
                    ),
                )
                entry["status"] = "analyzing"
                with publish_lock:
                    write_live_manifest()
                analyzer.max_queue_lag_ms = max(
                    0.0, (time.monotonic() - float(captured_at)) * 1000
                )
                for frame in frames:
                    analyzer.process(frame)
                result = analyzer.finish()
                result["bin_path"] = entry.get("bin_path")
                result["camera_path"] = entry.get("camera_path")
                # Frame offsets follow the captured 10-frame bins, even when a
                # stale analysis job was deferred to keep the live view current.
                previous_frames = int(entry["chunk_index"]) * RADAR_FRAMES_PER_CHUNK
                chunk_labels = normalize_labels(settings_snapshot.get("labels")) or preset_labels
                annotate_chunk_result(
                    result, settings_snapshot, room_config, chunk_labels,
                    folder_name, expected_chunks, previous_frames,
                )
                entry["result"] = result
                occupancy = result.get("occupancy", {})
                label = occupancy.get("label") or "empty"
                if int(occupancy.get("evaluated_frames") or 0) != RADAR_FRAMES_PER_CHUNK:
                    raise RuntimeError(
                        f"analyzed {occupancy.get('evaluated_frames', 0)} of "
                        f"{RADAR_FRAMES_PER_CHUNK} radar frames"
                    )
                entry.update({
                    "status": label,
                    "detected_frames": occupancy.get("detected_frames", 0),
                    "evaluated_frames": occupancy.get("evaluated_frames", 0),
                    "ratio": occupancy.get("ratio", 0.0),
                    "occupied": label == "occupied",
                    "location": result.get("location"),
                    "score": result.get("score"),
                    "people_count": result.get("people_count", 0),
                    "targets": result.get("targets") or [],
                    "labels": result.get("labels") or [],
                    "activity_labels": result.get("activity_labels") or [],
                    "activity": result.get("activity"),
                    "join": result.get("join"),
                    "performance": result.get("performance"),
                    "finished": iso_now(),
                })
                if settings_snapshot.get("auto_occupancy_label_enabled"):
                    manifest["auto_occupancy_label"] = occupancy
                publish_radar_results()
                enqueue_home_assistant(entry, occupancy, result)
                upload_queue.put(int(entry["chunk_index"]))
            except Exception as exc:
                if entry is not None:
                    entry["error"] = str(exc)
                    entry.update({
                        "status": "empty",
                        "classification": "red",
                        "occupied": False,
                        "data_quality": "analysis_error",
                        "finished": iso_now(),
                    })
                    with publish_lock:
                        write_live_manifest()
                    upload_queue.put(int(entry["chunk_index"]))
                if analyzer is not None:
                    try:
                        analyzer.handle.close()
                    except Exception:
                        pass
            finally:
                analysis_queue.task_done()

    def run_live_analysis_worker() -> None:
        """Analyze only the newest captured frame for the live Presence view.

        Saved chunk analysis remains a separate exact 10-frame pipeline. This
        lightweight worker is allowed to replace a pending visual frame when
        rendering falls behind, which keeps Example 2 close to the sensor
        without ever dropping a frame from the persisted chunk.
        """
        try:
            processor = create_signal_processor()
        except Exception as exc:
            print(f"Live radar visualization unavailable: {exc}", file=sys.stderr)
            while True:
                failed_job = live_analysis_queue.get()
                try:
                    if failed_job is None:
                        return
                finally:
                    live_analysis_queue.task_done()

        analyzer: StreamingChunkAnalyzer | None = None
        analyzer_chunk_index: int | None = None
        try:
            while True:
                job = live_analysis_queue.get()
                try:
                    if job is None:
                        return
                    _, _, frame, captured_at, chunk_index, settings_snapshot = job
                    chunk_index = int(chunk_index)
                    if analyzer is None or analyzer_chunk_index != chunk_index:
                        if analyzer is not None:
                            analyzer.handle.close()
                        mode = str(settings_snapshot.get("system_mode") or "balanced")
                        if hasattr(processor, "set_system_mode"):
                            processor.set_system_mode(mode)
                        analyzer = StreamingChunkAnalyzer(
                            processor,
                            None,
                            chunk_index,
                            1.0,
                            room_config,
                            0.45,
                            0.0,
                            0.0,
                            0.01,
                            identity=None,
                            radar_detection_threshold_db=float(
                                settings_snapshot.get("radar_detection_threshold_db") or 8.0
                            ),
                            live_example2_only=True,
                        )
                        analyzer_chunk_index = chunk_index
                    analyzer.max_queue_lag_ms = max(
                        0.0, (time.monotonic() - float(captured_at)) * 1000
                    )
                    analyzer.process(frame)
                except Exception as exc:
                    print(f"Live radar frame deferred: {exc}", file=sys.stderr)
                finally:
                    live_analysis_queue.task_done()
        finally:
            if analyzer is not None:
                analyzer.handle.close()

    def run_radar_reader(radar: Any) -> None:
        frames: list[bytes] = []
        frame_times: list[float] = []
        live_settings: dict[str, Any] = {}
        last_live_enqueue = 0.0
        while time.monotonic() < stop_at:
            remaining = stop_at - time.monotonic()
            try:
                full_frame = bytes(radar.frame_buffer.get(timeout=min(0.1, max(0.01, remaining))))
            except queue.Empty:
                continue
            captured_at = time.monotonic()
            if not frames:
                live_settings = load_processing_settings()
            if captured_at - last_live_enqueue >= LIVE_VISUALIZATION_INTERVAL_SECONDS:
                enqueue_latest_chunk_frame(
                    live_analysis_queue,
                    live_queue_key,
                    full_frame,
                    captured_at,
                    len(radar_chunk_results),
                    live_settings,
                )
                last_live_enqueue = captured_at
            frames.append(full_frame)
            frame_times.append(captured_at)
            if len(frames) < RADAR_FRAMES_PER_CHUNK:
                continue

            chunk_index = len(radar_chunk_results)
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            radar_path = output_dir / f"radar_{chunk_index:03d}_{timestamp}.bin"
            radar_path.write_bytes(b"".join(frames))
            duration = max(0.001, frame_times[-1] - frame_times[0])
            settings_snapshot = load_processing_settings()
            chunk_entry: dict[str, Any] = {
                "chunk_index": chunk_index,
                "bin_path": str(radar_path),
                "started": dt.datetime.fromtimestamp(
                    time.time() - duration
                ).astimezone().isoformat(timespec="milliseconds"),
                "finished_capture": iso_now(),
                "chunk_seconds": duration,
                "chunk_frames": RADAR_FRAMES_PER_CHUNK,
                "status": "loading",
                "settings": settings_snapshot,
                "frame_sequence_start": int.from_bytes(frames[0][4:8], "little"),
                "frame_sequence_end": int.from_bytes(frames[-1][4:8], "little"),
            }
            radar_chunk_results.append(chunk_entry)
            manifest["outputs"]["radar"]["chunks"].append(chunk_entry)
            manifest["expected_chunks"] = max(expected_chunks, len(radar_chunk_results))
            with publish_lock:
                write_live_manifest()
            camera_queue.put(chunk_entry)
            upload_queue.put(chunk_index)
            analysis_job = (
                "chunk",
                chunk_entry,
                settings_snapshot,
                tuple(frames),
                frame_times[-1],
            )
            dropped_entries = enqueue_analysis_chunk(analysis_queue, analysis_job)
            for dropped in dropped_entries:
                dropped.update({
                    "status": "captured",
                    "data_quality": "analysis_deferred",
                    "analysis_deferred_at": iso_now(),
                })
            frames = []
            frame_times = []

        if frames:
            manifest["warnings"].append(
                f"Discarded {len(frames)} radar frames at the minute boundary; "
                f"chunks require exactly {RADAR_FRAMES_PER_CHUNK} frames."
            )

    try:
        if not args.no_csi and csi_port is not None:
            csi_file = output_dir / "wifi_csi.csv"
            csi_started = iso_now()
            csi_thread = threading.Thread(
                target=collect_csi,
                args=(csi_port, args.csi_baud, csi_file, csi_stop),
                daemon=True,
            )
            csi_thread.start()
            manifest["outputs"]["wifi_csi"] = {
                "path": str(csi_file),
                "type": "csv",
                "device": csi_port,
                "baud": args.csi_baud,
                "detected_candidates": csi_candidates,
                "started": csi_started,
                "source": "USB-connected ESP32 receiver printing ESP-NOW CSI_DATA lines",
            }
        elif not args.no_csi and csi_port is None:
            csi_file = output_dir / "wifi_csi.csv"
            with open(csi_file, "w", encoding="utf-8", newline="") as output_fd:
                csv.writer(output_fd).writerow(["host_timestamp", "monotonic_ns", "serial_port", "raw_csi_line"])
            manifest["outputs"]["wifi_csi"] = {
                "path": str(csi_file),
                "type": "csv",
                "device": None,
                "baud": args.csi_baud,
                "detected_candidates": csi_candidates,
                "started": iso_now(),
                "source": "CSI receiver unavailable; header-only minute artifact",
                "data_quality": "sensor_missing",
            }
            manifest["warnings"].append("No ESP32 CSI serial device found; skipping CSI for this minute.")

        if not args.no_camera:
            camera = find_camera(args.camera)
            if camera is None:
                manifest["warnings"].append("No /dev/video* USB camera device found; skipping chunk images.")
            else:
                manifest["outputs"]["camera"] = {
                    "type": "chunked-jpeg",
                    "device": camera,
                    "frames_per_image": RADAR_FRAMES_PER_CHUNK,
                }
                camera_thread = threading.Thread(
                    target=run_camera_worker,
                    name="ChunkCamera",
                    daemon=True,
                )
                camera_thread.start()

        if not args.no_radar:
            manifest["outputs"]["radar"] = {
                "type": "chunked-bin",
                "config_dir": str(RADAR_CFG),
                "chunk_frames": RADAR_FRAMES_PER_CHUNK,
                "chunks": [],
                "note": "The hardware reader rotates one binary file for every 10 complete radar frames.",
            }
            with publish_lock:
                write_live_manifest()
            radar_analysis_thread = threading.Thread(target=run_analysis_worker, name="RadarAnalysis", daemon=True)
            radar_live_thread = threading.Thread(target=run_live_analysis_worker, name="RadarLive", daemon=True)
            radar_upload_thread = threading.Thread(target=run_upload_worker, name="RadarUpload", daemon=True)
            radar_analysis_thread.start()
            radar_live_thread.start()
            radar_upload_thread.start()
            radar = None
            try:
                radar = start_radar_capture()
            except Exception as exc:
                manifest["warnings"].append(f"Radar failed to start: {exc}")
            if radar is not None:
                radar_reader_thread = threading.Thread(
                    target=run_radar_reader,
                    args=(radar,),
                    name="RadarReader",
                    daemon=True,
                )
                radar_reader_thread.start()

        while True:
            remaining = stop_at - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.25))
        if radar_reader_thread is not None:
            radar_reader_thread.join(timeout=2)
        stop_radar_capture(radar if not args.no_radar else None)

    except KeyboardInterrupt:
        manifest["errors"].append("Interrupted by user.")
    except Exception as exc:
        manifest["errors"].append(str(exc))
    finally:
        csi_stop.set()
        if csi_thread is not None:
            csi_thread.join(timeout=5)
        if radar_analysis_thread is not None:
            analysis_queue.put(None)
            radar_analysis_thread.join(timeout=90.0)
            if radar_analysis_thread.is_alive():
                manifest["errors"].append("Radar analysis exceeded its shutdown deadline.")
        if radar_live_thread is not None:
            live_analysis_queue.put(None)
            radar_live_thread.join(timeout=5.0)
        if camera_thread is not None:
            camera_queue.put(None)
            camera_thread.join(timeout=30.0)
        if radar_upload_thread is not None:
            upload_queue.put(None)
            radar_upload_thread.join(timeout=15.0)
        completed_chunks: list[dict[str, Any]] = []
        for chunk in radar_chunk_results:
            if not isinstance(chunk, dict):
                continue
            result = chunk.pop("result", None)
            error = chunk.get("error")
            if isinstance(result, dict):
                completed_chunks.append(result)
                chunk.update({
                    "status": result.get("occupancy", {}).get("label", "empty"),
                    "detected_frames": result.get("occupancy", {}).get("detected_frames", 0),
                    "evaluated_frames": result.get("occupancy", {}).get("evaluated_frames", 0),
                    "occupied": result.get("occupancy", {}).get("label") == "occupied",
                    "location": result.get("location"),
                    "score": result.get("score"),
                    "targets": result.get("targets", []),
                    "people_count": result.get("people_count", 0),
                    "labels": result.get("labels", []),
                    "activity_labels": result.get("activity_labels") or [],
                    "activity": result.get("activity"),
                    "join": result.get("join"),
                    "analysis": {
                        "occupancy": result.get("occupancy"),
                        "location": result.get("location"),
                        "score": result.get("score"),
                        "people_count": result.get("people_count", 0),
                        "targets": result.get("targets") or [],
                        "xy_map": result.get("xy_map") or {},
                    },
                    "finished": iso_now(),
                })
                if not error:
                    chunk.pop("error", None)
            elif error:
                chunk.update({
                    "status": "empty",
                    "classification": "red",
                    "occupied": False,
                    "data_quality": "analysis_error",
                    "error": error,
                    "finished": iso_now(),
                })

        if completed_chunks:
            minute_settings = next((
                chunk.get("settings") for chunk in radar_chunk_results
                if isinstance(chunk.get("settings"), dict)
            ), load_processing_settings())
            minute_summary = summarize_minute_results(completed_chunks, minute_settings, preset_labels)
            manifest["preset_labels"] = preset_labels
            manifest["labels"] = minute_summary["labels"]
            manifest["primary_label"] = minute_summary["labels"][0]
            manifest["minute_summary"] = minute_summary
            manifest["chunk_metadata_schema_version"] = 3
            xy_payload = compile_minute_xy_payload(completed_chunks)
            xy_payload["live"] = False
            write_json_atomic(output_dir / "xy-tracking.json", xy_payload)
            manifest["outputs"].setdefault("radar", {})["xy_tracking"] = str(
                output_dir / "xy-tracking.json"
            )
            minute_entry = {"finished": iso_now()}
            enqueue_home_assistant(minute_entry, minute_summary["occupancy"], minute_summary, scope="minute")
            manifest["home_assistant"] = minute_entry.get("home_assistant")
        else:
            manifest["preset_labels"] = preset_labels
            radar_files_present = any(output_dir.glob("radar_*.bin"))
            quality_label = "radar-analysis-failed" if radar_files_present else "radar-missing"
            manifest["labels"] = list(dict.fromkeys([*preset_labels, "empty", "absent", quality_label]))
            manifest["primary_label"] = manifest["labels"][0]
            manifest["minute_summary"] = {
                "occupancy": {
                    "label": "empty",
                    "classification": "red",
                    "occupied_chunks": 0,
                    "evaluated_chunks": 0,
                    "detected_frames": 0,
                    "evaluated_frames": 0,
                    "ratio": 0.0,
                    "threshold_db": float(load_processing_settings().get("radar_detection_threshold_db") or 8.0),
                },
                "labels": manifest["labels"],
                "activity_labels": ["absent", "empty", quality_label],
                "data_quality": quality_label,
                "people_count": 0,
                "targets": [],
                "location": None,
                "score": None,
            }
            if not radar_files_present and not args.no_radar:
                manifest["errors"].append("Radar produced no complete 10-frame chunks for this minute.")

        radar_files = sorted(str(path) for path in output_dir.glob("radar_*.bin"))
        if radar_files:
            manifest["outputs"].setdefault("radar", {})["files"] = radar_files
        camera_files = sorted(str(path) for path in output_dir.glob("camera_*.jpg"))
        if camera_files:
            manifest["outputs"].setdefault("camera", {})["files"] = camera_files
        if camera is not None and len(camera_files) != len(radar_files):
            manifest["errors"].append(
                f"Camera captured {len(camera_files)} of {len(radar_files)} expected chunk images."
            )

        wifi_csi = manifest["outputs"].get("wifi_csi")
        if isinstance(wifi_csi, dict):
            path = Path(wifi_csi["path"])
            if not path.exists() and not args.no_csi:
                with open(path, "w", encoding="utf-8", newline="") as output_fd:
                    csv.writer(output_fd).writerow(["host_timestamp", "monotonic_ns", "serial_port", "raw_csi_line"])
                wifi_csi["data_quality"] = "capture_error"
                manifest["errors"].append("CSI capture failed; wrote the required header-only minute CSV.")
            if path.exists():
                wifi_csi["bytes"] = path.stat().st_size
            if path.exists() and path.stat().st_size < 100:
                manifest["warnings"].append("ESP32 CSI receiver was detected but produced no CSI_DATA samples this minute.")
        manifest["expected_chunks"] = len(radar_chunk_results)

        manifest["capture_started"] = capture_started
        manifest["capture_finished"] = iso_now()
        manifest["host"] = os.uname().nodename
        manifest["status"] = "success" if not manifest["errors"] else "partial" if manifest["warnings"] else "error"
        manifest_file = output_dir / "manifest.json"
        merge_home_assistant_status()
        write_json_atomic(manifest_file, manifest)

        chown_to_invoking_user(output_dir)

    print(f"Capture finished: {manifest['capture_finished']}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    if manifest["warnings"]:
        print("Warnings:")
        for warning in manifest["warnings"]:
            print(f"  - {warning}")
    if manifest["errors"]:
        print("Errors:")
        for error in manifest["errors"]:
            print(f"  - {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
