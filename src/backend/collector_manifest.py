"""Minute manifest, settings, and label helpers for the collector.

Pure functions extracted from ``minute_collector.py``: label normalization,
minute timing, atomic JSON writes, the compact v7 manifest projection,
processing-settings loading, and per-chunk / per-minute result annotation.
None of these touch hardware or collector runtime state — they transform
plain dicts and paths, which keeps them easy to test and reuse.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.config import Config  # type: ignore
    from backend.capture_hardware import RADAR_FRAMES_PER_SECOND  # type: ignore
else:
    from .config import Config
    from .capture_hardware import RADAR_FRAMES_PER_SECOND

DATA_ROOT = Path(Config.CAPTURE_DATA_DIR).expanduser()
CAPTURE_SETTINGS_PATH = Path(Config.CONFIG_DIR).expanduser() / "capture_settings.json"


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
    """Return the stable minute path; labels are manifest metadata only."""
    return DATA_ROOT / folder_name


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


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with open(temporary, "w", encoding="utf-8") as fd:
        json.dump(payload, fd, indent=2)
    temporary.replace(path)


def compact_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Create the deliberately small v7 minute index.

    Sensor payloads and processing intermediates live in capture.npz.  The
    manifest keeps only human-authored labels, sensor summaries, failures, and
    user-model timelines.
    """
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    compact_outputs: dict[str, Any] = {}

    wifi = outputs.get("wifi_csi") if isinstance(outputs.get("wifi_csi"), dict) else None
    if wifi is not None:
        raw_receivers = wifi.get("receivers") if isinstance(wifi.get("receivers"), list) else [wifi]
        receivers = []
        for index, receiver in enumerate(raw_receivers, start=1):
            if not isinstance(receiver, dict):
                continue
            receiver_summary = {
                "device_id": receiver.get("device_id") or f"csi-{index}",
                "port": receiver.get("device"),
                "samples": int(receiver.get("sample_count") or 0),
                "average_sampling_rate_hz": float(receiver.get("average_sampling_rate_hz") or 0.0),
            }
            if not manifest.get("container"):
                receiver_summary["file"] = Path(str(receiver.get("path") or f"wifi_csi_{index:02d}.csv")).name
            receivers.append(receiver_summary)
        compact_outputs["wifi_csi"] = {
            "display_name": f"csix{len(receivers)}" if len(receivers) > 1 else "csi",
            "receiver_count": len(receivers),
            "receivers": receivers,
        }

    radar = outputs.get("radar") if isinstance(outputs.get("radar"), dict) else None
    if radar is not None:
        compact_outputs["radar"] = {
            "sample_count": int(radar.get("sample_count") or 0),
            "average_sampling_rate_hz": float(radar.get("average_sampling_rate_hz") or 0.0),
            "second_count": len(radar.get("seconds") or radar.get("chunks") or []),
        }
        if not manifest.get("container"):
            compact_outputs["radar"]["files"] = [Path(str(value)).name for value in (radar.get("files") or [])]

    for sensor in ("camera", "sense_hat"):
        output = outputs.get(sensor) if isinstance(outputs.get(sensor), dict) else None
        if output is not None:
            compact_outputs[sensor] = {
                key: value for key, value in output.items()
                if key in {"type", "device", "sample_count", "average_sampling_rate_hz"}
            }
            if output.get("files"):
                if not manifest.get("container"):
                    compact_outputs[sensor]["files"] = [Path(str(value)).name for value in output["files"]]

    compact = {
        key: manifest[key] for key in (
            "folder_minute", "scheduled_start", "capture_started", "capture_finished",
            "duration_seconds", "chunk_seconds", "expected_seconds", "status", "host",
            "labels", "sensors_enabled", "warnings", "errors",
            "device_id", "device_name", "container", "model_predictions",
            "progress",
        ) if key in manifest
    }
    compact.update({
        "schema": "thoth-minute-manifest/v7",
        "outputs": compact_outputs,
    })
    return compact


def load_processing_settings() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "labels": [],
        "system_mode": "balanced",
        "sleep_study_enabled": False,
        "csi_device_ids": {},
        "camera_fps": 1.0,
        "radar_detection_threshold_db": 8.0,
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
    mode = str(defaults.get("system_mode") or "balanced").strip().lower()
    defaults["system_mode"] = mode if mode in {"responsive", "balanced", "precision"} else "balanced"
    defaults["sleep_study_enabled"] = bool(defaults.get("sleep_study_enabled"))
    defaults["csi_device_ids"] = {
        str(port): str(device_id).strip()
        for port, device_id in (defaults.get("csi_device_ids") or {}).items()
        if str(port).strip() and str(device_id).strip()
    } if isinstance(defaults.get("csi_device_ids"), dict) else {}
    defaults["labels"] = normalize_labels(defaults.get("labels"))
    return defaults


def prediction_label_for(label: str, style: str) -> str:
    if style == "presence":
        return "present" if label == "occupied" else "absent"
    return "occupied" if label == "occupied" else "empty"


def annotate_chunk_result(
    result: dict[str, Any], settings: dict[str, Any], room: dict[str, Any],
    preset_labels: list[str], minute: str, expected_seconds: int,
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

    second_index = int(result.get("second_index") or 0)
    result.update({
        "settings_revision": int(settings.get("revision") or 0),
        "settings_snapshot": {
            "revision": int(settings.get("revision") or 0),
            "system_mode": str(settings.get("system_mode") or "balanced"),
            "radar_detection_threshold_db": float(
                settings.get("radar_detection_threshold_db") or 8.0
            ),
            "chunk_frames": RADAR_FRAMES_PER_SECOND,
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
            "chunk_id": f"{minute}:{second_index:02d}",
            "second_index": second_index,
            "expected_seconds": expected_seconds,
            "previous_chunk_id": f"{minute}:{second_index - 1:02d}" if second_index else None,
            "next_chunk_id": f"{minute}:{second_index + 1:02d}" if second_index + 1 < expected_seconds else None,
            "start_offset_seconds": round(second_index * float(result.get("chunk_seconds") or 0.0), 3),
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
