"""Non-blocking per-chunk occupancy publishing for Home Assistant."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import requests

from .config import Config

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(Config.CONFIG_DIR) / "home_assistant.json"
DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "base_url": "http://127.0.0.1:8123",
    "token": "",
    "entity_id": "binary_sensor.thoth_occupancy",
}
_status_lock = threading.Lock()
_last_status: Dict[str, Any] = {"status": "unknown", "updated_at": None}


def _record_status(result: Dict[str, Any]) -> Dict[str, Any]:
    with _status_lock:
        _last_status.clear()
        _last_status.update(result)
        _last_status["updated_at"] = datetime.now(timezone.utc).isoformat()
    return result


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def load_home_assistant_config(include_token: bool = False) -> Dict[str, Any]:
    config = dict(DEFAULTS)
    try:
        if CONFIG_PATH.exists():
            loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config.update({key: loaded[key] for key in DEFAULTS if key in loaded})
    except Exception as exc:
        logger.warning("Unable to load Home Assistant settings: %s", exc)
    config["configured"] = bool(config.get("token"))
    with _status_lock:
        config["connection_status"] = dict(_last_status)
    if not include_token:
        config.pop("token", None)
    return config


def save_home_assistant_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    config = load_home_assistant_config(include_token=True)
    if "enabled" in updates:
        config["enabled"] = bool(updates["enabled"])
    if updates.get("base_url") is not None:
        base_url = str(updates.get("base_url") or "").strip().rstrip("/")
        if base_url:
            config["base_url"] = base_url
    if updates.get("entity_id") is not None:
        entity_id = str(updates.get("entity_id") or "").strip().lower()
        if entity_id:
            config["entity_id"] = entity_id if entity_id.startswith("binary_sensor.") else f"binary_sensor.{entity_id}"
    # An empty browser field deliberately preserves the locally stored secret.
    if updates.get("token"):
        config["token"] = str(updates["token"]).strip()
    _write_json_atomic(CONFIG_PATH, {key: config[key] for key in DEFAULTS})
    return load_home_assistant_config()


def _occupancy_payload(
    occupancy: Dict[str, Any],
    minute: str,
    *,
    chunk_index: Optional[int] = None,
    location: Any = None,
    confidence: Any = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    detected = max(0, int(occupancy.get("detected_frames") or 0))
    total = max(0, int(occupancy.get("evaluated_frames") or 0))
    ratio = float(occupancy.get("ratio") or (detected / total if total else 0.0))
    label = str(occupancy.get("label") or "empty")
    classification = str(occupancy.get("classification") or ("green" if label == "occupied" else "red"))
    coordinates = location
    if isinstance(location, (list, tuple)):
        coordinates = {"x": location[0] if len(location) > 0 else None, "y": location[1] if len(location) > 1 else None}
    return {
        "state": "on" if label == "occupied" else "off",
        "attributes": {
            "friendly_name": "Thoth Occupancy",
            "device_class": "occupancy",
            "label": label,
            "classification": classification,
            "capture_minute": minute,
            "chunk_index": chunk_index,
            "detected_frames": detected,
            "evaluated_frames": total,
            "ratio": ratio,
            "detected_percent": round(ratio * 100, 2),
            "threshold_percent": float(occupancy.get("threshold_percent", 50.0)),
            "yellow_threshold_percent": float(occupancy.get("yellow_threshold_percent", 20.0)),
            "green_threshold_percent": float(occupancy.get("green_threshold_percent", 60.0)),
            "occupied_chunks": occupancy.get("occupied_chunks"),
            "evaluated_chunks": occupancy.get("evaluated_chunks"),
            "vote_required_chunks": occupancy.get("vote_required_chunks"),
            "coordinates": coordinates,
            "confidence": confidence,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        },
    }


def publish_occupancy(
    occupancy: Dict[str, Any],
    minute: str,
    *,
    chunk_index: Optional[int] = None,
    location: Any = None,
    confidence: Any = None,
    targets: Any = None,
    scope: str = "chunk",
    people_count: Any = None,
    labels: Any = None,
    activity_labels: Any = None,
    activity: Any = None,
    timestamp: Optional[str] = None,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    config = load_home_assistant_config(include_token=True)
    if not config.get("enabled"):
        return _record_status({"success": False, "status": "disabled"})
    if not config.get("token"):
        return _record_status({"success": False, "status": "not_configured"})

    payload = _occupancy_payload(
        occupancy, minute, chunk_index=chunk_index, location=location,
        confidence=confidence, timestamp=timestamp,
    )
    configured_entity_id = str(config["entity_id"])
    base_name = configured_entity_id.split('.', 1)[-1]
    if base_name.endswith('_occupancy'):
        base_name = base_name[:-10]
    scope = "minute" if scope == "minute" else "chunk"
    prefix = f"{base_name}_minute" if scope == "minute" else base_name
    entity_id = f"binary_sensor.{prefix}_occupancy" if scope == "minute" else configured_entity_id
    people_entity_id = f"sensor.{prefix}_people_count"
    targets_entity_id = f"sensor.{prefix}_target_coordinates"
    labels_entity_id = f"sensor.{prefix}_labels"
    zones_entity_id = f"sensor.{prefix}_zones"
    activity_entity_id = f"sensor.{prefix}_activity"
    region_entity_id = f"sensor.{prefix}_detection_region"
    target_list = targets if isinstance(targets, list) else []
    active_zones = list(dict.fromkeys(
        [str(label)[5:] for label in (labels or []) if str(label).startswith('zone:')]
        + [str(zone) for target in target_list if isinstance(target, dict) for zone in (target.get('zones') or [])]
    ))
    target_tuples = []
    for target in target_list:
        position = target.get('position') if isinstance(target, dict) else None
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            continue
        target_tuples.append({
            'target_id': target.get('id'),
            'x': round(float(position[0]), 3),
            'y': round(float(position[1]), 3),
            'error_m': round(float(target.get('position_error_m') or 0.0), 3),
        })
    companion_payloads = {
        region_entity_id: {
            'state': payload['attributes']['classification'],
            'attributes': {
                'friendly_name': f'Thoth {scope.title()} Detection Region',
                'ratio': payload['attributes']['ratio'],
                'yellow_threshold_percent': payload['attributes']['yellow_threshold_percent'],
                'green_threshold_percent': payload['attributes']['green_threshold_percent'],
                'capture_minute': minute, 'chunk_index': chunk_index, 'scope': scope,
                'timestamp': timestamp or datetime.now(timezone.utc).isoformat(),
            },
        },
        people_entity_id: {
            'state': int(people_count) if people_count is not None else len(target_tuples),
            'attributes': {
                'friendly_name': 'Thoth People Count',
                'unit_of_measurement': 'people',
                'capture_minute': minute,
                'chunk_index': chunk_index,
                'scope': scope,
                'timestamp': timestamp or datetime.now(timezone.utc).isoformat(),
            },
        },
        targets_entity_id: {
            'state': ' | '.join(f"T{item['target_id']}:({item['x']},{item['y']})" for item in target_tuples) or 'none',
            'attributes': {
                'friendly_name': 'Thoth Target Coordinates',
                'targets': target_tuples,
                'capture_minute': minute,
                'chunk_index': chunk_index,
                'scope': scope,
                'timestamp': timestamp or datetime.now(timezone.utc).isoformat(),
            },
        },
        labels_entity_id: {
            'state': ', '.join(str(label) for label in (labels or [])) or 'none',
            'attributes': {
                'friendly_name': f'Thoth {scope.title()} Labels',
                'labels': list(labels or []),
                'capture_minute': minute,
                'chunk_index': chunk_index,
                'scope': scope,
                'timestamp': timestamp or datetime.now(timezone.utc).isoformat(),
            },
        },
        zones_entity_id: {
            'state': ', '.join(active_zones) or 'none',
            'attributes': {
                'friendly_name': f'Thoth {scope.title()} Active Zones',
                'active_zones': active_zones,
                'capture_minute': minute,
                'chunk_index': chunk_index,
                'scope': scope,
                'timestamp': timestamp or datetime.now(timezone.utc).isoformat(),
            },
        },
        activity_entity_id: {
            'state': str((activity or {}).get('state') or ({'green': 'occupied', 'yellow': 'intermediate'}.get(payload['attributes']['classification'], 'empty'))),
            'attributes': {
                'friendly_name': f'Thoth {scope.title()} Human Activity',
                'labels': list(activity_labels or []),
                'zones': active_zones,
                'capture_minute': minute,
                'chunk_index': chunk_index,
                'scope': scope,
                **(activity if isinstance(activity, dict) else {}),
                'timestamp': timestamp or datetime.now(timezone.utc).isoformat(),
            },
        },
    }
    try:
        headers = {"Authorization": f"Bearer {config['token']}", "Content-Type": "application/json"}
        for destination, entity_payload in {entity_id: payload, **companion_payloads}.items():
            response = requests.post(
                f"{config['base_url']}/api/states/{destination}",
                json=entity_payload,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
        return _record_status({
            "success": True,
            "status": "published",
            "entity_id": entity_id,
            "entity_ids": [entity_id, region_entity_id, people_entity_id, targets_entity_id, labels_entity_id, zones_entity_id, activity_entity_id],
            "state": payload["state"],
            "published_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        logger.warning("Home Assistant occupancy publish failed: %s", exc)
        return _record_status({"success": False, "status": "error", "error": str(exc)})


def test_home_assistant_connection(timeout: float = 5.0) -> Dict[str, Any]:
    config = load_home_assistant_config(include_token=True)
    if not config.get("token"):
        return _record_status({"success": False, "status": "not_configured", "message": "A Home Assistant token is required."})
    try:
        response = requests.get(
            f"{config['base_url']}/api/",
            headers={"Authorization": f"Bearer {config['token']}"},
            timeout=timeout,
        )
        response.raise_for_status()
        return _record_status({"success": True, "status": "connected", "base_url": config["base_url"]})
    except Exception as exc:
        return _record_status({"success": False, "status": "error", "message": str(exc)})


PublishCallback = Callable[[Dict[str, Any]], None]


class HomeAssistantPublisher:
    """A single bounded publisher worker; capture and analysis never wait on HTTP."""

    def __init__(self, max_queue: int = 12, retry_delays: tuple[float, ...] = (0.5, 1.0, 2.0)):
        self._queue: queue.Queue[Optional[tuple[Dict[str, Any], str, Dict[str, Any], Optional[PublishCallback]]]] = queue.Queue(maxsize=max_queue)
        self._retry_delays = retry_delays
        self._thread = threading.Thread(target=self._run, name="HomeAssistantPublisher", daemon=True)
        self._thread.start()

    def submit(self, occupancy: Dict[str, Any], minute: str, callback: Optional[PublishCallback] = None, **metadata: Any) -> bool:
        try:
            self._queue.put_nowait((dict(occupancy), minute, metadata, callback))
            return True
        except queue.Full:
            result = {"success": False, "status": "queue_full", "error": "Home Assistant publisher queue is full"}
            if callback:
                callback(result)
            return False

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                occupancy, minute, metadata, callback = job
                result = publish_occupancy(occupancy, minute, **metadata)
                for delay in self._retry_delays:
                    if result.get("success") or result.get("status") in {"disabled", "not_configured"}:
                        break
                    time.sleep(delay)
                    result = publish_occupancy(occupancy, minute, **metadata)
                if callback:
                    try:
                        callback(result)
                    except Exception:
                        logger.exception("Home Assistant publish callback failed")
            finally:
                self._queue.task_done()

    def drain(self) -> None:
        self._queue.join()


_publisher: Optional[HomeAssistantPublisher] = None
_publisher_lock = threading.Lock()


def get_home_assistant_publisher() -> HomeAssistantPublisher:
    global _publisher
    with _publisher_lock:
        if _publisher is None:
            _publisher = HomeAssistantPublisher()
        return _publisher
