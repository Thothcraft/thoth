"""Publish completed-minute occupancy to Home Assistant's REST API."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

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


def load_home_assistant_config(include_token: bool = False) -> Dict[str, Any]:
    config = dict(DEFAULTS)
    try:
        if CONFIG_PATH.exists():
            loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config.update(loaded)
    except Exception as exc:
        logger.warning("Unable to load Home Assistant settings: %s", exc)
    config["configured"] = bool(config.get("token"))
    if not include_token:
        config.pop("token", None)
    return config


def save_home_assistant_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    config = load_home_assistant_config(include_token=True)
    if "enabled" in updates:
        config["enabled"] = bool(updates["enabled"])
    if updates.get("base_url"):
        config["base_url"] = str(updates["base_url"]).strip().rstrip("/")
    if updates.get("entity_id"):
        entity_id = str(updates["entity_id"]).strip().lower()
        config["entity_id"] = entity_id if entity_id.startswith("binary_sensor.") else f"binary_sensor.{entity_id}"
    if updates.get("token"):
        config["token"] = str(updates["token"]).strip()
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps({key: config[key] for key in DEFAULTS}, indent=2), encoding="utf-8")
    return load_home_assistant_config()


def publish_occupancy(occupancy: Dict[str, Any], minute: str) -> Dict[str, Any]:
    config = load_home_assistant_config(include_token=True)
    if not config.get("enabled"):
        return {"success": False, "status": "disabled"}
    if not config.get("token"):
        return {"success": False, "status": "not_configured"}

    detected = max(0, int(occupancy.get("detected_frames") or 0))
    total = max(0, int(occupancy.get("evaluated_frames") or 0))
    label = str(occupancy.get("label") or "empty")
    entity_id = str(config["entity_id"])
    payload = {
        "state": "on" if label == "occupied" else "off",
        "attributes": {
            "friendly_name": "Thoth Occupancy",
            "device_class": "occupancy",
            "label": label,
            "capture_minute": minute,
            "detected_frames": detected,
            "evaluated_frames": total,
            "detected_percent": round(detected * 100 / total, 2) if total else 0.0,
            "threshold_percent": occupancy.get("threshold_percent", 50.0),
        },
    }
    try:
        response = requests.post(
            f"{config['base_url']}/api/states/{entity_id}",
            json=payload,
            headers={"Authorization": f"Bearer {config['token']}", "Content-Type": "application/json"},
            timeout=10,
        )
        response.raise_for_status()
        return {"success": True, "status": "published", "entity_id": entity_id, "state": payload["state"]}
    except Exception as exc:
        logger.warning("Home Assistant occupancy publish failed: %s", exc)
        return {"success": False, "status": "error", "error": str(exc)}
