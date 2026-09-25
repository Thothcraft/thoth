"""Room layout — the synced source of truth (CONTRACT §1.2).

``GET /api/v1/room`` returns the ``room/v1`` document; ``PUT`` replaces it
(version-stamped via ``updated_at``) and notifies the daemon through
``on_change`` so the new layout is pushed to Brain and broadcast to
subscribers.

The node stays authoritative: portal edits arrive as relayed PUTs and are
persisted to ``~/.thoth/room.json`` here, never stored only remotely.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .settings import config_dir

logger = logging.getLogger(__name__)

DEFAULT_ROOM: Dict[str, Any] = {
    "format": "room/v1",
    "room_id": "",
    "name": "",
    "dims": {"w": 6.0, "d": 4.0, "h": 2.6},
    "walls": [],
    "furniture": [],
    "devices": [],
    "updated_at": 0.0,
}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _vec3(value: Any, default=(0.0, 0.0, 0.0)) -> list:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return [_num(value[0]), _num(value[1]), _num(value[2])]
    return list(default)


def normalize_room(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce an incoming room body into a well-formed ``room/v1`` doc.

    Unknown keys are dropped; missing sections take defaults. Numeric
    fields are cast defensively so a sloppy editor can't wedge the
    renderer with strings.
    """
    doc = doc if isinstance(doc, dict) else {}
    out = dict(DEFAULT_ROOM)
    out["room_id"] = str(doc.get("room_id") or "")
    out["name"] = str(doc.get("name") or "")
    dims = doc.get("dims") if isinstance(doc.get("dims"), dict) else {}
    out["dims"] = {"w": _num(dims.get("w"), DEFAULT_ROOM["dims"]["w"]),
                   "d": _num(dims.get("d"), DEFAULT_ROOM["dims"]["d"]),
                   "h": _num(dims.get("h"), DEFAULT_ROOM["dims"]["h"])}

    walls = []
    for w in doc.get("walls") or []:
        if not isinstance(w, dict):
            continue
        walls.append({"p": _vec3(w.get("p")), "s": _vec3(w.get("s"))})
    out["walls"] = walls

    furniture = []
    for f in doc.get("furniture") or []:
        if not isinstance(f, dict):
            continue
        furniture.append({
            "id": str(f.get("id") or f"furniture-{len(furniture)}"),
            "type": str(f.get("type") or "table"),
            "pos": _vec3(f.get("pos")),
            "rot_y": _num(f.get("rot_y")),
            "dims": _vec3(f.get("dims"), (0.8, 0.5, 0.8)),
        })
    out["furniture"] = furniture

    devices = []
    for dev in doc.get("devices") or []:
        if not isinstance(dev, dict):
            continue
        sensors = []
        for s in dev.get("sensors") or []:
            if not isinstance(s, dict):
                continue
            sensors.append({
                "type": str(s.get("type") or "radar"),
                "pos": _vec3(s.get("pos")),
                "rot_y": _num(s.get("rot_y")),
                "tilt": _num(s.get("tilt")),
                "fov_deg": _num(s.get("fov_deg"), 60.0),
                "range_m": _num(s.get("range_m"), 6.0),
            })
        devices.append({
            "device_id": str(dev.get("device_id") or ""),
            "pos": _vec3(dev.get("pos")),
            "rot_y": _num(dev.get("rot_y")),
            "mount": str(dev.get("mount") or "wall"),
            "sensors": sensors,
        })
    out["devices"] = devices
    out["updated_at"] = _num(doc.get("updated_at"), 0.0)
    return out


class RoomManager:
    """Persists and versions the node's room document."""

    def __init__(self, path: Optional[Path] = None,
                 on_change: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.path = path or (config_dir() / "room.json")
        self._on_change = on_change
        self._doc = self._load()

    def _load(self) -> Dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text())
            if isinstance(raw, dict) and raw:
                return normalize_room(raw)
        except Exception as exc:
            logger.debug("room.json unreadable: %s", exc)
        return dict(DEFAULT_ROOM)

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._doc, indent=2))
        except Exception as exc:
            logger.warning("room.json save failed: %s", exc)

    def document(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._doc))

    def put(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Replace the room document — bumps ``updated_at`` and emits
        ``room_changed`` through the on_change callback."""
        self._doc = normalize_room(doc)
        self._doc["updated_at"] = time.time()
        self._save()
        if self._on_change is not None:
            try:
                self._on_change(self.document())
            except Exception as exc:
                logger.debug("room on_change failed: %s", exc)
        return self.document()


__all__ = ["RoomManager", "DEFAULT_ROOM", "normalize_room"]
