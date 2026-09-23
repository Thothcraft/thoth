"""Capture lifecycle — a logical session, not loose files (§22).

A capture records timestamped SensorSamples for a set of sensors into
``~/.thoth/captures/<capture_id>/`` as JSONL, with a manifest describing
the session.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..settings import config_dir


class CaptureManager:
    def __init__(self, root: Optional[Path] = None):
        self.root = root or (config_dir() / "captures")
        self.root.mkdir(parents=True, exist_ok=True)
        self._active: Dict[str, Dict[str, Any]] = {}

    def _dir(self, capture_id: str) -> Path:
        return self.root / capture_id

    @staticmethod
    def resolve_sensors(sensors: Optional[List[str]],
                        available: List[str]) -> List[str]:
        """Resolve a requested sensor list against the available ids.

        An omitted/empty request means "all available sensors" (the CLI's
        documented default). An explicit id that is not available raises
        ``ValueError`` so a typo never silently records nothing.
        """
        if not sensors:
            return list(available)
        unknown = [s for s in sensors if s not in available]
        if unknown:
            raise ValueError(
                f"unknown sensor id(s) {unknown}; available: {available}")
        return list(sensors)

    def start(self, device_id: str, sensors: Optional[List[str]],
              available: Optional[List[str]] = None) -> Dict[str, Any]:
        if available is not None:
            sensors = self.resolve_sensors(sensors, available)
        sensors = list(sensors or [])
        capture_id = uuid.uuid4().hex[:12]
        rec = {
            "id": capture_id,
            "device_id": device_id,
            "started_at": time.time(),
            "state": "active",
            "sensors": sensors,
            "sample_counts": {s: 0 for s in sensors},
        }
        self._dir(capture_id).mkdir(parents=True, exist_ok=True)
        (self._dir(capture_id) / "manifest.json").write_text(
            json.dumps(rec, indent=2))
        self._active[capture_id] = rec
        return rec

    def record(self, capture_id: str, sample: Any) -> None:
        rec = self._active.get(capture_id)
        if rec is None:
            return
        sid = sample.sensor_id
        with (self._dir(capture_id) / f"{sid}.jsonl").open("a") as fh:
            fh.write(json.dumps(sample.to_dict()) + "\n")
        rec["sample_counts"][sid] = rec["sample_counts"].get(sid, 0) + 1

    def stop(self, capture_id: str) -> Optional[Dict[str, Any]]:
        rec = self._active.pop(capture_id, None)
        if rec is None:
            manifest = self._dir(capture_id) / "manifest.json"
            if manifest.exists():
                rec = json.loads(manifest.read_text())
            else:
                return None
        rec["state"] = "stopped"
        rec["stopped_at"] = time.time()
        (self._dir(capture_id) / "manifest.json").write_text(
            json.dumps(rec, indent=2))
        return rec

    def list(self) -> List[Dict[str, Any]]:
        out = list(self._active.values())
        for d in sorted(self.root.iterdir()):
            manifest = d / "manifest.json"
            if d.is_dir() and manifest.exists():
                try:
                    rec = json.loads(manifest.read_text())
                except Exception:
                    continue
                if rec["id"] not in self._active:
                    out.append(rec)
        return out

    def get(self, capture_id: str) -> Optional[Dict[str, Any]]:
        if capture_id in self._active:
            return self._active[capture_id]
        manifest = self._dir(capture_id) / "manifest.json"
        if manifest.exists():
            return json.loads(manifest.read_text())
        return None
