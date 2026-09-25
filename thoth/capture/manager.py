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

    # -- annotation ------------------------------------------------------------
    def _write_manifest(self, rec: Dict[str, Any]) -> None:
        (self._dir(rec["id"]) / "manifest.json").write_text(
            json.dumps(rec, indent=2))

    def add_label(self, capture_id: str, label: str, source: str = "manual",
                  start: Optional[float] = None, end: Optional[float] = None,
                  confidence: Optional[float] = None,
                  model_id: str = "") -> Optional[Dict[str, Any]]:
        """Append one label to the capture manifest.

        ``source`` records provenance: ``manual`` from the dashboard,
        ``auto`` from a model prediction. ``start``/``end`` scope the
        label inside the capture; omitting them means "whole capture".
        """
        rec = self.get(capture_id)
        if rec is None:
            return None
        entry = {
            "label": str(label),
            "source": source,
            "start": float(start) if start is not None
                     else float(rec.get("started_at") or 0.0),
            "end": float(end) if end is not None
                   else float(rec.get("stopped_at") or time.time()),
            "confidence": confidence,
            "model_id": model_id,
            "at": time.time(),
        }
        rec.setdefault("labels", []).append(entry)
        self._write_manifest(rec)
        return entry

    def clear_labels(self, capture_id: str,
                     source: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Remove labels; ``source`` filters which provenance to drop."""
        rec = self.get(capture_id)
        if rec is None:
            return None
        labels = rec.get("labels") or []
        rec["labels"] = [l for l in labels
                         if source and l.get("source") != source]
        self._write_manifest(rec)
        return rec

    def autolabel(self, capture_id: str,
                  predictions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Label the capture from predictions inside its time span.

        Every prediction whose timestamp falls in [started_at, stopped_at]
        becomes a ``source="auto"`` label. Idempotent: existing auto labels
        are replaced, manual labels are preserved.
        """
        rec = self.get(capture_id)
        if rec is None:
            return None
        t0 = float(rec.get("started_at") or 0.0)
        t1 = float(rec.get("stopped_at") or time.time())
        manual = [l for l in (rec.get("labels") or [])
                  if l.get("source") != "auto"]
        auto = []
        for p in predictions:
            try:
                ts = float(p.get("timestamp") or p.get("at") or 0.0)
            except (TypeError, ValueError):
                continue
            if not (t0 <= ts <= t1):
                continue
            auto.append({
                "label": str(p.get("label") or p.get("class") or "?"),
                "source": "auto",
                "start": ts, "end": ts,
                "confidence": p.get("confidence"),
                "model_id": str(p.get("runtime_model_id")
                                or p.get("model_id") or ""),
                "at": ts,
            })
        rec["labels"] = manual + auto
        self._write_manifest(rec)
        return rec

    # -- lifecycle --------------------------------------------------------------
    def delete(self, capture_id: str) -> bool:
        """Remove a capture directory entirely. Active captures refuse."""
        if capture_id in self._active:
            return False
        d = self._dir(capture_id)
        if not d.is_dir():
            return False
        import shutil
        shutil.rmtree(d)
        return True

    def export(self, capture_id: str) -> Optional[Path]:
        """Write ``<capture_id>.zip`` next to the capture dir; return its path."""
        import zipfile
        d = self._dir(capture_id)
        if not d.is_dir():
            return None
        # Refresh the manifest so counts/labels are current in the bundle.
        rec = self.get(capture_id)
        if rec is not None:
            self._write_manifest(rec)
        out = self.root / f"{capture_id}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(d.iterdir()):
                zf.write(f, arcname=f"{capture_id}/{f.name}")
        return out
