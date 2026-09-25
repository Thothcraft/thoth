"""Device metadata — inferred + manual halves (CONTRACT §1.1).

``GET /api/v1/metadata`` returns::

    {"inferred": {location, activity, battery, application},
     "manual":   {room_name, friendly_name, room_id}}

The **manual** half is user-set via PUT and persisted to
``~/.thoth/metadata.json``. The **inferred** half is refreshed
periodically by the daemon (``refresh_inferred``): geo via the node's own
public egress IP, activity from recent predictions, battery via
sysfs/Win32, foreground app where the OS exposes it — ``null``/empty when
the OS does not allow it.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from .settings import config_dir

logger = logging.getLogger(__name__)

MANUAL_KEYS = ("room_name", "friendly_name", "room_id")

_DEFAULT_INFERRED: Dict[str, Any] = {
    "location": {"lat": 0.0, "lon": 0.0, "postal_code": "", "city": "",
                 "updated_at": 0.0},
    "activity": {"kind": "idle", "confidence": 0.0, "updated_at": 0.0},
    "battery": {"percent": None, "charging": None, "updated_at": 0.0},
    "application": {"foreground": "", "platform": "", "updated_at": 0.0},
}

# Labels that mean "nobody here" rather than a presence kind.
_IDLE_LABELS = {"", "empty", "idle", "quiet", "none", "unknown",
                "unoccupied", "no_face", "vacant"}

# Activity is computed from predictions within this window.
_ACTIVITY_WINDOW_S = 30.0


def _platform() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform.startswith("linux"):
        import os
        if os.getenv("ANDROID_ROOT") or os.getenv("ANDROID_DATA"):
            return "android"
        return "linux"
    return "linux"


def _probe_battery() -> Dict[str, Any]:
    """Best-effort battery read — nulls where the OS doesn't expose it."""
    base = Path("/sys/class/power_supply")
    try:
        for bat in sorted(base.glob("BAT*")):
            percent = float((bat / "capacity").read_text().strip())
            status = (bat / "status").read_text().strip().lower()
            return {"percent": percent,
                    "charging": status in ("charging", "full")}
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class _PS(ctypes.Structure):
                _fields_ = [
                    ("ACLineStatus", ctypes.c_ubyte),
                    ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte),
                    ("Reserved1", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong),
                    ("BatteryFullLifeTime", ctypes.c_ulong),
                ]

            ps = _PS()
            if ctypes.windll.kernel32.GetSystemPowerStatus(  # type: ignore[attr-defined]
                    ctypes.byref(ps)):
                pct = (None if ps.BatteryLifePercent in (255, 0) and
                       ps.BatteryFlag == 128      # 128 = no system battery
                       else (None if ps.BatteryLifePercent == 255
                             else float(ps.BatteryLifePercent)))
                charging = (None if pct is None else
                            bool(ps.ACLineStatus) and pct < 100.0)
                return {"percent": pct, "charging": charging}
        except Exception:
            pass
    return {"percent": None, "charging": None}


def _probe_foreground() -> str:
    """Foreground window title — Win32 only; empty string elsewhere."""
    if sys.platform == "win32":
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetForegroundWindow()  # type: ignore[attr-defined]
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)  # type: ignore[attr-defined]
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)  # type: ignore[attr-defined]
            return buf.value or ""
        except Exception:
            return ""
    return ""


def _activity_kind(label: str, confidence: float) -> str:
    l = label.strip().lower()
    if l in _IDLE_LABELS or confidence <= 0.0:
        return "idle"
    if any(k in l for k in ("motion", "moving", "walk", "fall", "gesture")):
        return "motion"
    if any(k in l for k in ("occup", "crowd", "meeting")):
        return "occupied"
    return "presence" if confidence >= 0.4 else "idle"


class MetadataManager:
    """Inferred + manual device metadata (CONTRACT §1.1)."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or (config_dir() / "metadata.json")
        self._inferred: Dict[str, Any] = json.loads(
            json.dumps(_DEFAULT_INFERRED))
        self._manual: Dict[str, str] = {k: "" for k in MANUAL_KEYS}
        self._load()

    # -- persistence -----------------------------------------------------------
    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text())
        except Exception:
            return
        manual = raw.get("manual") if isinstance(raw, dict) else None
        if isinstance(manual, dict):
            for k in MANUAL_KEYS:
                if manual.get(k) is not None:
                    self._manual[k] = str(manual[k])

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(
                {"format": "thoth-metadata/v1", "manual": self._manual},
                indent=2))
        except Exception as exc:
            logger.warning("metadata save failed: %s", exc)

    # -- document ----------------------------------------------------------------
    def document(self) -> Dict[str, Any]:
        return {"inferred": json.loads(json.dumps(self._inferred)),
                "manual": dict(self._manual)}

    def inferred(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._inferred))

    # -- manual half ---------------------------------------------------------------
    def set_manual(self, fields: Mapping[str, Any]) -> Dict[str, Any]:
        """PUT /api/v1/metadata — body is a subset of the manual section."""
        for k in MANUAL_KEYS:
            if fields.get(k) is not None:
                self._manual[k] = str(fields[k])
        self._save()
        return self.document()

    # -- inferred half --------------------------------------------------------------
    def refresh_inferred(self,
                         predictions: Iterable[Mapping[str, Any]] = (),
                         now: Optional[float] = None) -> Dict[str, Any]:
        """Recompute all inferred sections. Returns sections that changed.

        Probing is best-effort: each probe failure leaves the previous
        values untouched (or the null defaults) — an offline node still
        serves a complete, honest document.
        """
        now = time.time() if now is None else now
        changed: Dict[str, Any] = {}

        # location — public_geo() caches for an hour internally.
        try:
            from whispy.geo import public_geo
            geo = public_geo() or {}
        except Exception:
            geo = {}
        if geo:
            loc = {"lat": float(geo.get("latitude") or 0.0),
                   "lon": float(geo.get("longitude") or 0.0),
                   "postal_code": str(geo.get("postal_code") or ""),
                   "city": str(geo.get("city") or ""),
                   "updated_at": now}
            if loc != self._inferred["location"]:
                changed["location"] = loc
                self._inferred["location"] = loc

        # activity — latest confident prediction within the window.
        recent = [p for p in list(predictions)[-50:]
                  if now - float(p.get("timestamp") or 0.0)
                  <= _ACTIVITY_WINDOW_S]
        if recent:
            last = max(recent, key=lambda p: float(p.get("timestamp") or 0.0))
            label = str(last.get("label") or "")
            conf = float(last.get("confidence") or 0.0)
            act = {"kind": _activity_kind(label, conf),
                   "confidence": conf, "updated_at": now}
        else:
            act = {"kind": "idle", "confidence": 0.0, "updated_at": now}
        if act["kind"] != self._inferred["activity"]["kind"] or \
                abs(act["confidence"]
                    - self._inferred["activity"]["confidence"]) > 0.05:
            changed["activity"] = act
            self._inferred["activity"] = act

        # battery — nulls when the machine has none.
        bat = _probe_battery()
        bat["updated_at"] = now
        prev = self._inferred["battery"]
        if bat["percent"] != prev["percent"] or \
                bat["charging"] != prev["charging"]:
            changed["battery"] = bat
            self._inferred["battery"] = bat

        # application — platform is always knowable; foreground is
        # best-effort (empty where the OS hides it).
        app = {"foreground": _probe_foreground(), "platform": _platform(),
               "updated_at": now}
        if app["foreground"] != self._inferred["application"]["foreground"] or \
                app["platform"] != self._inferred["application"]["platform"]:
            changed["application"] = app
            self._inferred["application"] = app

        return changed


__all__ = ["MetadataManager", "MANUAL_KEYS"]
