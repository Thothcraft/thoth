"""Known BLE device map — the node's enrolled/bonded identities.

Records are keyed by the *node-scoped* address hash (HMAC of the raw
MAC under ``ble_salt``), so the store is stable across restarts and the
raw MAC stays local. ``subject`` is what leaves the node in
observations — ``device:<uuid>`` for enrolled devices — never a MAC.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


class KnownDeviceStore:
    """JSON-backed map: addr-hash → enrolled device record."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text())
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=2))
            try:
                import os
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        except OSError:
            pass

    # -- CRUD -------------------------------------------------------------------
    def enroll(self, addr_hash: str, address: str, *,
               kind: str = "device", name: Optional[str] = None,
               device_id: Optional[str] = None,
               attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Bind an addr-hash to a stable internal device identity."""
        with self._lock:
            rec = self._data.get(addr_hash) or {}
            rec.update({
                "device_id": device_id or rec.get("device_id")
                             or f"device:{uuid.uuid4()}",
                "address": address,           # local-only — never uplinked
                "kind": rec.get("kind") or kind,
                "name": name or rec.get("name"),
                "attributes": dict(rec.get("attributes") or {},
                                   **(attributes or {})),
                "enrolled_at": rec.get("enrolled_at") or time.time(),
                "last_seen": rec.get("last_seen"),
            })
            self._data[addr_hash] = rec
            self._save()
            return dict(rec)

    def unenroll(self, addr_hash: str) -> bool:
        with self._lock:
            if self._data.pop(addr_hash, None) is None:
                return False
            self._save()
            return True

    def get(self, addr_hash: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            rec = self._data.get(addr_hash)
            return dict(rec) if rec else None

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._data.values()]

    def touch(self, addr_hash: str, ts: Optional[float] = None) -> None:
        """Refresh last_seen for an enrolled device (cheap path, still
        serialized — enrollment state drives subject identity)."""
        with self._lock:
            rec = self._data.get(addr_hash)
            if rec is not None:
                rec["last_seen"] = ts if ts is not None else time.time()
                self._save()


__all__ = ["KnownDeviceStore"]
