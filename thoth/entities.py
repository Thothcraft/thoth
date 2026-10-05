"""Entity + relationship store — persons, devices, spaces and the
edges between them ("wears", "owns", "located_in").

Backed by ``~/.thoth/entities.json``. Ids are namespaced keys
(``person:<uuid>``, ``device:<uuid>``, ``space:<slug>``) — the same
keyspace observation ``subject``/``entity_id`` use, so evidence and
identity join without translation.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


class EntityStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text())
            if isinstance(raw, dict) and "entities" in raw:
                return raw
        except Exception:
            pass
        return {"entities": {}, "relations": []}

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

    # -- entities ----------------------------------------------------------------
    def upsert(self, entity_id: Optional[str] = None,
               type: str = "device", name: Optional[str] = None,
               attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create or update an entity; returns the record."""
        with self._lock:
            ent = self._data["entities"].get(entity_id) if entity_id \
                else None
            if ent is None:
                entity_id = entity_id or f"{type}:{uuid.uuid4()}"
                ent = {"id": entity_id, "type": type,
                       "created_at": time.time()}
                self._data["entities"][entity_id] = ent
            if name is not None:
                ent["name"] = name
            if attributes:
                ent.setdefault("attributes", {}).update(attributes)
            self._save()
            return dict(ent)

    def remove(self, entity_id: str) -> bool:
        with self._lock:
            if self._data["entities"].pop(entity_id, None) is None:
                return False
            self._data["relations"] = [
                r for r in self._data["relations"]
                if r["from"] != entity_id and r["to"] != entity_id]
            self._save()
            return True

    def get(self, entity_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            ent = self._data["entities"].get(entity_id)
            return dict(ent) if ent else None

    def list(self, type: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(e) for e in self._data["entities"].values()
                    if type is None or e["type"] == type]

    # -- relations -------------------------------------------------------------
    def relate(self, from_id: str, to_id: str, rel: str) -> Dict[str, Any]:
        """from_id --rel--> to_id (e.g. person:x wears device:y)."""
        with self._lock:
            for r in self._data["relations"]:
                if (r["from"], r["to"], r["rel"]) == (from_id, to_id, rel):
                    return dict(r)
            rec = {"from": from_id, "to": to_id, "rel": rel,
                   "since": time.time()}
            self._data["relations"].append(rec)
            self._save()
            return dict(rec)

    def unrelate(self, from_id: str, to_id: str, rel: str) -> bool:
        with self._lock:
            before = len(self._data["relations"])
            self._data["relations"] = [
                r for r in self._data["relations"]
                if (r["from"], r["to"], r["rel"]) != (from_id, to_id, rel)]
            if len(self._data["relations"]) == before:
                return False
            self._save()
            return True

    def relations(self, entity_id: Optional[str] = None,
                  rel: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._data["relations"]
                    if (entity_id is None
                        or r["from"] == entity_id or r["to"] == entity_id)
                    and (rel is None or r["rel"] == rel)]

    def document(self) -> Dict[str, Any]:
        return {"entities": self.list(), "relations": self.relations()}


__all__ = ["EntityStore"]
