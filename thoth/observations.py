"""Normalized observation envelope + uplink spool (observation-contract-v1).

The edge side of the canonical flow:

    sources → observations → predictions/evidence → context state
            → context events → applications

An :class:`Observation` is the *curated* uplink unit — low-rate,
context-relevant, JSON-safe. High-rate raw data (camera/radar frames,
raw IMU samples) never enters the spool; it stays on the capture/tail
path (``/api/v1/sources/{id}/observations``).

:class:`ObservationSpool` gives producers a bounded, restart-durable
outbox. ``ThothDaemon._tick`` drains it into ``observation_batch`` frames
on the Brain WS channel; Brain maps each item to a ``ContextEvidence``
row keyed by ``external_id="obs:<observation_id>"``, so redelivery after
a reconnect is idempotent.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional

logger = logging.getLogger(__name__)

OBSERVATION_ENVELOPE = "observation/v1"
BATCH_FRAME_TYPE = "observation_batch"
BATCH_MAX_ITEMS = 200
SPOOL_FILE = "observations.jsonl"
SPOOL_MAX_BYTES = 10 * 1024 * 1024          # 10 MB disk spool
MEM_MAX_ITEMS = 4096                         # in-memory pending window

_SCHEMA_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)*\.v\d+$")


class ObservationError(ValueError):
    """Raised when an envelope violates observation/v1."""


class Observation:
    """One normalized physical-world fact (contract §1).

    Producers set ``schema``, ``source_id``, ``value`` and normally a
    ``subject``; ``observation_id``, ``timestamp`` and ``provenance``
    defaults are filled on construction. ``validate()`` enforces the
    frozen wire shape — producers should validate before spooling.
    """

    __slots__ = (
        "schema", "observation_id", "batch_id", "timestamp", "source_id",
        "subject", "value", "units", "confidence", "sequence",
        "provenance", "sync",
    )

    def __init__(self, schema: str, source_id: str, value: Any, *,
                 subject: Optional[str] = None,
                 observer: Optional[str] = None,
                 timestamp: Optional[float] = None,
                 observation_id: Optional[str] = None,
                 batch_id: Optional[str] = None,
                 units: Optional[Mapping[str, str]] = None,
                 confidence: Optional[float] = None,
                 sequence: Optional[int] = None,
                 provenance: Optional[Mapping[str, Any]] = None,
                 sync: Optional[Mapping[str, Any]] = None):
        self.schema = str(schema)
        self.source_id = str(source_id)
        self.value = value
        self.subject = str(subject) if subject else None
        self.timestamp = float(timestamp if timestamp is not None
                               else time.time())
        self.observation_id = str(observation_id or uuid.uuid4())
        self.batch_id = str(batch_id) if batch_id else None
        self.units = dict(units) if units else None
        self.confidence = (float(confidence)
                           if confidence is not None else None)
        self.sequence = int(sequence) if sequence is not None else None
        self.provenance = dict(provenance or {})
        if observer is not None:
            self.provenance.setdefault("observer", str(observer))
        self.sync = dict(sync) if sync else None

    # -- validation -----------------------------------------------------------
    def validate(self) -> "Observation":
        if not _SCHEMA_RE.match(self.schema):
            raise ObservationError(
                f"schema {self.schema!r} must match <domain>.<measure>.v<N>")
        if not self.source_id:
            raise ObservationError("source_id is required")
        if self.timestamp <= 0:
            raise ObservationError("timestamp must be epoch seconds > 0")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ObservationError("confidence must be within [0,1]")
        if self.value is None:
            raise ObservationError("value is required")
        # JSON-safety is a wire requirement — raise early on the edge.
        try:
            json.dumps(self.value)
        except (TypeError, ValueError) as exc:
            raise ObservationError(f"value is not JSON-safe: {exc}")
        return self

    # -- wire -----------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "schema": self.schema,
            "observation_id": self.observation_id,
            "timestamp": self.timestamp,
            "source_id": self.source_id,
            "value": self.value,
            "provenance": self.provenance,
        }
        if self.batch_id:
            out["batch_id"] = self.batch_id
        if self.subject:
            out["subject"] = self.subject
        if self.units:
            out["units"] = self.units
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.sequence is not None:
            out["sequence"] = self.sequence
        if self.sync:
            out["sync"] = self.sync
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Observation":
        prov = dict(data.get("provenance") or {})
        return cls(
            schema=str(data.get("schema") or ""),
            source_id=str(data.get("source_id") or ""),
            value=data.get("value"),
            subject=data.get("subject"),
            observer=prov.get("observer"),
            timestamp=data.get("timestamp"),
            observation_id=data.get("observation_id"),
            batch_id=data.get("batch_id"),
            units=data.get("units"),
            confidence=data.get("confidence"),
            sequence=data.get("sequence"),
            provenance=prov,
            sync=data.get("sync"),
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"Observation({self.schema} src={self.source_id} "
                f"sub={self.subject} id={self.observation_id[:8]})")


def build_batch(items: Iterable[Mapping[str, Any]],
                batch_id: Optional[str] = None) -> Dict[str, Any]:
    """Wrap validated observation dicts in the ``observation_batch`` frame
    (contract §2). Stamps ``batch_id`` onto every item."""
    bid = str(batch_id or uuid.uuid4())
    out = []
    for item in items:
        d = dict(item)
        d.setdefault("batch_id", bid)
        out.append(d)
    return {"type": BATCH_FRAME_TYPE, "id": bid, "ts": time.time(),
            "items": out}


class ObservationSpool:
    """Bounded, restart-durable observation outbox.

    Items live in an in-memory deque (cap ``mem_max``) mirrored to a JSONL
    file (cap ``max_bytes``). ``pending()``/`ack()` implement peek-then-
    acknowledge semantics: a batch that never reached the socket stays
    pending and is re-sent after reconnect — Brain dedupes on
    ``observation_id`` so duplicates are harmless.

    Overflow drops the *oldest* pending items and counts them; the next
    flush emits a ``observation.dropped.v1`` marker so the loss is
    visible upstream instead of silent.
    """

    def __init__(self, directory: Path, *,
                 mem_max: int = MEM_MAX_ITEMS,
                 max_bytes: int = SPOOL_MAX_BYTES):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / SPOOL_FILE
        self.mem_max = int(mem_max)
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._pending: Deque[Dict[str, Any]] = deque()
        self._dropped = 0
        self._bytes_appended = 0
        self._load()

    # -- persistence ----------------------------------------------------------
    def _load(self) -> None:
        """Rehydrate pending items on daemon start (oldest dropped first
        when the spool exceeds the memory window)."""
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    self._pending.append(item)
                    while len(self._pending) > self.mem_max:
                        self._pending.popleft()
                        self._dropped += 1
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("observation spool unreadable: %s", exc)

    def _rewrite(self) -> None:
        """Compact the JSONL file to exactly the pending set."""
        tmp = self.path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                for item in self._pending:
                    fh.write(json.dumps(item, separators=(",", ":")))
                    fh.write("\n")
            tmp.replace(self.path)
            self._bytes_appended = self.path.stat().st_size
        except OSError as exc:
            logger.warning("observation spool rewrite failed: %s", exc)

    # -- producer API ---------------------------------------------------------
    def append(self, obs: Observation | Mapping[str, Any]) -> str:
        """Validate + enqueue one observation. Returns observation_id."""
        if isinstance(obs, Observation):
            item = obs.validate().to_dict()
        else:
            item = dict(obs)
        if not item.get("observation_id"):
            item["observation_id"] = str(uuid.uuid4())
        with self._lock:
            self._pending.append(item)
            while len(self._pending) > self.mem_max:
                self._pending.popleft()
                self._dropped += 1
            try:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(item, separators=(",", ":")))
                    fh.write("\n")
                self._bytes_appended = self.path.stat().st_size
            except OSError as exc:
                logger.warning("observation spool append failed: %s", exc)
            if self._bytes_appended > self.max_bytes:
                # Disk cap: shed oldest half, compact.
                shed = max(1, len(self._pending) // 2)
                for _ in range(min(shed, len(self._pending))):
                    self._pending.popleft()
                    self._dropped += 1
                self._rewrite()
        return str(item["observation_id"])

    # -- consumer API ---------------------------------------------------------
    def pending(self, limit: int = BATCH_MAX_ITEMS) -> List[Dict[str, Any]]:
        """Oldest-first view of pending items (peek — nothing removed)."""
        with self._lock:
            items = list(self._pending)[:limit]
        out: List[Dict[str, Any]] = []
        if self._dropped:
            marker = Observation(
                schema="observation.dropped.v1",
                source_id="node:spool",
                value={"dropped": self._dropped,
                       "reason": "spool_overflow"}).validate().to_dict()
            out.append(marker)
            with self._lock:
                self._dropped = 0
        out.extend(items)
        return out[:limit]

    def ack(self, observation_ids: Iterable[str]) -> int:
        """Remove delivered items. Returns count acked."""
        ids = set(observation_ids)
        if not ids:
            return 0
        with self._lock:
            before = len(self._pending)
            self._pending = deque(
                (i for i in self._pending
                 if i.get("observation_id") not in ids),
                maxlen=self.mem_max)
            removed = before - len(self._pending)
            if removed:
                self._rewrite()
        return removed

    def requeue_front(self, items: Iterable[Mapping[str, Any]]) -> None:
        """Return a failed batch to the front of the queue (socket died
        mid-send). Bounded the same way as ``append``."""
        with self._lock:
            for item in reversed(list(items)):
                self._pending.appendleft(dict(item))
            while len(self._pending) > self.mem_max:
                self._pending.pop()
                self._dropped += 1
            self._rewrite()

    def __len__(self) -> int:
        with self._lock:
            return len(self._pending)


__all__ = [
    "OBSERVATION_ENVELOPE", "BATCH_FRAME_TYPE", "BATCH_MAX_ITEMS",
    "Observation", "ObservationError", "ObservationSpool", "build_batch",
]
