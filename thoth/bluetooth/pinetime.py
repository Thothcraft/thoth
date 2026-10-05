"""InfiniTime/PineTime link — central-role motion subscription +
context push.

On connect the node subscribes to InfiniTime's motion service and
buffers raw x/y/z into decimated ``imu.window.v1`` observations (the
contract's uplink unit — never the raw stream). ``push_context`` writes
the current context state back over a context characteristic; the
firmware side that renders it is a separate deliverable.
"""

from __future__ import annotations

import json
import logging
import struct
import time
from typing import Any, Callable, Dict, List, Optional

from ..observations import Observation

logger = logging.getLogger(__name__)

# InfiniTime motion service + RawMotion characteristic (upstream UUIDs)
MOTION_SERVICE = "00030000-78fc-48fe-8e23-433b3a1942d0"
CHAR_RAW_MOTION = "00030002-78fc-48fe-8e23-433b3a1942d0"
# Node→watch context channel — requires matching firmware support.
CHAR_CONTEXT = "00030003-78fc-48fe-8e23-433b3a1942d0"

WINDOW_S = 2.0          # observation window length
MAX_SAMPLES = 32        # decimation bound — keeps windows JSON-small


class ImuWindowBuffer:
    """Accumulates motion samples; flushes a window every WINDOW_S."""

    def __init__(self, window_s: float = WINDOW_S):
        self.window_s = window_s
        self.xs: List[float] = []
        self.ys: List[float] = []
        self.zs: List[float] = []
        self.first_ts: Optional[float] = None
        self.last_ts: Optional[float] = None

    def add(self, x: float, y: float, z: float, ts: float) -> None:
        if self.first_ts is None:
            self.first_ts = ts
        self.xs.append(x)
        self.ys.append(y)
        self.zs.append(z)
        self.last_ts = ts

    def ready(self, now: float) -> bool:
        return self.first_ts is not None and \
            now - self.first_ts >= self.window_s

    def flush(self) -> Optional[Dict[str, Any]]:
        n = len(self.xs)
        if n == 0 or self.first_ts is None or self.last_ts is None:
            return None
        span = max(self.last_ts - self.first_ts, 1e-3)
        value = {
            "rate_hz": round(n / span, 1),
            "samples": n,
            "axes": {
                "x": self.xs[:MAX_SAMPLES],
                "y": self.ys[:MAX_SAMPLES],
                "z": self.zs[:MAX_SAMPLES],
            },
        }
        self.xs, self.ys, self.zs = [], [], []
        self.first_ts = self.last_ts = None
        return value


def parse_motion(data: bytes) -> Optional[tuple]:
    """InfiniTime RawMotion payload: little-endian int16 x,y,z (mg)."""
    if len(data) < 6:
        return None
    x, y, z = struct.unpack_from("<hhh", data, 0)
    return x / 1000.0, y / 1000.0, z / 1000.0   # → g


class PinetimeLink:
    """One enrolled watch: motion subscribe + context push, driven by a
    managed central session on the BLE subsystem's asyncio loop."""

    def __init__(self, ble: Any, address: str, subject: str,
                 emit: Callable[[Observation], Any],
                 source_id: str):
        self._ble = ble
        self.address = address
        self.subject = subject              # enrolled device:<uuid>
        self._emit = emit
        self.source_id = source_id          # ble:hci0
        self._buf = ImuWindowBuffer()
        self._seq = 0
        self._up = False
        self._attached = False

    # -- lifecycle ------------------------------------------------------------------
    def attach(self) -> bool:
        if self._attached:
            return True
        ok = self._ble.central_connect(
            self.address,
            subscriptions={CHAR_RAW_MOTION: self._on_motion},
            on_state=self._on_state)
        self._attached = ok
        return ok

    def detach(self) -> None:
        self._attached = False
        self._ble.central_disconnect(self.address)

    @property
    def connected(self) -> bool:
        return self._up

    def _on_state(self, address: str, up: bool) -> None:
        self._up = up
        if up:
            logger.info("watch link up: %s", self.subject)

    # -- motion ingest ----------------------------------------------------------------
    def _on_motion(self, data: bytes) -> None:
        """Notification callback — runs on the BLE loop thread."""
        parsed = parse_motion(data)
        if parsed is None:
            return
        now = time.time()
        self._buf.add(*parsed, ts=now)
        if not self._buf.ready(now):
            return
        value = self._buf.flush()
        if value is None:
            return
        self._seq += 1
        try:
            self._emit(Observation(
                schema="imu.window.v1",
                source_id=self.source_id,
                subject=self.subject,
                value=value,
                sequence=self._seq,
                timestamp=now).validate())
        except Exception as exc:
            logger.debug("imu window emit failed: %s", exc)

    # -- context push --------------------------------------------------------------------
    def push_context(self, payload: Dict[str, Any]) -> bool:
        """Write the current context snapshot to the watch. Returns False
        when the link is down — the caller may retry on next change."""
        return self._ble.central_write(
            self.address, CHAR_CONTEXT,
            json.dumps(payload, separators=(",", ":")).encode())


__all__ = [
    "CHAR_CONTEXT", "CHAR_RAW_MOTION", "ImuWindowBuffer", "MOTION_SERVICE",
    "PinetimeLink", "parse_motion",
]
