"""InfiniTime/PineTime link — central-role motion + neighbor-scan
subscription and context push.

On connect the node subscribes to the motion service and buffers
x/y/z into decimated ``imu.window.v1`` observations (the contract's
uplink unit — never the raw stream). The thoth-fork stamped char
``00030003`` (x/y/z + tick_ms + seq) is preferred: it carries the
watch's own clock and lets us count dropped notifications. Stock
firmware lacks it — ``subsystem`` tolerates the missing char and the
legacy ``00030002`` stream covers it.

When the fork's neighbor-scan service (0004) is present the watch's own
BLE sightings are relayed upstream as ``ble.rssi.v1`` — the wrist is a
third scanner viewpoint for the fleet map.
"""

from __future__ import annotations

import json
import logging
import struct
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..observations import Observation

logger = logging.getLogger(__name__)

# InfiniTime motion service + characteristics (upstream UUIDs).
MOTION_SERVICE = "00030000-78fc-48fe-8e23-433b3a1942d0"
CHAR_RAW_MOTION = "00030002-78fc-48fe-8e23-433b3a1942d0"
# thoth-fork stamped motion: x/y/z int16 + tick_ms uint32 + seq u8 (11 B).
CHAR_MOTION_STAMPED = "00030003-78fc-48fe-8e23-433b3a1942d0"
# thoth-fork neighbor-scan service: NOTIFY results + WRITE control.
CHAR_SCAN_RESULT = "00040001-78fc-48fe-8e23-433b3a1942d0"
CHAR_SCAN_CONTROL = "00040002-78fc-48fe-8e23-433b3a1942d0"
# Node→watch context channel — the fork has no context char yet; kept as
# a hook so ``push_context`` stays API-stable. (Previously aliased the
# stamped-motion char — writes there were rejected.)
CHAR_CONTEXT: Optional[str] = None

# BMA421 register scale: MotionController.X/Y/Z are binary milli-g,
# 1 g = 1024 (same scale the app uses).
_SCALE = 1024.0

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
        self.first_tick_ms: Optional[int] = None
        self.last_tick_ms: Optional[int] = None
        self.dropped = 0

    def add(self, x: float, y: float, z: float, ts: float,
            tick_ms: Optional[int] = None) -> None:
        if self.first_ts is None:
            self.first_ts = ts
        self.xs.append(x)
        self.ys.append(y)
        self.zs.append(z)
        self.last_ts = ts
        if tick_ms is not None:
            if self.first_tick_ms is None:
                self.first_tick_ms = tick_ms
            self.last_tick_ms = tick_ms

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
        if self.first_tick_ms is not None:
            value["tick_ms"] = [self.first_tick_ms, self.last_tick_ms]
        if self.dropped:
            value["dropped"] = self.dropped
        self.xs, self.ys, self.zs = [], [], []
        self.first_ts = self.last_ts = None
        self.first_tick_ms = self.last_tick_ms = None
        self.dropped = 0
        return value


def parse_motion(data: bytes) -> Optional[Tuple[float, float, float]]:
    """InfiniTime RawMotion payload: little-endian int16 x,y,z
    (binary milli-g — 1 g = 1024)."""
    if len(data) < 6:
        return None
    x, y, z = struct.unpack_from("<hhh", data, 0)
    return x / _SCALE, y / _SCALE, z / _SCALE   # → g


def parse_motion_stamped(data: bytes
                         ) -> Optional[Tuple[float, float, float,
                                             int, int]]:
    """thoth-fork stamped payload: ``<hhhIB`` — x,y,z int16 (binary
    milli-g) + tick_ms uint32 (1 kHz FreeRTOS tick) + seq uint8."""
    if len(data) < 11:
        return None
    x, y, z, tick_ms, seq = struct.unpack_from("<hhhIB", data, 0)
    return x / _SCALE, y / _SCALE, z / _SCALE, tick_ms, seq


def parse_scan_results(data: bytes) -> List[Tuple[str, int, Optional[str]]]:
    """Neighbor-scan notify: ``[0]=seq [1]=n | mac[6] rssi(i8)
    nameLen(u8) name[8]`` — returns (MAC, rssi_dbm, name) rows."""
    if len(data) < 2:
        return []
    n = data[1]
    stride = 6 + 1 + 1 + 8
    out: List[Tuple[str, int, Optional[str]]] = []
    for i in range(n):
        off = 2 + i * stride
        if off + stride > len(data):
            break
        mac = ":".join(f"{b:02X}" for b in data[off:off + 6])
        (rssi,) = struct.unpack_from("<b", data, off + 6)
        name_len = data[off + 7]
        name: Optional[str] = None
        if name_len:
            name = data[off + 8:off + 8 + name_len].decode(
                "utf-8", "replace")
        out.append((mac, rssi, name))
    return out


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
        self._scan_seq = 0
        self._up = False
        self._attached = False
        # Stamped-stream bookkeeping: once the fork char notifies, the
        # legacy char's duplicates are dropped and seq gaps are counted
        # as dropped notifications.
        self._stamped_seen = False
        self._last_stamped_seq: Optional[int] = None

    # -- lifecycle ------------------------------------------------------------------
    def attach(self) -> bool:
        if self._attached:
            return True
        # All three are attempted; ``subsystem._central_task`` tolerates
        # missing chars, so stock firmware just skips 00030003/0004.
        ok = self._ble.central_connect(
            self.address,
            subscriptions={
                CHAR_MOTION_STAMPED: self._on_motion_stamped,
                CHAR_RAW_MOTION: self._on_motion,
                CHAR_SCAN_RESULT: self._on_scan,
            },
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
            # Enable the duty-cycled neighbor scan — its results flow
            # back on CHAR_SCAN_RESULT and feed the fleet map.
            self._ble.central_write(self.address, CHAR_SCAN_CONTROL,
                                    b"\x01")

    # -- motion ingest ----------------------------------------------------------------
    def _on_motion_stamped(self, data: bytes) -> None:
        """Stamped notify (fork) — preferred stream."""
        parsed = parse_motion_stamped(data)
        if parsed is None:
            return
        x, y, z, tick_ms, seq = parsed
        self._stamped_seen = True
        if self._last_stamped_seq is not None:
            gap = (seq - self._last_stamped_seq - 1) & 0xFF
            # A gap larger than half the counter space is a wrap-order
            # artifact, not real drops.
            if 0 < gap < 128:
                self._buf.dropped += gap
        self._last_stamped_seq = seq
        self._add_sample(x, y, z, tick_ms=tick_ms)

    def _on_motion(self, data: bytes) -> None:
        """Legacy notify — stock firmware fallback; ignored once the
        stamped stream proves itself (same sensor, would double-count)."""
        if self._stamped_seen:
            return
        parsed = parse_motion(data)
        if parsed is None:
            return
        self._add_sample(*parsed)

    def _add_sample(self, x: float, y: float, z: float,
                    tick_ms: Optional[int] = None) -> None:
        """Notification ingest — runs on the BLE loop thread."""
        now = time.time()
        self._buf.add(x, y, z, ts=now, tick_ms=tick_ms)
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

    # -- neighbor scan relay ------------------------------------------------------------
    def _on_scan(self, data: bytes) -> None:
        """Watch-heard advertisers → ``ble.rssi.v1`` so Brain's map
        gains the wrist viewpoint (subject ble:<MAC> — the map's join
        key for the same radio seen by node+phone+watch)."""
        rows = parse_scan_results(data)
        if not rows:
            return
        now = time.time()
        for mac, rssi, name in rows[:8]:
            self._scan_seq += 1
            try:
                self._emit(Observation(
                    schema="ble.rssi.v1",
                    source_id=self.source_id,
                    subject=f"ble:{mac}",
                    value={"rssi_dbm": rssi, "mac": mac, "name": name,
                           "via": "pinetime-scan"},
                    units={"rssi_dbm": "dBm"},
                    sequence=self._scan_seq,
                    timestamp=now).validate())
            except Exception as exc:
                logger.debug("watch scan emit failed: %s", exc)

    # -- context push --------------------------------------------------------------------
    def push_context(self, payload: Dict[str, Any]) -> bool:
        """Write the current context snapshot to the watch. Returns
        False while no context char exists (fork pending) — the caller
        may retry on next change."""
        if CHAR_CONTEXT is None:
            return False
        return self._ble.central_write(
            self.address, CHAR_CONTEXT,
            json.dumps(payload, separators=(",", ":")).encode())


__all__ = [
    "CHAR_CONTEXT", "CHAR_MOTION_STAMPED", "CHAR_RAW_MOTION",
    "CHAR_SCAN_CONTROL", "CHAR_SCAN_RESULT", "ImuWindowBuffer",
    "MOTION_SERVICE", "PinetimeLink", "parse_motion",
    "parse_motion_stamped", "parse_scan_results",
]
