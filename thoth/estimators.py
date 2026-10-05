"""Node-side context estimators — observations → versioned state keys
(contract §4/§8).

``EstimatorHub`` consumes the observation stream the daemon already
fans out (``emit_observation``), runs small deterministic estimators,
and yields ``context.state.v1`` entries: canonical keys with
``estimator``, ``confidence``, ``evidence_ids`` back-references and a
monotonic ``version`` per key so consumers can detect stale state.

Phase 7 scope:

- ``occupancy.v1``    — BLE presence/RSSI freshness → space occupied
- ``activity.motion.v1`` — ``imu.window.v1`` variance → still|moving
- ``location.zone.v1`` — RSSI fingerprint nearest-zone estimate
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA_STATE = "context.state.v1"


class _State:
    """One versioned context-state entry (key + evidence links)."""
    __slots__ = ("key", "entity_id", "value", "confidence", "estimator",
                 "evidence_ids", "ts", "version")

    def __init__(self, key: str, entity_id: str, estimator: str):
        self.key = key
        self.entity_id = entity_id
        self.estimator = estimator
        self.value: Any = None
        self.confidence: Optional[float] = None
        self.evidence_ids: List[str] = []
        self.ts: float = 0.0
        self.version = 0

    def apply(self, value: Any, confidence: float, evidence: List[str],
              ts: float) -> bool:
        """Bump the version only on real transitions — consumers use it
        to detect change, so heartbeats must not churn it."""
        if self.value == value:
            self.ts = ts
            self.evidence_ids = evidence
            return False
        self.value = value
        self.confidence = confidence
        self.evidence_ids = evidence
        self.ts = ts
        self.version += 1
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "entity_id": self.entity_id,
                "value": self.value, "confidence": self.confidence,
                "estimator": self.estimator,
                "evidence_ids": list(self.evidence_ids),
                "version": self.version, "ts": self.ts}


class EstimatorHub:
    """Feeds observations to estimators; states() is the current view.

    Thread-safe: consume() is called from ``emit_observation`` (any
    producer thread), tick() from the SMA loop, states() from the API.
    """

    def __init__(self, device_id: str,
                 fingerprints_path: Optional[Path] = None):
        self.device_id = device_id
        self._lock = threading.Lock()
        # recent evidence: subject → deque of (ts, observation_id, rssi)
        self._ble_seen: Dict[str, deque] = {}
        # subject → current zone rssi snapshot for fingerprinting
        self._imu_windows: deque = deque(maxlen=8)
        self._fp_path = fingerprints_path
        self._fingerprints: Dict[str, Dict[str, float]] = self._load_fp()
        self._states: Dict[str, _State] = {
            "occupancy.v1": _State("occupancy.v1", device_id,
                                   "ble-presence/1"),
            "activity.motion.v1": _State("activity.motion.v1", device_id,
                                         "imu-variance/1"),
            "location.zone.v1": _State("location.zone.v1", device_id,
                                       "rssi-fingerprint/1"),
        }

    # -- fingerprint persistence ---------------------------------------------------
    def _load_fp(self) -> Dict[str, Dict[str, float]]:
        if self._fp_path is None:
            return {}
        try:
            import json as _json
            raw = _json.loads(self._fp_path.read_text())
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def _save_fp(self) -> None:
        if self._fp_path is None:
            return
        try:
            import json as _json
            self._fp_path.parent.mkdir(parents=True, exist_ok=True)
            self._fp_path.write_text(_json.dumps(self._fingerprints,
                                                 indent=2))
        except OSError:
            pass

    # -- consume --------------------------------------------------------------------
    def consume(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Every emitted observation flows through here. Returns the
        context states whose version bumped — the daemon uplinks them
        as ``context.state.v1`` observations."""
        schema = str(obs.get("schema", ""))
        now = float(obs.get("timestamp") or time.time())
        changed: List[Dict[str, Any]] = []
        with self._lock:
            if schema == "ble.rssi.v1" or schema == "ble.presence.v1":
                subj = str(obs.get("subject") or "")
                if not subj:
                    return changed
                dq = self._ble_seen.setdefault(subj, deque(maxlen=64))
                dq.append((now, str(obs.get("observation_id") or ""),
                           (obs.get("value") or {}).get("rssi_dbm")))
                self._eval_occupancy(now, changed)
                if schema == "ble.rssi.v1":
                    self._eval_zone(now, changed)
            elif schema == "imu.window.v1":
                self._imu_windows.append(
                    (now, str(obs.get("observation_id") or ""),
                     obs.get("value") or {},
                     obs.get("subject")))
                self._eval_motion(now, changed)
        return changed

    def tick(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Periodic re-eval — sources going silent produce no new
        observations, so decay (empty space, stale zone) only shows up
        here. Called from the SMA loop."""
        now = now if now is not None else time.time()
        changed: List[Dict[str, Any]] = []
        with self._lock:
            self._eval_occupancy(now, changed)
            self._eval_zone(now, changed)
        return changed

    # -- estimators ---------------------------------------------------------------
    _OCC_WINDOW_S = 120.0

    def _eval_occupancy(self, now: float,
                        changed: Optional[List[Dict[str, Any]]] = None
                        ) -> None:
        """occupied ⟸ ≥1 subject (enrolled or anonymous) seen recently."""
        cutoff = now - self._OCC_WINDOW_S
        live: Dict[str, List[str]] = {}
        for subj, dq in self._ble_seen.items():
            recent = [oid for ts, oid, _ in dq if ts >= cutoff]
            if recent:
                live[subj] = recent[-4:]
        occupied = bool(live)
        evidence = [oid for ids in live.values() for oid in ids]
        conf = min(1.0, 0.6 + 0.1 * len(live)) if occupied else 0.4
        if self._states["occupancy.v1"].apply(
                {"occupied": occupied, "distinct_subjects": len(live)},
                conf, evidence[-10:], now) and changed is not None:
            changed.append(self._states["occupancy.v1"].to_dict())

    def _eval_motion(self, now: float,
                     changed: Optional[List[Dict[str, Any]]] = None
                     ) -> None:
        """imu.window.v1 → stationary|moving from per-axis variance."""
        windows = list(self._imu_windows)[-4:]
        variances: List[float] = []
        evidence: List[str] = []
        subject = None
        for ts, oid, value, subj in windows:
            axes = (value or {}).get("axes") or {}
            xs = axes.get("x") or []
            ys = axes.get("y") or []
            zs = axes.get("z") or []
            for arr in (xs, ys, zs):
                if len(arr) >= 4:
                    mean = sum(arr) / len(arr)
                    variances.append(
                        sum((v - mean) ** 2 for v in arr) / len(arr))
            evidence.append(oid)
            subject = subject or subj
        if not variances:
            return
        var = sum(variances) / len(variances)
        moving = var > 0.02    # g² threshold — resting wrist ≈ 0.001–0.01
        st = self._states["activity.motion.v1"]
        if st.apply({"motion": "moving" if moving else "stationary",
                     "variance": round(var, 5)},
                    0.8, evidence[-8:], now) and changed is not None:
            changed.append(st.to_dict())
        if subject:
            st.entity_id = str(subject)

    def _eval_zone(self, now: float,
                   changed: Optional[List[Dict[str, Any]]] = None
                   ) -> None:
        """Nearest zone by RSSI fingerprint — only over subjects that
        carry fingerprints (enrolled anchors)."""
        if not self._fingerprints:
            return
        # current per-subject rssi (latest sample per subject)
        current: Dict[str, float] = {}
        for subj, dq in self._ble_seen.items():
            if dq:
                rssi = dq[-1][2]
                if rssi is not None:
                    current[subj] = float(rssi)
        if not current:
            return
        best_zone, best_dist = None, float("inf")
        evidence: List[str] = []
        for zone, fp in self._fingerprints.items():
            common = [s for s in fp if s in current]
            if not common:
                continue
            dist = math.sqrt(sum(
                (current[s] - fp[s]) ** 2 for s in common)) / len(common)
            if dist < best_dist:
                best_dist, best_zone = dist, zone
        if best_zone is None:
            return
        # distance→confidence: 0 dB error ≈ 1.0, ≥15 dB ≈ 0.2
        conf = max(0.2, min(1.0, 1.0 - best_dist / 15.0))
        for dq in self._ble_seen.values():
            if dq:
                evidence.append(dq[-1][1])
        st = self._states["location.zone.v1"]
        if st.apply({"zone": best_zone,
                     "rssi_distance": round(best_dist, 2)},
                    conf, evidence[-10:], now) and changed is not None:
            changed.append(st.to_dict())

    def calibrate_zone(self, zone: str) -> Optional[Dict[str, float]]:
        """Snapshot current per-subject RSSI as ``zone``'s fingerprint —
        run while the node sits in that zone."""
        with self._lock:
            snap: Dict[str, float] = {}
            for subj, dq in self._ble_seen.items():
                if dq and dq[-1][2] is not None:
                    snap[subj] = float(dq[-1][2])
            if not snap:
                return None
            self._fingerprints[zone] = snap
            self._save_fp()
            return dict(snap)

    def fingerprints(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {z: dict(fp) for z, fp in self._fingerprints.items()}

    # -- read ----------------------------------------------------------------------
    def states(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [s.to_dict() for s in self._states.values()
                    if s.value is not None]

    def state_for(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            st = self._states.get(key)
            return st.to_dict() if st and st.value is not None else None


__all__ = ["EstimatorHub"]
