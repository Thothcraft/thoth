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
        # subject → deque of imu windows (per-subject — a shared buffer
        # let one person's motion contaminate another's state)
        self._imu_windows: Dict[str, deque] = {}
        # subject → its own activity.motion.v1 state entry
        self._motion_states: Dict[str, _State] = {}
        self._fp_path = fingerprints_path
        self._fingerprints: Dict[str, Dict[str, float]] = self._load_fp()
        self._states: Dict[str, _State] = {
            "occupancy.v1": _State("occupancy.v1", device_id,
                                   "ble-presence/1"),
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
                subj = str(obs.get("subject") or self.device_id)
                dq = self._imu_windows.setdefault(subj, deque(maxlen=8))
                dq.append((now, str(obs.get("observation_id") or ""),
                           obs.get("value") or {}))
                self._eval_motion(subj, now, changed)
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
    _ZONE_TTL_S = 120.0

    # Anonymous BLE subjects (``device:ble:<hmac12>``) are radio
    # presence only — a printer/neighbour must not claim a person is
    # home. Occupancy requires ≥1 enrolled subject; anonymous-only or
    # zero-evidence windows report ``occupied: None`` (unknown) unless
    # an enrolled device was previously seen and went silent (empty).
    _ANON_PREFIX = "device:ble:"

    def _eval_occupancy(self, now: float,
                        changed: Optional[List[Dict[str, Any]]] = None
                        ) -> None:
        """occupied ⟸ ≥1 *enrolled* subject seen recently."""
        cutoff = now - self._OCC_WINDOW_S
        enrolled: Dict[str, List[str]] = {}
        anon: Dict[str, List[str]] = {}
        for subj, dq in self._ble_seen.items():
            recent = [oid for ts, oid, _ in dq if ts >= cutoff]
            if not recent:
                continue
            (anon if subj.startswith(self._ANON_PREFIX)
             else enrolled)[subj] = recent[-4:]
        enrolled_known = any(
            not s.startswith(self._ANON_PREFIX) for s in self._ble_seen)
        if enrolled:
            occupied: Optional[bool] = True
            conf = min(1.0, 0.6 + 0.1 * len(enrolled))
        elif anon or not enrolled_known:
            occupied = None   # presence without identity ≠ occupancy
            conf = 0.3
        else:
            occupied = False  # enrolled subjects exist but went silent
            conf = 0.5
        evidence = [oid for ids in enrolled.values() for oid in ids]
        if self._states["occupancy.v1"].apply(
                {"occupied": occupied,
                 "distinct_subjects": len(enrolled),
                 "anonymous_subjects": len(anon)},
                conf, evidence[-10:], now) and changed is not None:
            changed.append(self._states["occupancy.v1"].to_dict())

    def _eval_motion(self, subject: str, now: float,
                     changed: Optional[List[Dict[str, Any]]] = None
                     ) -> None:
        """imu.window.v1 → stationary|moving from per-axis variance,
        partitioned per subject — never mix windows across people."""
        windows = list(self._imu_windows.get(subject, ()))[-4:]
        variances: List[float] = []
        evidence: List[str] = []
        for ts, oid, value in windows:
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
        if not variances:
            return
        var = sum(variances) / len(variances)
        moving = var > 0.02    # g² threshold — resting wrist ≈ 0.001–0.01
        st = self._motion_states.get(subject)
        if st is None:
            st = self._motion_states[subject] = _State(
                "activity.motion.v1", subject, "imu-variance/1")
        if st.apply({"motion": "moving" if moving else "stationary",
                     "variance": round(var, 5),
                     "subject": subject},
                    0.8, evidence[-8:], now) and changed is not None:
            changed.append(st.to_dict())

    def _eval_zone(self, now: float,
                   changed: Optional[List[Dict[str, Any]]] = None
                   ) -> None:
        """Nearest zone by RSSI fingerprint — only over subjects that
        carry fingerprints (enrolled anchors)."""
        if not self._fingerprints:
            return
        # current per-subject rssi (latest *fresh* sample per subject —
        # anchors gone silent must not hold a zone estimate confident)
        cutoff = now - self._ZONE_TTL_S
        current: Dict[str, float] = {}
        for subj, dq in self._ble_seen.items():
            if dq and dq[-1][0] >= cutoff and dq[-1][2] is not None:
                current[subj] = float(dq[-1][2])
        if not current:
            # Evidence expired — publish 'unknown' so consumers don't
            # act on a zone anchored to stale RSSI.
            st = self._states["location.zone.v1"]
            if st.value is not None and st.value.get("zone"):
                if st.apply({"zone": None, "expired": True},
                            0.1, [], now) and changed is not None:
                    changed.append(st.to_dict())
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
            out = [s for s in self._states.values() if s.value is not None]
            out += [s for s in self._motion_states.values()
                    if s.value is not None]
            return [s.to_dict() for s in out]

    def state_for(self, key: str,
                  subject: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Latest state for ``key`` — per-subject motion keys resolve by
        ``subject`` (or return the freshest when omitted/ambiguous)."""
        with self._lock:
            if key == "activity.motion.v1":
                pool = self._motion_states
                st = pool.get(subject) if subject else None
                if st is None and pool:
                    st = max((s for s in pool.values()
                              if s.value is not None),
                             key=lambda s: s.ts, default=None)
            else:
                st = self._states.get(key)
            return st.to_dict() if st and st.value is not None else None


__all__ = ["EstimatorHub"]
