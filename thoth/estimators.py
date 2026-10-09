"""Node-side context estimators — observations → versioned state keys
(contract §4/§8).

``EstimatorHub`` consumes the observation stream the daemon already
fans out (``emit_observation``), runs small deterministic estimators,
and yields ``context.state.v1`` entries: canonical keys with
``estimator``, ``confidence``, ``evidence_ids`` back-references and a
monotonic ``version`` per key so consumers can detect stale state.

Phase 7 scope:

- ``presence.radio.v1`` — BLE subjects heard recently (radio presence
  only — a device on air is not a person in the room)
- ``occupancy.v1``    — enrolled-subject presence → occupied|empty|unknown
- ``activity.motion.v1`` — ``imu.window.v1`` variance → still|moving
- ``location.zone.v1`` — probabilistic RSSI fingerprinting: each zone
  keeps a per-anchor Gaussian (mean/std/count) learned over a
  calibration window and merged across repeated calibrations; a window
  is scored by per-anchor log-likelihood, zones compared via a softmax
  posterior, and the estimate abstains (``zone=None``) when every zone
  is implausible or the posterior is ambiguous — RADAR-style
  fingerprinting rather than single-snapshot nearest-match.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        # zone → subject → ts watermark of the last calibration merge,
        # so repeated calibrate_zone() calls fold only fresh samples
        # into the distribution instead of double-counting overlap
        self._cal_mark: Dict[str, Dict[str, float]] = {}
        self._states: Dict[str, _State] = {
            "presence.radio.v1": _State("presence.radio.v1", device_id,
                                        "ble-presence/1"),
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
    _CAL_WINDOW_S = 120.0    # calibration samples within this horizon
    _MIN_STD_DB = 2.0        # BLE RSSI never sits perfectly still
    _MISS_PENALTY = 2.0      # per calibrated anchor not currently heard
    _MIN_SCORE = -8.0        # mean log-likelihood floor — below this no
                           # zone is plausible → abstain (softmax over a
                           # single zone would otherwise report p=1.0)
    _MIN_POSTERIOR = 0.4     # top posterior under this → ambiguous

    @staticmethod
    def _fp_stats(v: Any) -> Dict[str, float]:
        """Normalize a fingerprint entry — legacy scalar means or the
        current ``{mean,std,count}`` distribution form."""
        if isinstance(v, dict):
            return {"mean": float(v.get("mean", 0.0)),
                    "std": max(EstimatorHub._MIN_STD_DB,
                               float(v.get("std")
                                     or EstimatorHub._MIN_STD_DB)),
                    "count": float(v.get("count") or 1)}
        return {"mean": float(v), "std": EstimatorHub._MIN_STD_DB,
                "count": 1.0}

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
        # Radio presence is reported on its own key — anonymous devices
        # count here even though they can never claim occupancy.
        all_ids = evidence + [oid for ids in anon.values()
                              for oid in ids]
        if self._states["presence.radio.v1"].apply(
                {"present": bool(enrolled or anon),
                 "enrolled_subjects": len(enrolled),
                 "anonymous_subjects": len(anon)},
                0.9 if (enrolled or anon) else 0.5,
                all_ids[-10:], now) and changed is not None:
            changed.append(self._states["presence.radio.v1"].to_dict())

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
        """Zone posterior by Gaussian RSSI fingerprinting — the mean
        fresh RSSI per anchor is scored against each zone's calibrated
        per-anchor Gaussian (log-likelihood, mean over observed
        anchors, penalized for anchors that went silent), softmaxed
        into a posterior. The estimate abstains when the best zone is
        implausible (score below _MIN_SCORE) or ambiguous (posterior
        below _MIN_POSTERIOR)."""
        if not self._fingerprints:
            return
        # current per-subject rssi — mean over samples still fresh
        # (anchors gone silent must not hold a zone estimate confident)
        cutoff = now - self._ZONE_TTL_S
        current: Dict[str, float] = {}
        last_oid: Dict[str, str] = {}
        for subj, dq in self._ble_seen.items():
            vals = [r for ts, _, r in dq
                    if ts >= cutoff and r is not None]
            if vals:
                current[subj] = sum(vals) / len(vals)
                last_oid[subj] = dq[-1][1]
        st = self._states["location.zone.v1"]
        if not current:
            # Evidence expired — publish 'unknown' so consumers don't
            # act on a zone anchored to stale RSSI.
            if st.value is not None and st.value.get("zone"):
                if st.apply({"zone": None, "expired": True},
                            0.1, [], now) and changed is not None:
                    changed.append(st.to_dict())
            return
        scores: Dict[str, float] = {}
        for zone, fp in self._fingerprints.items():
            anchors = {s: self._fp_stats(e) for s, e in fp.items()}
            observed = [s for s in anchors if s in current]
            if not observed:
                continue
            ll = 0.0
            for s in observed:
                a = anchors[s]
                z = (current[s] - a["mean"]) / a["std"]
                ll += -0.5 * z * z - math.log(a["std"])
            ll /= len(observed)
            # An anchor heard strongly that the fingerprint can't
            # explain counts against the zone (Horus-style coverage
            # penalty) — silence itself is not evidence either way.
            unseen = len(current) - len(observed)
            ll -= self._MISS_PENALTY * unseen / max(1, len(current))
            scores[zone] = ll
        if not scores:
            return
        mx = max(scores.values())
        weights = {z: math.exp(s - mx) for z, s in scores.items()}
        total = sum(weights.values())
        top_zone = max(weights, key=weights.get)
        top_p = weights[top_zone] / total
        runner = sorted((w for z, w in weights.items()
                         if z != top_zone), reverse=True)
        margin = top_p - (runner[0] / total if runner else 0.0)
        evidence = [last_oid[s] for s in current if last_oid.get(s)]
        if scores[top_zone] < self._MIN_SCORE or top_p < self._MIN_POSTERIOR:
            if st.apply({"zone": None,
                         "reason": ("implausible"
                                    if scores[top_zone] < self._MIN_SCORE
                                    else "ambiguous"),
                         "best_zone": top_zone,
                         "posterior": round(top_p, 3),
                         "score": round(scores[top_zone], 2)},
                        max(0.05, min(0.3, top_p)),
                        evidence[-10:], now) and changed is not None:
                changed.append(st.to_dict())
            return
        if st.apply({"zone": top_zone,
                     "posterior": round(top_p, 3),
                     "margin": round(margin, 3),
                     "score": round(scores[top_zone], 2),
                     "anchors": len([s for s in self._fingerprints[top_zone]
                                     if s in current])},
                    round(min(1.0, 0.5 + 0.5 * top_p + 0.3 * margin), 3),
                    evidence[-10:], now) and changed is not None:
            changed.append(st.to_dict())

    def calibrate_zone(self, zone: str) -> Optional[Dict[str, Any]]:
        """Fold the recent per-subject RSSI samples into ``zone``'s
        fingerprint — per anchor a Gaussian {mean, std, count} merged
        across repeated calibrations (Chan's parallel variance), so
        recalibrating a zone refines rather than replaces its
        distribution."""
        with self._lock:
            cutoff = time.time() - self._CAL_WINDOW_S
            marks = self._cal_mark.setdefault(zone, {})
            samples: Dict[str, List[Tuple[float, float]]] = {}
            for subj, dq in self._ble_seen.items():
                since = marks.get(subj)
                vals = [(ts, float(r)) for ts, _, r in dq
                        if ts >= cutoff and r is not None
                        and (since is None or ts > since)]
                if vals:
                    samples[subj] = vals
            if not samples:
                return None
            fp = self._fingerprints.setdefault(zone, {})
            for subj, vals in samples.items():
                marks[subj] = max(ts for ts, _ in vals)
                xs = [r for _, r in vals]
                n2 = float(len(xs))
                m2 = sum(xs) / len(xs)
                var2 = (sum((x - m2) ** 2 for x in xs) / (len(xs) - 1)
                        if len(xs) > 1 else 0.0)
                old = fp.get(subj)
                if old is None:
                    fp[subj] = {"mean": round(m2, 3),
                                "std": round(max(self._MIN_STD_DB,
                                                 math.sqrt(var2)), 3),
                                "count": int(n2)}
                    continue
                o = self._fp_stats(old)
                n1, m1, s1 = o["count"], o["mean"], o["std"]
                n = n1 + n2
                mean = (n1 * m1 + n2 * m2) / n
                # pooled within-group SS + between-group SS → merged var
                ss = (s1 * s1 * max(n1 - 1, 0)
                      + var2 * max(n2 - 1, 0)
                      + n1 * (m1 - mean) ** 2
                      + n2 * (m2 - mean) ** 2)
                fp[subj] = {"mean": round(mean, 3),
                            "std": round(max(self._MIN_STD_DB,
                                             math.sqrt(ss / max(n - 1, 1))),
                                         3),
                            "count": int(n)}
            self._save_fp()
            return {s: dict(v) for s, v in fp.items()}

    def fingerprints(self) -> Dict[str, Dict[str, Any]]:
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
