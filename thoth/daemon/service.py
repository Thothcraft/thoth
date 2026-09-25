"""ThothDaemon — the persistent node service.

Runs the Sensor→Model→Actuator loop on Whispy abstractions:

    sensors (whispy.local) → SampleStream (bounded, non-blocking)
        → WindowSynchronizer (missing/stale markers)
        → active processors → Prediction
        → action dispatcher → actuators (explicit ActionResult)

Also records captures, serves the authenticated local API, and reports
health. Sensor ingestion never blocks the loop (§55).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from ..capture import CaptureManager
from ..deployment import DeploymentManager
from ..models import ModelRegistry
from ..settings import ConfigStore
from .actions import ActionScheduler

logger = logging.getLogger(__name__)


def _lan_ip() -> Optional[str]:
    """Best-effort primary LAN IPv4 (UDP connect — no traffic is sent)."""
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = str(s.getsockname()[0])
        s.close()
        return ip
    except Exception:
        return None


class ThothDaemon:
    """Persistent node service — owns the SMA loop and local API."""

    def __init__(self, config: Optional[ConfigStore] = None,
                 window_seconds: float = 2.0, tick_hz: float = 2.0):
        self.config = config or ConfigStore()
        self.registry = ModelRegistry()
        self.deployments = DeploymentManager(self.registry)
        self.captures = CaptureManager()
        self.dispatcher = ActionScheduler()
        from ..automation import AutomationManager
        self.automations = AutomationManager(self.dispatcher.fire_now)
        self.window_seconds = window_seconds
        self.tick_hz = tick_hz

        self._device = None
        self._streams: Dict[str, Any] = {}
        self._sync = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._api = None
        # action_id → ActionResult dict — idempotent actuator dispatch
        self._action_results: Dict[str, Dict[str, Any]] = {}
        self.predictions: Deque[Dict[str, Any]] = deque(maxlen=500)
        self._cap_subs: Dict[str, Dict[str, Any]] = {}
        self._started_at: Optional[float] = None
        self._last_heartbeat = 0.0

    # -- lifecycle --------------------------------------------------------------
    def start(self) -> "ThothDaemon":
        import whispy
        # One installation identity: the persisted config device_id is the
        # node's identity everywhere (pairing, captures, telemetry, Brain).
        # A pre-set _device (tests, embedded hosts) is used as-is.
        if self._device is None:
            self._device = whispy.local(device_id=self.config.device_id)
        self._open_streams()
        from whispy.synchronization import WindowSynchronizer
        self._sync = WindowSynchronizer(
            self._streams, expected=list(self._streams), stale_after_s=5.0)
        self._started_at = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="thoth-sma",
                                        daemon=True)
        self._thread.start()
        self._start_api()
        logger.info("ThothDaemon started: %d sensor stream(s)", len(self._streams))
        return self

    def _open_streams(self) -> None:
        from whispy.streams import SampleStream
        for sensor in self._device.sensors():
            try:
                handle = self._device.sensor(sensor.id)
                stream = SampleStream(handle.stream(), maxlen=4096,
                                      name=sensor.id)
                stream.start()
                self._streams[sensor.id] = stream
            except Exception as exc:
                logger.warning("sensor %s failed to open: %s", sensor.id, exc)

    def _start_api(self) -> None:
        from ..local_api import LocalAPIServer
        # Loopback by default; set local_host=0.0.0.0 to opt into LAN access.
        host = str(self.config.get("local_host", "127.0.0.1"))
        self._api = LocalAPIServer(self, host=host,
                                   port=int(self.config.get("local_port", 5000)),
                                   token=self.config.local_token)
        self._api.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        try:
            self.dispatcher.stop()
        except Exception:
            pass
        for cap_id, subs in self._cap_subs.items():
            for sid, sub in subs.items():
                stream = self._streams.get(sid)
                if stream:
                    try:
                        stream.unsubscribe(sub)
                    except Exception:
                        pass
        self._cap_subs.clear()
        for stream in self._streams.values():
            try:
                stream.close()
            except Exception:
                pass
        if self._device:
            try:
                self._device.close()
            except Exception:
                pass
        if self._api:
            self._api.stop()
        logger.info("ThothDaemon stopped")

    # -- SMA loop -----------------------------------------------------------------
    def _loop(self) -> None:
        period = 1.0 / self.tick_hz if self.tick_hz > 0 else 0.5
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:
                logger.warning("SMA tick error: %s", exc)
            time.sleep(period)

    def _tick(self) -> None:
        self._maybe_heartbeat()
        if self._sync is None:
            # No streams: still drive time/schedule triggers (no features).
            self.automations.tick()
            return
        window = self._sync.rolling(self.window_seconds)
        self._record_captures()
        from whispy.windows import WindowFeatures
        feats = WindowFeatures(window)
        last_pred: Optional[Any] = None
        for model in self.registry.active():
            try:
                proc = model.processor_impl()
                prediction = proc.predict(window)
                prediction.device_id = self.device_id
                prediction.runtime_model_id = model.runtime_model_id
            except Exception as exc:
                logger.warning("model %s predict failed: %s",
                               model.runtime_model_id, exc)
                continue
            self.predictions.append(prediction.to_dict())
            self._fire_actions(model, prediction)
            self.automations.on_prediction(prediction, features=feats)
            last_pred = prediction
        self.automations.tick(prediction=last_pred, features=feats)

    def _fire_actions(self, model: Any, prediction: Any) -> None:
        for action_cfg in model.config.get("actions") or []:
            self.dispatcher.submit(action_cfg, prediction,
                                   model_id=model.runtime_model_id)

    # -- Brain heartbeat ---------------------------------------------------------
    def _maybe_heartbeat(self) -> None:
        """POST /api/device/heartbeat so the portal sees this node online.

        Runs at most every ``heartbeat_interval_s`` (default 30s — Brain's
        online timeout is 90s). No-op until a device token is configured.
        Failures are logged at debug level: an unreachable Brain must never
        disturb local sensing.
        """
        token = self.config.device_token
        if not token:
            return
        interval = float(self.config.get("heartbeat_interval_s", 30.0))
        now = time.time()
        if now - self._last_heartbeat < interval:
            return
        self._last_heartbeat = now
        try:
            import json as _json
            import socket as _socket
            import urllib.request as _req
            body = {
                "device_id": self.device_id,
                "device_name": self.config.device_name,
                "device_type": "thoth",
                "device_hostname": _socket.gethostname(),
                "online": True,
                "hardware_info": self._heartbeat_inventory(),
            }
            req = _req.Request(
                f"{self.config.brain_url.rstrip('/')}/api/device/heartbeat",
                data=_json.dumps(body).encode(), method="POST",
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {token}"})
            with _req.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    logger.debug("heartbeat rejected: %s", resp.status)
        except Exception as exc:
            logger.debug("heartbeat failed: %s", exc)

    def _heartbeat_inventory(self) -> Dict[str, Any]:
        """Sensor/actuator inventory + LAN endpoint for the registry.

        Brain stores this under ``device.hardware_info``: ``local_api``
        lets same-LAN clients (and Brain-side actuation) reach the node;
        ``sensors`` feeds ``/v1/devices/{id}/sensors``.
        """
        sensors: List[Dict[str, Any]] = []
        actuators: List[Dict[str, Any]] = []
        if self._device is not None:
            try:
                sensors = [s.to_dict() for s in self._device.sensors()]
            except Exception:
                pass
            try:
                actuators = [a.to_dict() for a in self._device.actuators()]
            except Exception:
                pass
        local_api: Dict[str, Any] = {}
        host = _lan_ip()
        if host:
            local_api = {
                "host": host,
                "port": int(self.config.get("local_port", 5000)),
                "token": self.config.local_token,
            }
        return {"sensors": sensors, "actuators": actuators,
                "local_api": local_api}

    def _record_captures(self) -> None:
        """Flush each capture's private subscription queue to disk.

        Subscriptions are non-destructive: recording never removes samples
        from the shared buffer the synchronizer reads, and concurrent
        captures each receive their own copy of every sample.
        """
        active_ids = set(self.captures._active)
        # Drop subscriptions for captures that have stopped.
        for cap_id in [c for c in self._cap_subs if c not in active_ids]:
            for sid, sub in self._cap_subs.pop(cap_id).items():
                stream = self._streams.get(sid)
                if stream:
                    stream.unsubscribe(sub)

        for cap in self.captures._active.values():
            subs = self._cap_subs.setdefault(cap["id"], {})
            for sid in cap["sensors"]:
                stream = self._streams.get(sid)
                if not stream:
                    continue
                sub = subs.get(sid)
                if sub is None:
                    sub = stream.subscribe(name=f"capture-{cap['id']}")
                    subs[sid] = sub
                for sample in sub.read():
                    self.captures.record(cap["id"], sample)

    # -- LAN tail (§24) ---------------------------------------------------------
    def _exposure(self) -> Dict[str, Any]:
        """Capability exposure lists set by ``thoth expose``.

        ``{"sensors": [...], "actuators": [...]}`` — empty/absent lists
        mean "all". Permissions describe capabilities, not HTTP routes.
        """
        exp = self.config.get("exposed") or {}
        return {"sensors": list(exp.get("sensors") or []),
                "actuators": list(exp.get("actuators") or [])}

    def sensor_exposed(self, sensor_id: str) -> bool:
        allowed = self._exposure()["sensors"]
        return not allowed or sensor_id in allowed

    def actuator_exposed(self, actuator_id: str) -> bool:
        allowed = self._exposure()["actuators"]
        return not allowed or actuator_id in allowed

    def tail_sensor(self, sensor_id: str, cursor: int = 0) -> Optional[Dict[str, Any]]:
        """Return buffered samples newer than ``cursor`` for one sensor.

        Reads the stream's ring buffer non-destructively (``snapshot``), so
        tailing never disturbs inference or captures. Each sample's own
        ``sequence`` is the ordering key, so a client's cursor is just the
        last sequence it saw — stateless and safe for concurrent clients.
        Returns ``None`` for unknown or non-exposed sensors.
        """
        if not self.sensor_exposed(sensor_id):
            return None
        stream = self._streams.get(sensor_id)
        if stream is None:
            return None
        snap = stream.snapshot()
        latest = max((s.sequence for s in snap), default=cursor)
        samples = [s.to_dict() for s in snap if s.sequence > cursor]
        return {"sensor_id": sensor_id, "cursor": latest,
                "samples": samples}

    # -- actuators ---------------------------------------------------------------
    def actuators(self) -> List[Dict[str, Any]]:
        """Actuator inventory as descriptor dicts (exposure-filtered)."""
        if self._device is None:
            return []
        try:
            descriptors = self._device.actuators()
        except Exception:
            return []
        return [d.to_dict() for d in descriptors
                if self.actuator_exposed(d.id)]

    def execute_actuator(self, actuator_id: str,
                         command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an ActuatorCommand on a local actuator by id/kind.

        Returns an ActionResult dict — explicit outcomes only (§7.6).
        """
        from whispy.contracts import (
            ActionResult, ActionStatus, ActuatorCommand)
        action_id = command.get("action_id")
        expires_at = command.get("expires_at")
        # Idempotent dispatch: a replayed action_id returns the stored
        # result instead of re-executing (at-least-once → exactly-once).
        if action_id and action_id in self._action_results:
            out = dict(self._action_results[action_id])
            out["deduplicated"] = True
            return out
        # Expired commands never execute — explicit terminal status.
        if expires_at is not None and float(expires_at) < time.time():
            out = ActionResult(
                status=ActionStatus.EXPIRED, action_type="actuator",
                detail="command expired before dispatch").to_dict()
            if action_id:
                self._action_results[action_id] = dict(out)
            return out
        if not self.actuator_exposed(actuator_id):
            return ActionResult(
                status=ActionStatus.UNSUPPORTED,
                action_type="actuator",
                detail=f"actuator {actuator_id!r} is not exposed").to_dict()
        if self._device is None:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type="actuator",
                                detail="device not started").to_dict()
        try:
            handle = self._device.actuator(actuator_id)
        except KeyError:
            return ActionResult(
                status=ActionStatus.UNSUPPORTED, action_type="actuator",
                detail=f"unknown actuator {actuator_id!r}").to_dict()
        try:
            result = handle.execute(ActuatorCommand.from_dict(command))
            out = result.to_dict() if hasattr(result, "to_dict") \
                else dict(result)
        except Exception as exc:
            out = ActionResult(status=ActionStatus.FAILED,
                               action_type="actuator",
                               detail=str(exc)).to_dict()
        if action_id:
            self._action_results[action_id] = dict(out)
        return out

    # -- introspection -------------------------------------------------------------
    @property
    def device_id(self) -> str:
        return self._device.info.id if self._device else self.config.device_id

    def status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_name": self.config.device_name,
            "uptime_s": (time.time() - self._started_at) if self._started_at else 0,
            "sensors": [s.to_dict() for s in (self._device.sensors() if self._device else [])
                        if self.sensor_exposed(s.id)],
            "actuators": self.actuators(),
            "streams": {sid: {"dropped": st.dropped,
                              "last_ts": st.last_timestamp}
                        for sid, st in self._streams.items()},
            "models": [m.to_dict() for m in self.registry.list()],
            "active_models": len(self.registry.active()),
            "predictions": len(self.predictions),
            "captures": len(self.captures.list()),
            "running": not self._stop.is_set(),
        }

    def recent_predictions(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(self.predictions)[-limit:]

    def model_catalog(self) -> List[Dict[str, Any]]:
        """Runnable models: local whispy inventory + Brain's cloud catalog
        when the node is paired. Cloud failures degrade to local-only."""
        from whispy.models.registry import models as whispy_models
        out: List[Dict[str, Any]] = []
        try:
            out.extend(whispy_models())
        except Exception as exc:
            logger.debug("local model inventory failed: %s", exc)
        token = self.config.device_token
        if token and self.config.brain_url:
            try:
                import json as _json
                import urllib.request as _req
                req = _req.Request(
                    f"{self.config.brain_url.rstrip('/')}/v1/models",
                    headers={"Authorization": f"Bearer {token}"})
                with _req.urlopen(req, timeout=8) as res:
                    for m in _json.loads(res.read().decode()).get("models", []):
                        m = dict(m); m.setdefault("source", "cloud")
                        out.append(m)
            except Exception as exc:
                logger.debug("cloud model catalog unreachable: %s", exc)
        return out

    # -- v1 API surface (§12) ---------------------------------------------------
    def compute(self) -> Dict[str, Any]:
        """Compute capability advertisement — probed, never fabricated.

        Cached for 60s: probing spawns subprocesses (nvidia-smi) that can
        be slow, and compute capability is semi-static.
        """
        now = time.time()
        cached = getattr(self, "_compute_cache", None)
        if cached and now - cached[0] < 60.0:
            return dict(cached[1])
        from whispy.compute import probe_compute
        result = probe_compute().to_dict()
        self._compute_cache = (now, result)
        return dict(result)

    def health(self) -> Dict[str, Any]:
        """Structured health: per-source state, models, actuators, uptime."""
        sources = []
        for s in (self._device.sensors() if self._device else []):
            stream = self._streams.get(s.id)
            sources.append({
                "id": s.id, "type": s.type, "online": s.online,
                "exposed": self.sensor_exposed(s.id),
                "last_sample_ts": getattr(stream, "last_timestamp", None)
                    if stream else None,
                "dropped": getattr(stream, "dropped", 0) if stream else 0,
            })
        return {
            "device_id": self.device_id,
            "running": not self._stop.is_set(),
            "uptime_s": (time.time() - self._started_at)
                if self._started_at else 0,
            "sources": sources,
            "actuators": self.actuators(),
            "models": {"installed": len(self.registry.list()),
                       "active": len(self.registry.active())},
            "captures_active": len(self.captures._active),
        }

    def sources(self) -> List[Dict[str, Any]]:
        """All observation-source descriptors (exposure-filtered)."""
        if self._device is None:
            return []
        try:
            descriptors = self._device.sensor_descriptors()
        except Exception:
            descriptors = []
        if not descriptors:
            # Older devices only expose the Sensor inventory contract.
            return [s.to_dict() for s in self._device.sensors()
                    if self.sensor_exposed(s.id)]
        return [d.to_dict() for d in descriptors
                if self.sensor_exposed(d.id)]

    def source(self, key: str) -> Optional[Dict[str, Any]]:
        for desc in self.sources():
            if key in (desc.get("id"), desc.get("name")):
                return desc
        matches = [d for d in self.sources()
                   if d.get("modality") == key or d.get("type") == key]
        return matches[0] if len(matches) == 1 else None

    def source_observations(self, source_id: str,
                            cursor: int = 0) -> Optional[Dict[str, Any]]:
        """Canonical name for the sensor tail endpoint."""
        return self.tail_sensor(source_id, cursor)

    def _minutes_root(self):
        from pathlib import Path
        from ..settings import config_dir
        configured = self.config.get("data_dir")
        return Path(configured).expanduser() if configured \
            else config_dir() / "minutes"

    def minutes(self) -> List[Dict[str, Any]]:
        """Minute summaries under the node's data root."""
        from whispy.minutes import iter_minute_dirs, read_minute
        out = []
        for d in iter_minute_dirs(self._minutes_root()):
            try:
                m = read_minute(d)
                out.append({
                    "minute_id": m.minute_id, "device_id": m.device_id,
                    "start_timestamp": m.start_timestamp,
                    "end_timestamp": m.end_timestamp,
                    "sources": len(m.sources),
                    "predictions": len(m.predictions),
                    "labels": m.labels.get("labels") or [],
                    "canonical": (d / "minute.json").exists(),
                })
            except Exception as exc:
                out.append({"minute_id": d.name, "error": str(exc)})
        return out

    def minute(self, minute_id: str) -> Optional[Dict[str, Any]]:
        from whispy.minutes import iter_minute_dirs, read_minute
        for d in iter_minute_dirs(self._minutes_root()):
            if d.name == minute_id:
                return read_minute(d).to_dict()
        return None

    def minute_second(self, minute_id: str, index: int
                      ) -> Optional[Dict[str, Any]]:
        """Second-level view of one minute (legacy chunk data normalized)."""
        manifest = self.minute(minute_id)
        if manifest is None:
            return None
        seconds = (manifest.get("quality") or {}).get("seconds") or []
        for entry in seconds:
            if entry.get("second_index") == index:
                return entry
        return {"minute_id": minute_id, "second_index": index,
                "status": "missing"}

    def infer(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """One-shot inference on the live window → canonical InferenceResult."""
        from whispy.contracts import (
            InferenceRequest, InferenceResult, InferenceTrace)
        req = InferenceRequest.from_dict(request)
        started = time.time()
        model = None
        for m in self.registry.list():
            if req.model_id in (m.runtime_model_id, m.name):
                model = m
                break
        if model is None:
            return InferenceResult(
                request_id=req.request_id, status="failed",
                error=f"unknown model {req.model_id!r}").to_dict()
        if self._sync is None:
            return InferenceResult(
                request_id=req.request_id, status="failed",
                error="daemon not started").to_dict()
        try:
            window = self._sync.rolling(
                req.window_seconds or self.window_seconds)
            pred = model.processor_impl().predict(window)
            pred.device_id = self.device_id
            pred.runtime_model_id = model.runtime_model_id
            latency_ms = (time.time() - started) * 1000.0
            trace = InferenceTrace(
                model_id=req.model_id, runtime_id=model.runtime_model_id,
                execution_device=self.device_id, execution_class="local",
                input_interval={"start": window.start_timestamp,
                                "end": window.end_timestamp},
                inference_timestamp=started, latency_ms=latency_ms,
                confidence=pred.confidence)
            self.predictions.append(pred.to_dict())
            self._fire_actions(model, pred)
            return InferenceResult(request_id=req.request_id,
                                   status="succeeded", prediction=pred,
                                   trace=trace).to_dict()
        except Exception as exc:
            return InferenceResult(request_id=req.request_id,
                                   status="failed", error=str(exc)).to_dict()

    def privacy(self) -> Dict[str, Any]:
        """Current exposure/privacy posture — what leaves this node."""
        return {
            "exposed": self._exposure(),
            "local_api_host": str(self.config.get("local_host", "127.0.0.1")),
            "brain_url": self.config.brain_url,
            "paired": bool(self.config.device_token),
        }

    def sync_state(self) -> Dict[str, Any]:
        """Upload/sync posture for captures and minutes."""
        captures = self.captures.list()
        return {
            "captures_total": len(captures),
            "captures_active": len(self.captures._active),
            "minutes_root": str(self._minutes_root()),
            "brain_url": self.config.brain_url,
            "paired": bool(self.config.device_token),
        }

    def run_forever(self) -> None:
        self.start()
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
