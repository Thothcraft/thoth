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


class ThothDaemon:
    """Persistent node service — owns the SMA loop and local API."""

    def __init__(self, config: Optional[ConfigStore] = None,
                 window_seconds: float = 2.0, tick_hz: float = 2.0):
        self.config = config or ConfigStore()
        self.registry = ModelRegistry()
        self.deployments = DeploymentManager(self.registry)
        self.captures = CaptureManager()
        self.dispatcher = ActionScheduler()
        self.window_seconds = window_seconds
        self.tick_hz = tick_hz

        self._device = None
        self._streams: Dict[str, Any] = {}
        self._sync = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._api = None
        self.predictions: Deque[Dict[str, Any]] = deque(maxlen=500)
        self._cap_subs: Dict[str, Dict[str, Any]] = {}
        self._started_at: Optional[float] = None

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
        if self._sync is None:
            return
        window = self._sync.rolling(self.window_seconds)
        self._record_captures()
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

    def _fire_actions(self, model: Any, prediction: Any) -> None:
        for action_cfg in model.config.get("actions") or []:
            self.dispatcher.submit(action_cfg, prediction,
                                   model_id=model.runtime_model_id)

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
            return result.to_dict() if hasattr(result, "to_dict") \
                else dict(result)
        except Exception as exc:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type="actuator",
                                detail=str(exc)).to_dict()

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

    def run_forever(self) -> None:
        self.start()
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
