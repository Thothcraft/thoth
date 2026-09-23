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

logger = logging.getLogger(__name__)


class ActionDispatcher:
    """Fires actuators for predictions with cooldown/debounce (§17)."""

    def __init__(self) -> None:
        self._last_fired: Dict[str, float] = {}
        self.results: Deque[Dict[str, Any]] = deque(maxlen=200)

    def dispatch(self, action_cfg: Dict[str, Any], prediction: Any) -> Optional[Dict[str, Any]]:
        from whispy.contracts import Action
        from whispy.actuators import create_actuator

        action = Action.from_dict(action_cfg)
        now = time.time()
        key = action.type + ":" + str(action_cfg.get("name") or "")
        last = self._last_fired.get(key, 0.0)
        if action.cooldown_seconds and (now - last) < action.cooldown_seconds:
            return None
        try:
            actuator = create_actuator(action)
            result = actuator.trigger(action, prediction)
        except Exception as exc:
            from whispy.contracts import ActionResult, ActionStatus
            result = ActionResult(status=ActionStatus.FAILED,
                                  action_type=action.type, detail=str(exc))
        self._last_fired[key] = now
        rec = {"action": action.type, "result": result.to_dict(),
               "prediction": prediction.label, "at": now}
        self.results.append(rec)
        return rec


class ThothDaemon:
    """Persistent node service — owns the SMA loop and local API."""

    def __init__(self, config: Optional[ConfigStore] = None,
                 window_seconds: float = 2.0, tick_hz: float = 2.0):
        self.config = config or ConfigStore()
        self.registry = ModelRegistry()
        self.deployments = DeploymentManager(self.registry)
        self.captures = CaptureManager()
        self.dispatcher = ActionDispatcher()
        self.window_seconds = window_seconds
        self.tick_hz = tick_hz

        self._device = None
        self._streams: Dict[str, Any] = {}
        self._sync = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._api = None
        self.predictions: Deque[Dict[str, Any]] = deque(maxlen=500)
        self._started_at: Optional[float] = None

    # -- lifecycle --------------------------------------------------------------
    def start(self) -> "ThothDaemon":
        import whispy
        self._device = whispy.local()
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
                handle = self._device.sensor(sensor.id.split("-")[0])
                stream = SampleStream(handle.stream(), maxlen=4096,
                                      name=sensor.id)
                stream.start()
                self._streams[sensor.id] = stream
            except Exception as exc:
                logger.warning("sensor %s failed to open: %s", sensor.id, exc)

    def _start_api(self) -> None:
        from ..local_api import LocalAPIServer
        self._api = LocalAPIServer(self, host="127.0.0.1",
                                   port=int(self.config.get("local_port", 5000)),
                                   token=self.config.local_token)
        self._api.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
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
            self.dispatcher.dispatch(action_cfg, prediction)

    def _record_captures(self) -> None:
        for cap in self.captures._active.values():
            for sid in cap["sensors"]:
                stream = self._streams.get(sid)
                if not stream:
                    continue
                for sample in stream.drain():
                    self.captures.record(cap["id"], sample)

    # -- introspection -------------------------------------------------------------
    @property
    def device_id(self) -> str:
        return self._device.info.id if self._device else self.config.device_id

    def status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_name": self.config.device_name,
            "uptime_s": (time.time() - self._started_at) if self._started_at else 0,
            "sensors": [s.to_dict() for s in (self._device.sensors() if self._device else [])],
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
