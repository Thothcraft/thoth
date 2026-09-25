"""Action scheduling — delay, debounce, cooldown, cancellation (§17, §55).

The SMA loop calls :meth:`ActionScheduler.submit` once per (model, action)
per tick with the latest prediction. Submission is non-blocking: timing
policy is evaluated here and due actions are executed on a bounded worker
pool so a slow actuator never stalls inference or capture recording.

Semantics (Architecture v3.0 §17, §36):

- **debounce_seconds** — the qualifying condition must hold continuously
  for this long before the action may fire. A flicker resets the clock.
- **delay_seconds** — after the condition holds, wait this long before
  firing (e.g. "turn the light off 180 s after the room is empty").
- **cancellation** — if the condition clears while a fire is pending
  (occupancy returns), the pending fire is cancelled.
- **cooldown_seconds** — minimum interval between actual fires of the
  same action. Cooldown is only consumed by a real dispatch, never by a
  gated/rejected prediction.
- **stable identity** — each action is keyed by model + actuator type +
  target config, so two distinct actions of the same type never share a
  timer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional

logger = logging.getLogger(__name__)


def _qualifies(action: Any, prediction: Any) -> bool:
    """Whether a prediction passes an action's confidence/label gates."""
    if prediction.confidence < action.min_confidence:
        return False
    if action.trigger_labels and prediction.label not in action.trigger_labels:
        return False
    return True


def _action_key(model_id: str, action: Any, action_cfg: Dict[str, Any]) -> str:
    """Stable identity for an action: model + type + name + target config.

    Two unnamed actions of the same type but different targets get
    different keys because their ``config`` differs.
    """
    target = json.dumps(action.config, sort_keys=True, default=str)
    digest = hashlib.sha1(target.encode()).hexdigest()[:10]
    name = action_cfg.get("name") or action.config.get("name") or ""
    return f"{model_id}:{action.type}:{name}:{digest}"


class _ActionState:
    __slots__ = ("key", "action", "action_cfg", "since", "due", "pred",
                 "last_fired")

    def __init__(self, key: str, action: Any, action_cfg: Dict[str, Any]):
        self.key = key
        self.action = action
        self.action_cfg = action_cfg
        self.since: Optional[float] = None      # condition true since (mono)
        self.due: Optional[float] = None        # scheduled fire time (mono)
        self.pred: Optional[Any] = None         # latest qualifying prediction
        self.last_fired: Optional[float] = None  # last actual dispatch (mono)


class ActionScheduler:
    """Non-blocking action dispatcher with timing policy and workers."""

    def __init__(self, workers: int = 4, queue_size: int = 64,
                 clock: Callable[[], float] = time.monotonic,
                 autostart: bool = True,
                 executor: Optional[Callable[[Any, Any], Any]] = None):
        self._clock = clock
        self._states: Dict[str, _ActionState] = {}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self.results: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._queue: "queue.Queue[tuple]" = queue.Queue(maxsize=queue_size)
        self._executor = executor or self._default_executor
        # workers=0 → execute inline at fire time (deterministic for tests).
        self._inline = workers <= 0
        self._workers = []
        for i in range(max(0, workers)):
            t = threading.Thread(target=self._worker, name=f"thoth-act-{i}",
                                 daemon=True)
            t.start()
            self._workers.append(t)
        self._thread: Optional[threading.Thread] = None
        if autostart:
            self._thread = threading.Thread(target=self._run,
                                            name="thoth-act-sched",
                                            daemon=True)
            self._thread.start()

    # -- submission (called from the SMA loop; must not block) ---------------
    def submit(self, action_cfg: Dict[str, Any], prediction: Any,
               model_id: str = "") -> None:
        from whispy.contracts import Action
        try:
            action = Action.from_dict(action_cfg)
        except Exception as exc:
            logger.warning("bad action config: %s", exc)
            return
        key = _action_key(model_id, action, action_cfg)
        now = self._clock()
        with self._cond:
            st = self._states.get(key)
            if st is None:
                st = _ActionState(key, action, action_cfg)
                self._states[key] = st
            st.action = action
            if _qualifies(action, prediction):
                st.pred = prediction
                if st.since is None:
                    # Condition just became true: arm debounce + delay.
                    st.since = now
                    st.due = now + action.debounce_seconds + action.delay_seconds
                    self._cond.notify_all()
                # else: already pending — keep the original due time so the
                # delay counts from when the condition began, not now.
            else:
                # Condition cleared → cancel any pending fire (occupancy
                # returned) and reset the debounce clock.
                st.since = None
                st.due = None
                st.pred = None

    # Backwards-compatible alias used by the daemon/local API.
    def dispatch(self, action_cfg: Dict[str, Any], prediction: Any,
                 model_id: str = "") -> None:
        self.submit(action_cfg, prediction, model_id=model_id)

    def fire_now(self, action_cfg: Dict[str, Any], prediction: Any,
                 model_id: str = "") -> None:
        """Dispatch immediately — the caller (automations) already decided
        the action should fire. Cooldown is still honored via last_fired."""
        from whispy.contracts import Action
        try:
            action = Action.from_dict(action_cfg)
        except Exception as exc:
            logger.warning("bad fire_now action config: %s", exc)
            return
        key = _action_key(model_id or "auto", action, action_cfg)
        now = self._clock()
        with self._cond:
            st = self._states.get(key)
            if st is None:
                st = _ActionState(key, action, action_cfg)
                self._states[key] = st
            st.action = action
            if action.cooldown_seconds and st.last_fired is not None and \
                    (now - st.last_fired) < action.cooldown_seconds:
                return
            st.pred = prediction
            st.due = now
            self._cond.notify_all()

    # -- scheduler loop -------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cond:
                now = self._clock()
                due_keys = [k for k, st in self._states.items()
                            if st.due is not None and st.due <= now]
                next_due = min((st.due for st in self._states.values()
                                if st.due is not None), default=None)
                for key in due_keys:
                    self._fire(key, now)
                # Sleep until the next pending fire (or a short cap).
                wait = 0.25
                if next_due is not None:
                    wait = max(0.01, min(0.25, next_due - self._clock()))
                self._cond.wait(timeout=wait)

    def _fire(self, key: str, now: float) -> None:
        st = self._states.get(key)
        if st is None or st.due is None or st.pred is None:
            return
        action = st.action
        # Cooldown gates the actual dispatch only.
        if action.cooldown_seconds and st.last_fired is not None and \
                (now - st.last_fired) < action.cooldown_seconds:
            st.due = None
            st.since = None
            return
        pred = st.pred
        st.due = None
        st.since = None
        st.last_fired = now
        if self._inline:
            self._executor(key, action, pred)
            return
        try:
            self._queue.put_nowait((key, action, pred))
        except queue.Full:
            logger.warning("action queue saturated; dropping %s", key)
            self._record(key, action, pred, status="failed",
                         detail="action queue saturated")

    def fire_due(self, now: Optional[float] = None) -> int:
        """Synchronously fire all due actions (test/manual driver)."""
        now = self._clock() if now is None else now
        fired = 0
        with self._cond:
            for key, st in list(self._states.items()):
                if st.due is not None and st.due <= now:
                    self._fire(key, now)
                    fired += 1
        return fired

    # -- execution workers ------------------------------------------------------
    def _worker(self) -> None:
        while True:
            try:
                key, action, pred = self._queue.get(timeout=0.25)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            try:
                self._executor(key, action, pred)
            except Exception as exc:  # a bad actuator must not kill workers
                logger.warning("action %s executor raised: %s", key, exc)
            finally:
                self._queue.task_done()

    def _default_executor(self, key: str, action: Any, pred: Any) -> None:
        from whispy.contracts import ActionResult, ActionStatus
        from whispy.actuators import create_actuator
        try:
            actuator = create_actuator(action)
            result = actuator.trigger(action, pred)
        except Exception as exc:
            result = ActionResult(status=ActionStatus.FAILED,
                                  action_type=action.type, detail=str(exc))
        self._record(key, action, pred, result=result)

    def _record(self, key: str, action: Any, pred: Any,
                result: Optional[Any] = None, status: Optional[str] = None,
                detail: str = "") -> None:
        rec = {"key": key, "action": action.type,
               "result": result.to_dict() if result is not None else
                         {"status": status or "failed", "detail": detail},
               "prediction": getattr(pred, "label", None),
               "at": time.time()}
        self.results.append(rec)

    # -- lifecycle --------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for w in self._workers:
            w.join(timeout=1.0)

    def pending(self) -> Dict[str, float]:
        """Snapshot of armed actions and their due times (diagnostics)."""
        with self._lock:
            return {k: st.due for k, st in self._states.items()
                    if st.due is not None}


__all__ = ["ActionScheduler"]
