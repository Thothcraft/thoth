"""Context uplink — sensor descriptors, predictions and metadata to Brain
at a configurable rate and detail level.

Config (``~/.thoth/config.json`` → ``context``)::

    {"enabled": true, "rate_s": 60, "detail": "descriptors",
     "text": {"speech": true, "person": true, "face": true}}

* ``minimal``      — latest prediction per model + estimator states.
* ``descriptors``  — + per-sensor physical + textual descriptors
                     (default; ``scene`` = one line for the whole node).
* ``full``         — + radio scan emitters, room/metadata summary.

Textual cues come from :class:`whispy.textual.TextualDescriber` — rule
sentences for every sensor plus tiny on-device models where installed
(whisper-stt transcript, person count, face identity). Set a ``text``
flag to false to skip that model.

Each emission is one ``context.descriptors.v1`` observation on the
existing durable spool → ``observation_batch`` WS path, so delivery is
idempotent and survives reconnects. Brain's context builder consumes
these as evidence; raw frames never leave the node.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Mapping, Optional

from .observations import Observation

SCHEMA = "context.descriptors.v1"
TEXT_DEFAULTS: Dict[str, Any] = {"speech": True, "person": True,
                                 "face": True, "face_recognizer": None}
DEFAULTS: Dict[str, Any] = {"enabled": True, "rate_s": 60.0,
                            "detail": "descriptors",
                            "text": dict(TEXT_DEFAULTS)}
DETAIL_LEVELS = ("minimal", "descriptors", "full")
MIN_RATE_S = 5.0
MAX_RATE_S = 3600.0


def normalize_config(raw: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    for k, v in dict(raw or {}).items():
        if k in DEFAULTS:
            cfg[k] = v
    try:
        cfg["rate_s"] = min(MAX_RATE_S, max(MIN_RATE_S, float(cfg["rate_s"])))
    except (TypeError, ValueError):
        cfg["rate_s"] = DEFAULTS["rate_s"]
    if cfg["detail"] not in DETAIL_LEVELS:
        cfg["detail"] = DEFAULTS["detail"]
    cfg["enabled"] = bool(cfg["enabled"])
    text = dict(TEXT_DEFAULTS)
    if isinstance(cfg.get("text"), Mapping):
        text.update({k: v for k, v in cfg["text"].items()
                     if k in TEXT_DEFAULTS})
    for k in ("speech", "person", "face"):
        text[k] = bool(text[k])
    cfg["text"] = text
    return cfg


class ContextUplink:
    """Rate-limited builder of ``context.descriptors.v1`` observations."""

    def __init__(self, get_config: Callable[[], Mapping[str, Any]],
                 device_id: str, clock: Callable[[], float] = time.time,
                 describer_factory: Optional[Callable[[Dict[str, Any]], Any]] = None):
        self._get_config = get_config
        self.device_id = device_id
        self._clock = clock
        self._describer_factory = describer_factory
        self._describer: Any = None
        self._describer_key: Optional[str] = None
        self._last_emit = 0.0
        self.last: Optional[Dict[str, Any]] = None
        self.emitted = 0

    @property
    def config(self) -> Dict[str, Any]:
        return normalize_config(self._get_config())

    def describer(self, text_cfg: Dict[str, Any]) -> Any:
        """Persistent describer (model caches + rate limits survive
        ticks); rebuilt only when the text flags change."""
        import json as _json
        key = _json.dumps(text_cfg, sort_keys=True, default=str)
        if self._describer is None or key != self._describer_key:
            if self._describer_factory is not None:
                self._describer = self._describer_factory(text_cfg)
            else:
                from whispy.textual import TextualDescriber
                self._describer = TextualDescriber.default(
                    speech=text_cfg["speech"], person=text_cfg["person"],
                    face=text_cfg["face"],
                    face_recognizer=text_cfg.get("face_recognizer"))
            self._describer_key = key
        return self._describer

    def due(self, now: Optional[float] = None) -> bool:
        cfg = self.config
        now = self._clock() if now is None else now
        return cfg["enabled"] and now - self._last_emit >= cfg["rate_s"]

    def build(self, *, window: Any = None,
              predictions: Optional[List[Mapping[str, Any]]] = None,
              estimates: Optional[List[Mapping[str, Any]]] = None,
              room: Optional[Mapping[str, Any]] = None,
              now: Optional[float] = None) -> Dict[str, Any]:
        cfg = self.config
        now = self._clock() if now is None else now
        detail = cfg["detail"]
        latest: Dict[str, Dict[str, Any]] = {}
        for p in predictions or []:
            mid = str(p.get("runtime_model_id") or p.get("model_id") or "")
            if mid:
                latest[mid] = {"label": p.get("label"),
                               "confidence": p.get("confidence"),
                               "timestamp": p.get("timestamp")}
        value: Dict[str, Any] = {
            "detail": detail,
            "rate_s": cfg["rate_s"],
            "predictions": latest,
            "estimates": [
                {k: e.get(k) for k in ("key", "subject", "value",
                                       "confidence") if k in e}
                for e in (estimates or [])][:32],
        }
        if detail != "minimal" and window is not None:
            from whispy.descriptors import window_descriptors
            value["window_s"] = round(
                float(window.end_timestamp - window.start_timestamp), 2)
            physical = window_descriptors(window, detail, now)
            value["sensors"] = physical
            try:
                desc = self.describer(cfg["text"])
                text = desc.describe(window, window_descriptors(
                    window, "full", now), now)
                for sid, t in text.items():
                    if sid in physical:
                        physical[sid]["text"] = t.get("text")
                        if t.get("cues"):
                            physical[sid]["cues"] = t["cues"]
                value["scene"] = desc.summary(text)
            except Exception as exc:
                value["text_error"] = str(exc)[:160]
        if detail == "full" and room:
            value["room"] = {
                "room_id": room.get("room_id"),
                "name": room.get("name"),
                "rooms": [r.get("name") or r.get("room_id")
                          for r in room.get("rooms") or []][:16],
            }
        return value

    def maybe_emit(self, emit: Callable[[Observation], Any], **kwargs: Any
                   ) -> Optional[Dict[str, Any]]:
        now = kwargs.pop("now", None)
        now = self._clock() if now is None else now
        if not self.due(now):
            return None
        value = self.build(now=now, **kwargs)
        obs = Observation(SCHEMA, source_id=f"node:{self.device_id}",
                          value=value, subject=f"device:{self.device_id}",
                          observer=self.device_id, timestamp=now)
        emit(obs)
        self._last_emit = now
        self.last = value
        self.emitted += 1
        return value

    def status(self) -> Dict[str, Any]:
        return {"config": self.config, "emitted": self.emitted,
                "last_emit": self._last_emit or None, "last": self.last}


__all__ = ["SCHEMA", "DEFAULTS", "DETAIL_LEVELS", "ContextUplink",
           "normalize_config"]
