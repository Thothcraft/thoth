"""Node runtime for whispy's built-in discriminators.

Each tick the daemon hands the current window's descriptors here:

* an active guided :class:`~whispy.discriminators.CalibrationSession`
  records them for the current step;
* a bounded feature history feeds automated (2-means) calibration;
* every calibrated + enabled discriminator predicts, at most once per
  ``interval_s``. Predictions flow through the same path as model
  predictions (edge events, automations, notifications).

Config (``config["discriminators"]``)::

    {"interval_s": 2.0, "disabled": ["noise-mic"]}
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional

from whispy.discriminators import (CalibrationSession, DiscriminatorBank,
                                   features_from_descriptors)

HISTORY_MAX = 4000


class DiscriminatorRuntime:
    def __init__(self, get_config: Callable[[], Mapping[str, Any]],
                 directory: Optional[Path] = None,
                 clock: Callable[[], float] = time.time):
        self._get_config = get_config
        self.bank = DiscriminatorBank(directory)
        self._clock = clock
        self._lock = threading.Lock()
        self.session: Optional[CalibrationSession] = None
        self.history: Deque[Dict[str, float]] = deque(maxlen=HISTORY_MAX)
        self._last_predict = 0.0

    # -- config ------------------------------------------------------------
    def _cfg(self) -> Dict[str, Any]:
        return dict(self._get_config() or {})

    def enabled_names(self) -> List[str]:
        disabled = set(self._cfg().get("disabled") or [])
        return [n for n in self.bank.items if n not in disabled]

    # -- per-tick ----------------------------------------------------------
    def observe(self, descriptors: Mapping[str, Any], *, device_id: str = "",
                now: Optional[float] = None) -> List[Any]:
        now = self._clock() if now is None else now
        feats = features_from_descriptors(descriptors)
        with self._lock:
            if feats:
                self.history.append(feats)
            if self.session is not None:
                self.session.record(descriptors)
        interval = float(self._cfg().get("interval_s", 2.0))
        if now - self._last_predict < interval:
            return []
        self._last_predict = now
        return self.bank.predict_all(descriptors, device_id=device_id,
                                     enabled=self.enabled_names())

    # -- calibration -------------------------------------------------------
    def start_guided(self, name: str) -> Dict[str, Any]:
        disc = self.bank.get(name)
        with self._lock:
            self.session = CalibrationSession(disc)
            return self.session.status()

    def next_step(self) -> Dict[str, Any]:
        with self._lock:
            if self.session is None:
                raise ValueError("no calibration in progress")
            self.session.next_step()
            return self.session.status()

    def finish(self) -> Dict[str, Any]:
        with self._lock:
            if self.session is None:
                raise ValueError("no calibration in progress")
            sess, self.session = self.session, None
        sess.finish()
        self.bank.save(sess.recipe.name)
        return self.bank.get(sess.recipe.name).to_dict()

    def cancel(self) -> Dict[str, Any]:
        with self._lock:
            self.session = None
        return {"ok": True}

    def calibrate_auto(self, name: str) -> Dict[str, Any]:
        disc = self.bank.get(name)
        with self._lock:
            hist = list(self.history)
        disc.calibrate_auto(hist)
        self.bank.save(name)
        return disc.to_dict()

    def reset(self, name: str) -> Dict[str, Any]:
        self.bank.reset(name)
        return self.bank.get(name).to_dict()

    def status(self) -> Dict[str, Any]:
        enabled = set(self.enabled_names())
        items = []
        for d in self.bank.to_list():
            d["enabled"] = d["name"] in enabled
            items.append(d)
        with self._lock:
            sess = self.session.status() if self.session else None
            n = len(self.history)
        return {"discriminators": items, "session": sess, "history": n}


__all__ = ["DiscriminatorRuntime"]
