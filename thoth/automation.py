"""Automation runtime — portable whispy.Automation specs driven by the SMA loop.

Automations are stored in ``~/.thoth/automations.json`` in the
``whispy-automation/v1`` record format (:class:`whispy.automation.Automation`),
so a spec authored on one node, pushed from Brain, or created in the
dashboard is identical everywhere.

The daemon drives two evaluation paths per tick:

- ``tick(now)`` — time and condition triggers, evaluated against the
  latest prediction context + window features.
- ``on_prediction(prediction, features)`` — event triggers and the
  prediction context for condition/labels gates.

Actions are submitted through ``ActionScheduler`` (debounce/delay/cooldown
still apply — automation ``for_s`` adds a trigger-side hold on top).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from whispy.automation import Automation, should_fire
from whispy.contracts import Prediction

from .settings import config_dir

logger = logging.getLogger(__name__)


class AutomationManager:
    """Persistent automation store + per-tick evaluator."""

    def __init__(self, submit: Callable[[Dict[str, Any], Prediction, str], None],
                 path: Optional[Path] = None):
        self.path = path or (config_dir() / "automations.json")
        self._submit = submit
        self._autos: Dict[str, Automation] = {}
        self._ctx: Dict[str, Any] = {}
        self._feats: Any = None     # latest WindowFeatures (lazy scalars)
        self._load()

    # -- persistence -----------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
            for a in raw.get("automations") or []:
                auto = Automation.from_dict(a)
                self._autos[auto.id] = auto
        except Exception as exc:
            logger.warning("automations.json unreadable: %s", exc)

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(
                {"format": "whispy-automation/v1",
                 "automations": [a.to_dict() for a in self._autos.values()]},
                indent=2))
        except Exception as exc:
            logger.warning("automations save failed: %s", exc)

    # -- CRUD --------------------------------------------------------------------
    def list(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self._autos.values()]

    def get(self, automation_id: str) -> Optional[Automation]:
        return self._autos.get(automation_id)

    def upsert(self, spec: Mapping[str, Any]) -> Automation:
        auto = Automation.from_dict(spec)
        existing = self._autos.get(auto.id)
        if existing is not None:
            # Partial update: unspecified fields keep the existing spec.
            merged = existing.to_dict()
            for k in ("name", "enabled", "trigger", "action"):
                if spec.get(k) is not None:
                    merged[k] = spec[k]
            auto = Automation.from_dict(merged)
            auto.state = existing.state          # keep trigger state
        self._autos[auto.id] = auto
        self._save()
        return auto

    def remove(self, automation_id: str) -> bool:
        if self._autos.pop(automation_id, None) is None:
            return False
        self._save()
        return True

    # -- evaluation ---------------------------------------------------------------
    def on_prediction(self, prediction: Prediction,
                      features: Any = None) -> None:
        """Update the latest-prediction context; fire event triggers."""
        if features is not None:
            self._feats = features
        self._ctx = {
            "label": prediction.label,
            "confidence": prediction.confidence,
            "model_id": prediction.runtime_model_id,
        }
        for auto in self._autos.values():
            if not auto.enabled:
                continue
            if str(auto.trigger.get("type")) != "event":
                continue
            if should_fire(auto, self._ctx, features=self._feats):
                self._fire(auto, prediction)

    def tick(self, now: Optional[float] = None,
             prediction: Optional[Prediction] = None,
             features: Any = None) -> None:
        """Evaluate time and condition triggers for this tick."""
        if features is not None:
            self._feats = features
        for auto in self._autos.values():
            if not auto.enabled:
                continue
            ttype = str(auto.trigger.get("type"))
            if ttype not in ("time", "condition"):
                continue
            if should_fire(auto, self._ctx, now=now, features=self._feats):
                self._fire(auto, prediction)

    def _fire(self, auto: Automation,
              prediction: Optional[Prediction]) -> None:
        action_cfg = dict(auto.action or {})
        if not action_cfg:
            return
        pred = prediction or Prediction(
            label=str(self._ctx.get("label") or ""),
            confidence=float(self._ctx.get("confidence") or 0.0))
        logger.info("automation %s (%s) firing %s",
                    auto.id, auto.name, action_cfg.get("type"))
        auto.state["last_fired_at"] = time.time()
        auto.state["fires"] = int(auto.state.get("fires") or 0) + 1
        self._save()
        self._submit(action_cfg, pred, model_id=f"automation:{auto.id}")


__all__ = ["AutomationManager"]
