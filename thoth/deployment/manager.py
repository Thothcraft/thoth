"""Deployment lifecycle on the node — receive → validate → install → ack.

Mirrors the Brain v1 deployment state machine (§19) and persists state
so a daemon restart resumes cleanly.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ..models import ModelRegistry
from ..settings import config_dir

logger = logging.getLogger(__name__)

_STATES = ("queued", "received", "validated", "installed",
           "acknowledged", "active", "failed")


class DeploymentManager:
    """Tracks deployment state transitions on the device."""

    def __init__(self, registry: ModelRegistry,
                 path: Optional[Path] = None):
        self.registry = registry
        self.path = path or (config_dir() / "deployments.json")
        self._states: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._states, indent=2))

    def state(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        return self._states.get(deployment_id)

    def _set(self, deployment_id: str, state: str, **extra: Any) -> Dict[str, Any]:
        rec = self._states.setdefault(deployment_id, {"history": []})
        rec["state"] = state
        rec.update(extra)
        rec["history"].append({"state": state})
        self._save()
        return rec

    # -- state machine ---------------------------------------------------------
    def receive(self, deployment_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._set(deployment_id, "received", payload=payload)

    def validate(self, deployment_id: str) -> Dict[str, Any]:
        rec = self._states.get(deployment_id) or {}
        payload = rec.get("payload") or {}
        manifest = payload.get("manifest") or {}
        errors = []
        if manifest.get("format") not in (None, "thoth-model/v1"):
            errors.append("unsupported manifest format")
        processor = payload.get("processor") or manifest.get("processor")
        if processor not in ("rule", "torchscript", "fusion"):
            errors.append(f"unsupported processor {processor!r}")
        sha = payload.get("artifact_sha256")
        blob = payload.get("model_data")
        if sha and isinstance(blob, str):
            actual = hashlib.sha256(blob.encode()).hexdigest()
            if actual != sha:
                errors.append("artifact hash mismatch")
        if errors:
            return self._set(deployment_id, "failed",
                             failure={"stage": "validated",
                                      "code": "manifest_invalid",
                                      "message": "; ".join(errors)})
        return self._set(deployment_id, "validated")

    def install(self, deployment_id: str) -> Dict[str, Any]:
        rec = self._states.get(deployment_id) or {}
        payload = rec.get("payload") or {}
        manifest = payload.get("manifest") or {}
        name = payload.get("name") or manifest.get("name") or "model"
        processor = payload.get("processor") or manifest.get("processor") or "rule"
        config = payload.get("config") or manifest.get("config") or {}
        rm = self.registry.install(
            name=name, processor=processor, config=config,
            deployment_id=deployment_id)
        return self._set(deployment_id, "installed",
                         runtime_model_id=rm.runtime_model_id)

    def acknowledge(self, deployment_id: str) -> Dict[str, Any]:
        rec = self._states.get(deployment_id) or {}
        return self._set(deployment_id, "acknowledged",
                         runtime_model_id=rec.get("runtime_model_id"))

    def activate(self, deployment_id: str) -> Dict[str, Any]:
        rec = self._states.get(deployment_id) or {}
        rmid = rec.get("runtime_model_id")
        if rmid:
            self.registry.activate(rmid, True)
        return self._set(deployment_id, "active", runtime_model_id=rmid)

    def fail(self, deployment_id: str, stage: str, code: str,
             message: str) -> Dict[str, Any]:
        return self._set(deployment_id, "failed",
                         failure={"stage": stage, "code": code,
                                  "message": message})

    def process(self, deployment_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full receive→ack chain for a pushed deployment."""
        self.receive(deployment_id, payload)
        rec = self.validate(deployment_id)
        if rec.get("state") == "failed":
            return rec
        rec = self.install(deployment_id)
        rec = self.acknowledge(deployment_id)
        return rec
