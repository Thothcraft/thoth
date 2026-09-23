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

    @staticmethod
    def _artifact_bytes(payload: Dict[str, Any]) -> Optional[bytes]:
        """Decode the deployment's model artifact to raw bytes.

        ``model_data`` may arrive as raw bytes or a base64 string; anything
        else is treated as absent.
        """
        import base64
        blob = payload.get("model_data")
        if blob is None:
            return None
        if isinstance(blob, bytes):
            return blob
        if isinstance(blob, str):
            try:
                return base64.b64decode(blob, validate=True)
            except Exception:
                return blob.encode()
        return None

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
        from whispy.contracts import (
            ModelManifest, SUPPORTED_MANIFEST_FORMATS)
        rec = self._states.get(deployment_id) or {}
        payload = rec.get("payload") or {}
        manifest_dict = payload.get("manifest") or {}
        errors = []

        # A supplied manifest must declare a supported contract format
        # explicitly (``whispy-model/v1``; legacy ``thoth-model/v1`` is
        # still accepted during the rename transition).
        manifest = None
        if manifest_dict:
            if manifest_dict.get("format") not in SUPPORTED_MANIFEST_FORMATS:
                errors.append(
                    "manifest format must be one of "
                    f"{sorted(SUPPORTED_MANIFEST_FORMATS)}, got "
                    f"{manifest_dict.get('format')!r}")
            else:
                manifest = ModelManifest.from_dict(manifest_dict)

        processor = payload.get("processor") or (
            manifest.processor if manifest else None)
        if processor not in ("rule", "torchscript", "fusion"):
            errors.append(f"unsupported processor {processor!r}")

        # Artifact: required for torchscript; hash-checked against the
        # manifest's declared sha256 over the artifact *bytes*.
        artifact = self._artifact_bytes(payload)
        if processor == "torchscript" and artifact is None:
            errors.append("torchscript deployment requires a model artifact")
        declared_sha = (manifest.artifact_sha256 if manifest else "") or \
            payload.get("artifact_sha256") or ""
        if declared_sha:
            if artifact is None:
                errors.append("artifact_sha256 declared but no artifact supplied")
            elif hashlib.sha256(artifact).hexdigest() != declared_sha:
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

        # Full processor config: manifest-level contract fields (inputs,
        # outputs, task, rules…) merged with the inline config so nothing
        # the processor needs is dropped.
        config = dict(manifest.get("config") or {})
        config.update(payload.get("config") or {})
        for key in ("inputs", "outputs", "task", "rules", "rule",
                    "else", "params", "on_missing", "actions"):
            if key in manifest and key not in config:
                config[key] = manifest[key]
        config.setdefault("name", name)

        # Persist the artifact bytes so the processor can load them.
        artifact_path = None
        blob = self._artifact_bytes(payload)
        if blob is not None:
            adir = config_dir() / "artifacts"
            adir.mkdir(parents=True, exist_ok=True)
            apath = adir / f"{deployment_id}.bin"
            apath.write_bytes(blob)
            artifact_path = str(apath)

        rm = self.registry.install(
            name=name, processor=processor, config=config,
            artifact_path=artifact_path, deployment_id=deployment_id,
            runtime_model_id=rec.get("runtime_model_id"))
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
        """Run the full receive→ack chain for a pushed deployment.

        ``deployment_id`` is the idempotency key: if this deployment already
        reached ``installed``/``acknowledged``/``active`` with a runtime
        model, a redelivery (e.g. a lost ack) resumes by re-acknowledging
        the existing install instead of creating a duplicate runtime model.
        """
        existing = self._states.get(deployment_id)
        if existing and existing.get("runtime_model_id") and \
                existing.get("state") in ("installed", "acknowledged", "active"):
            return self.acknowledge(deployment_id)
        self.receive(deployment_id, payload)
        rec = self.validate(deployment_id)
        if rec.get("state") == "failed":
            return rec
        rec = self.install(deployment_id)
        rec = self.acknowledge(deployment_id)
        return rec
