"""Runtime model registry — persistent runtime_model_id across restarts.

Each installed model gets a stable ``runtime_model_id`` persisted to
``~/.thoth/models.json`` so deployment acknowledgements and predictions
survive daemon restarts (Architecture v3.0 §19).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..settings import config_dir


class RuntimeModel:
    """A model installed on this node."""

    def __init__(self, runtime_model_id: str, name: str, processor: str,
                 config: Dict[str, Any], active: bool = False,
                 artifact_path: Optional[str] = None,
                 deployment_id: Optional[str] = None):
        self.runtime_model_id = runtime_model_id
        self.name = name
        self.processor = processor
        self.config = config
        self.active = active
        self.artifact_path = artifact_path
        self.deployment_id = deployment_id
        self._processor_impl = None  # lazy-loaded whispy processor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runtime_model_id": self.runtime_model_id,
            "name": self.name,
            "processor": self.processor,
            "config": self.config,
            "active": self.active,
            "artifact_path": self.artifact_path,
            "deployment_id": self.deployment_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeModel":
        return cls(
            runtime_model_id=data["runtime_model_id"],
            name=data.get("name", "model"),
            processor=data.get("processor", "rule"),
            config=dict(data.get("config") or {}),
            active=bool(data.get("active", False)),
            artifact_path=data.get("artifact_path"),
            deployment_id=data.get("deployment_id"),
        )

    def processor_impl(self):
        """Lazy-load the whispy processor for this model."""
        if self._processor_impl is None:
            from whispy.processors import create_processor
            artifact = None
            if self.artifact_path:
                try:
                    artifact = Path(self.artifact_path).read_bytes()
                except OSError:
                    artifact = None
            cfg = {**self.config, "processor": self.processor,
                   "name": self.name}
            self._processor_impl = create_processor(cfg, artifact=artifact)
        return self._processor_impl


class ModelRegistry:
    """Persistent registry of installed runtime models."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or (config_dir() / "models.json")
        self._models: Dict[str, RuntimeModel] = self._load()

    def _load(self) -> Dict[str, RuntimeModel]:
        try:
            raw = json.loads(self.path.read_text())
            return {m["runtime_model_id"]: RuntimeModel.from_dict(m)
                    for m in raw.get("models", [])}
        except Exception:
            return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"models": [m.to_dict() for m in self._models.values()]},
            indent=2))

    def install(self, name: str, processor: str, config: Dict[str, Any],
                *, artifact_path: Optional[str] = None,
                deployment_id: Optional[str] = None,
                runtime_model_id: Optional[str] = None) -> RuntimeModel:
        """Install a model; returns the RuntimeModel with a stable ID."""
        rm = RuntimeModel(
            runtime_model_id=runtime_model_id or f"rm-{uuid.uuid4().hex[:12]}",
            name=name, processor=processor, config=config,
            active=False, artifact_path=artifact_path,
            deployment_id=deployment_id)
        self._models[rm.runtime_model_id] = rm
        self._save()
        return rm

    def get(self, runtime_model_id: str) -> Optional[RuntimeModel]:
        return self._models.get(runtime_model_id)

    def activate(self, runtime_model_id: str, active: bool = True) -> bool:
        rm = self._models.get(runtime_model_id)
        if not rm:
            return False
        rm.active = active
        self._save()
        return True

    def remove(self, runtime_model_id: str) -> bool:
        if self._models.pop(runtime_model_id, None) is None:
            return False
        self._save()
        return True

    def list(self) -> List[RuntimeModel]:
        return list(self._models.values())

    def active(self) -> List[RuntimeModel]:
        return [m for m in self._models.values() if m.active]
