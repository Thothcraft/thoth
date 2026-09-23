"""Persistent node configuration (~/.thoth/config.json)."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any, Dict, Optional


def config_dir() -> Path:
    root = os.getenv("THOTH_HOME", "~/.thoth")
    path = Path(root).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


class ConfigStore:
    """JSON-backed node config: identity, pairing, local token, runtime."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or (config_dir() / "config.json")
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.save()

    # -- identity -------------------------------------------------------------
    @property
    def device_id(self) -> str:
        did = self._data.get("device_id")
        if not did:
            import uuid
            did = str(uuid.uuid5(uuid.NAMESPACE_DNS,
                                 f"thoth-{os.uname().nodename if hasattr(os,'uname') else 'node'}-{os.getlogin() if hasattr(os,'getlogin') else 'user'}"))
            self.set("device_id", did)
        return did

    @property
    def device_name(self) -> str:
        return self._data.get("device_name") or os.getenv("THOTH_HOSTNAME") or "thoth-node"

    # -- local API token ---------------------------------------------------------
    @property
    def local_token(self) -> str:
        token = self._data.get("local_token")
        if not token:
            token = secrets.token_urlsafe(24)
            self.set("local_token", token)
        return token

    # -- brain ------------------------------------------------------------------
    @property
    def brain_url(self) -> str:
        return self._data.get("brain_url") or os.getenv(
            "BRAIN_URL", "https://api.thothcraft.com")

    @property
    def device_token(self) -> Optional[str]:
        return self._data.get("device_token") or os.getenv("BRAIN_AUTH_TOKEN")
