"""CLI → daemon IPC over the authenticated loopback local API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from ..settings import ConfigStore


class DaemonUnavailable(RuntimeError):
    pass


class DaemonClient:
    """Thin client for the daemon's local API."""

    def __init__(self, config: Optional[ConfigStore] = None,
                 port: Optional[int] = None, timeout: int = 10):
        self.config = config or ConfigStore()
        self.port = port or int(self.config.get("local_port", 5000))
        self.base = f"http://127.0.0.1:{self.port}"
        self.token = self.config.local_token
        self.timeout = timeout

    def _req(self, method: str, path: str, body: Optional[dict] = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Authorization": f"Bearer {self.token}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                return json.loads(res.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise DaemonUnavailable(
                f"thoth daemon unreachable at {self.base} — is it running? "
                f"({exc})") from exc

    def get(self, path: str) -> Any:
        return self._req("GET", path)

    def post(self, path: str, body: Optional[dict] = None) -> Any:
        return self._req("POST", path, body)

    # convenience
    def status(self) -> Dict[str, Any]:
        return self.get("/api/status")

    def is_running(self) -> bool:
        try:
            self.status()
            return True
        except DaemonUnavailable:
            return False
