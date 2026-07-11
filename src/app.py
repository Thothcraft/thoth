#!/usr/bin/env python3
"""Raspberry Pi entry point for the Thoth dashboard."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path


THOTH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THOTH_ROOT / "src"))
os.environ.setdefault("THOTH_ROOT", str(THOTH_ROOT))
os.environ.setdefault("FLASK_PORT", "5000")

from backend.app import app, device_manager, socketio  # noqa: E402
from backend.config import Config  # noqa: E402


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("thoth")
CREDENTIALS_FILE = Path("/boot/firmware/thoth_credentials.json")


def load_imager_credentials() -> None:
    """Import optional credentials placed on the boot partition."""
    if not CREDENTIALS_FILE.exists():
        return
    try:
        credentials = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
        token = credentials.get("auth_token")
        if not token:
            logger.warning("Ignoring boot credentials without auth_token")
            return

        os.environ["BRAIN_AUTH_TOKEN"] = token
        Config.BRAIN_AUTH_TOKEN = token
        Config.USER_AUTH_TOKEN = token
        if credentials.get("brain_server_url"):
            Config.BRAIN_SERVER_URL = credentials["brain_server_url"]

        config_dir = Path(Config.CONFIG_DIR)
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "auth.json").write_text(json.dumps({"token": token}), encoding="utf-8")
        CREDENTIALS_FILE.unlink(missing_ok=True)
    except Exception as exc:
        logger.error("Failed to import Raspberry Pi Imager credentials: %s", exc)


def main() -> None:
    load_imager_credentials()
    token = getattr(Config, "USER_AUTH_TOKEN", None)
    if token:
        success, message = device_manager.register_device(token)
        if success:
            device_manager.start_heartbeat(Config.HEARTBEAT_INTERVAL)
        else:
            logger.warning("Initial device registration failed: %s", message)

    socketio.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
