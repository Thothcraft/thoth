#!/usr/bin/env python3
"""Thoth Raspberry Pi Entry Point.

On first boot the application:
  1. Reads WiFi credentials from Raspberry Pi Imager config (automatic)
  2. Reads /boot/firmware/thoth_credentials.json for account association
  3. Authenticates with the Brain server using the JWT in that file
  4. Stores the token locally and deletes the credentials file
  5. Starts the Flask dashboard + sensor collection

WiFi is handled entirely by Raspberry Pi Imager — no captive portal,
no NoDogSplash, no hotspot.
"""

import os
import sys
import json
import logging

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
THOTH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, THOTH_ROOT)
os.environ["THOTH_ROOT"] = THOTH_ROOT

from thoth_core.backend.config import Config
from thoth_core.backend.app import socketio, app
from thoth_core.backend.auth_manager import AuthManager

# Register RPi-specific sensors
from thoth_rpi.sensors import camera, csi  # noqa: F401

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("thoth_rpi")

# ---------------------------------------------------------------------------
# Credentials file written by Raspberry Pi Imager
# ---------------------------------------------------------------------------
CREDENTIALS_FILE = "/boot/firmware/thoth_credentials.json"


def load_imager_credentials():
    """Read the JWT credential file placed by Imager on the boot partition.

    Expected format:
        {
            "auth_token": "eyJ...",
            "brain_server_url": "https://..."   (optional override)
        }

    After reading, the file is deleted for security.
    """
    if not os.path.exists(CREDENTIALS_FILE):
        logger.info("No Imager credentials file found at %s — skipping", CREDENTIALS_FILE)
        return

    logger.info("Found Imager credentials file: %s", CREDENTIALS_FILE)
    try:
        with open(CREDENTIALS_FILE, "r") as f:
            creds = json.load(f)

        auth_token = creds.get("auth_token")
        if not auth_token:
            logger.warning("Credentials file exists but contains no auth_token")
            return

        # Apply token to env and config
        os.environ["BRAIN_AUTH_TOKEN"] = auth_token
        Config.BRAIN_AUTH_TOKEN = auth_token
        Config.USER_AUTH_TOKEN = auth_token

        # Optional server URL override
        server_url = creds.get("brain_server_url")
        if server_url:
            os.environ["BRAIN_SERVER_URL"] = server_url
            Config.BRAIN_SERVER_URL = server_url

        # Persist token locally so subsequent boots don't need the file
        os.makedirs(Config.CONFIG_DIR, exist_ok=True)
        local_auth = os.path.join(Config.CONFIG_DIR, "auth.json")
        with open(local_auth, "w") as f:
            json.dump({"token": auth_token}, f)

        logger.info("Credentials loaded and persisted locally")

        # Delete the credentials file from boot partition for security
        try:
            os.remove(CREDENTIALS_FILE)
            logger.info("Deleted credentials file from boot partition")
        except PermissionError:
            logger.warning("Could not delete credentials file — may need root")

    except Exception as e:
        logger.error("Error reading credentials file: %s", e)


def main():
    load_imager_credentials()
    logger.info("Starting Thoth RPi server on port %s", Config.PORT)
    socketio.run(
        app,
        host="0.0.0.0",
        port=Config.PORT,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
