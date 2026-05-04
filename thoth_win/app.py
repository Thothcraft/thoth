#!/usr/bin/env python3
"""Thoth Windows System Tray Application.

Runs as a Windows system-tray icon.  Right-click shows a menu with
options to open the local dashboard in a browser, start / stop data
collection, and quit.  A background thread runs the Flask server from
thoth_core so the dashboard is always available at localhost.
"""

import os
import sys
import threading
import webbrowser
import logging

# ---------------------------------------------------------------------------
# Path setup — make thoth_core importable
# ---------------------------------------------------------------------------
THOTH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, THOTH_ROOT)
os.environ["THOTH_ROOT"] = THOTH_ROOT

import pystray
from PIL import Image, ImageDraw

from thoth_core.backend.config import Config
from thoth_core.backend.app import app, socketio, run_server

# Register Windows-specific sensors
from thoth_win.sensors import camera, microphone, imu, csi  # noqa: F401

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("thoth_win")

DASHBOARD_URL = f"http://localhost:{Config.PORT}"

_collecting = False
_server_thread = None


# ---------------------------------------------------------------------------
# Icon generation (simple "T" icon if no .ico file is provided)
# ---------------------------------------------------------------------------

def _create_icon_image(size=64):
    """Generate a simple tray icon programmatically."""
    img = Image.new("RGBA", (size, size), (102, 126, 234, 255))
    draw = ImageDraw.Draw(img)
    # Draw a "T" letter
    draw.rectangle([10, 10, size - 10, 20], fill="white")
    draw.rectangle([size // 2 - 5, 10, size // 2 + 5, size - 10], fill="white")
    return img


# ---------------------------------------------------------------------------
# Menu callbacks
# ---------------------------------------------------------------------------

def _open_dashboard(icon, item):
    webbrowser.open(DASHBOARD_URL)


def _toggle_collection(icon, item):
    global _collecting
    _collecting = not _collecting
    state = "started" if _collecting else "stopped"
    logger.info("Data collection %s", state)


def _quit_app(icon, item):
    logger.info("Quitting Thoth")
    icon.stop()


def _uninstall_app(icon, item):
    """Run the uninstall script."""
    import subprocess
    script_dir = os.path.dirname(os.path.abspath(__file__))
    uninstall_script = os.path.join(script_dir, "uninstall.ps1")
    
    if os.path.exists(uninstall_script):
        logger.info("Starting uninstallation...")
        # Run uninstall script in a new PowerShell window
        subprocess.Popen([
            "powershell", "-ExecutionPolicy", "Bypass", 
            "-File", uninstall_script
        ], cwd=script_dir)
        icon.stop()
    else:
        logger.error("Uninstall script not found")
        # Fallback: just quit
        icon.stop()


def _collection_label(item):
    return "Stop Collection" if _collecting else "Start Collection"


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def _start_server():
    global _server_thread
    if _server_thread and _server_thread.is_alive():
        return
    _server_thread = threading.Thread(
        target=run_server,
        kwargs={"host": "127.0.0.1", "port": Config.PORT, "debug": False},
        daemon=True,
    )
    _server_thread.start()
    logger.info("Flask server started on %s", DASHBOARD_URL)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    _start_server()

    icon = pystray.Icon(
        name="Thoth",
        icon=_create_icon_image(),
        title="Thoth — Smart Home Sensor Platform",
        menu=pystray.Menu(
            pystray.MenuItem("Open Dashboard", _open_dashboard, default=True),
            pystray.MenuItem(_collection_label, _toggle_collection),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Uninstall Thoth", _uninstall_app),
            pystray.MenuItem("Quit Thoth", _quit_app),
        ),
    )
    icon.run()


if __name__ == "__main__":
    main()
