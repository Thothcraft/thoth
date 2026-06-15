#!/usr/bin/env python3
"""Thoth macOS Status Bar Application.

Runs as a macOS status bar (menu bar) app.  On click it shows a dropdown
with options to open the local dashboard in a browser, start / stop
data collection, and quit.  A background thread runs the Flask server
from thoth_core so the dashboard is always available at localhost.
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

# Tell the core config where the project root is
os.environ["THOTH_ROOT"] = THOTH_ROOT

import rumps

from thoth_core.backend.config import Config
from thoth_core.backend.app import app, socketio, run_server
import thoth_core.backend.app as core_app

# Register macOS-specific sensors so they appear in SensorRegistry
from thoth_mac.sensors import camera, microphone, imu, csi  # noqa: F401

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("thoth_mac")

DASHBOARD_URL = f"http://localhost:{Config.PORT}"

# Resolve icon path (works both from source and inside .app bundle)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ICON_PATH = os.path.join(_SCRIPT_DIR, "icon.png")

# Also check parent directory in case running from extracted archive
if not os.path.exists(_ICON_PATH):
    _ICON_PATH = os.path.join(os.path.dirname(_SCRIPT_DIR), "thoth_mac", "icon.png")


class ThothStatusBarApp(rumps.App):
    """macOS status-bar application for Thoth."""

    def __init__(self):
        icon_file = _ICON_PATH if os.path.exists(_ICON_PATH) else None
        if icon_file:
            logger.info(f"Loading icon from: {icon_file}")
        else:
            logger.warning(f"Icon not found at: {_ICON_PATH}, using text fallback")
        super().__init__(
            name="Thoth",
            title=None if icon_file else "T",
            icon=icon_file,
            quit_button=None,  # we supply our own
            template=False,  # keep custom square "T" icon visible
        )
        self.menu = [
            rumps.MenuItem("Open Dashboard", callback=self.open_dashboard),
            rumps.MenuItem("Start Collection", callback=self.toggle_collection),
            None,  # separator
            rumps.MenuItem("Quit Thoth", callback=self.quit_app),
        ]
        self._collecting = False
        self._server_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _start_server(self):
        """Launch Flask in a daemon thread."""
        if self._server_thread and self._server_thread.is_alive():
            return
        self._server_thread = threading.Thread(
            target=run_server,
            kwargs={"host": "127.0.0.1", "port": Config.PORT, "debug": False},
            daemon=True,
        )
        self._server_thread.start()
        logger.info("Flask server started on %s", DASHBOARD_URL)

    # rumps calls this automatically after the run-loop starts
    @rumps.timer(1)
    def _ensure_server(self, _sender):
        """Make sure the server is alive (checked once per second)."""
        if not self._server_thread or not self._server_thread.is_alive():
            self._start_server()

    # ------------------------------------------------------------------
    # Menu callbacks
    # ------------------------------------------------------------------

    def open_dashboard(self, _sender):
        webbrowser.open(DASHBOARD_URL)

    def toggle_collection(self, sender):
        self._collecting = not self._collecting
        core_app.collection_active = self._collecting
        if self._collecting:
            sender.title = "Stop Collection"
            logger.info("Data collection started")
        else:
            sender.title = "Start Collection"
            logger.info("Data collection stopped")

    def quit_app(self, _sender):
        logger.info("Quitting Thoth")
        rumps.quit_application()


def main():
    ThothStatusBarApp().run()


if __name__ == "__main__":
    main()
