"""py2app setup — builds Thoth.app macOS application bundle.

Usage:
    python setup_app.py py2app

The output lands in dist/Thoth.app.
"""

import os
from setuptools import setup

THOTH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

APP = ["app.py"]
DATA_FILES = []

OPTIONS = {
    "argv_emulation": False,
    "iconfile": "Thoth.icns",
    "plist": {
        "CFBundleName": "Thoth",
        "CFBundleDisplayName": "Thoth",
        "CFBundleIdentifier": "com.thothcraft.thoth",
        "CFBundleVersion": "1.0.0",
        "CFBundleShortVersionString": "1.0.0",
        "LSUIElement": True,  # hide from Dock (status-bar only)
        "NSHumanReadableCopyright": "© 2026 Thothcraft",
        "LSEnvironment": {
            "THOTH_ROOT": THOTH_ROOT,
        },
    },
    "packages": [
        "thoth_core",
        "thoth_mac",
        "flask",
        "flask_socketio",
        "flask_cors",
        "jinja2",
        "werkzeug",
        "rumps",
        "cv2",
        "numpy",
        "pandas",
        "requests",
        "dotenv",
        "apscheduler",
        "netifaces",
        "psutil",
        "jwt",
    ],
    "includes": [
        "thoth_core.backend.app",
        "thoth_core.backend.config",
        "thoth_core.backend.auth_manager",
        "thoth_core.backend.device_manager",
        "thoth_core.backend.file_manager",
        "thoth_core.backend.models",
        "thoth_core.sensors",
        "thoth_core.data_manager",
        "thoth_core.fl_client",
        "thoth_mac.sensors.camera",
        "thoth_mac.sensors.microphone",
        "thoth_mac.sensors.imu",
        "thoth_mac.sensors.csi",
    ],
    "resources": [
        os.path.join(THOTH_ROOT, "thoth_core", "backend", "templates"),
        os.path.join(THOTH_ROOT, "thoth_core", "backend", "static"),
        os.path.join(THOTH_ROOT, ".env"),
    ],
}

setup(
    name="Thoth",
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
