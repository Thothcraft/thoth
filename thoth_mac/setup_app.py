"""
Setup script for building Thoth macOS .app bundle using py2app.

Usage:
    python setup_app.py py2app
"""

from setuptools import setup
import os
import sys

APP = ['app.py']
DATA_FILES = [
    'icon.png',
    'Thoth.icns',
]

# Include the sensors directory
for root, dirs, files in os.walk('sensors'):
    for file in files:
        rel_path = os.path.join(root, file)
        DATA_FILES.append(rel_path)

# Add thoth_core to Python path so py2app can find it
THOTH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, THOTH_ROOT)

OPTIONS = {
    'argv_emulation': False,
    'iconfile': 'Thoth.icns',
    'plist': {
        'CFBundleName': 'Thoth',
        'CFBundleDisplayName': 'Thoth',
        'CFBundleIdentifier': 'com.thothcraft.thoth',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'LSUIElement': True,  # Run as menu bar app (no dock icon)
    },
    'packages': [
        'rumps',
        'flask',
        'flask_cors',
        'flask_socketio',
        'werkzeug',
        'requests',
        'apscheduler',
        'thoth_core',
    ],
    'includes': [
        'rumps',
        'flask',
        'flask_cors',
        'flask_socketio',
        'werkzeug',
        'requests',
        'apscheduler',
        'cv2',
        'numpy',
        'PIL',
        'socketio',
        'engineio',
    ],
    'excludes': [
        'matplotlib',
        'pytest',
        'tkinter',
        'gevent',
        'geventwebsocket',
        'eventlet',
    ],
    'site_packages': True,
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
