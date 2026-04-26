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
        'opencv-python',
        'numpy',
        'PIL',
        'sounddevice',
        'pyaudio',
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
        'sounddevice',
        'pyaudio',
        'socketio',
        'engineio',
    ],
    'excludes': [
        'matplotlib',
        'pytest',
        'tkinter',
    ],
    'site_packages': True,
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
