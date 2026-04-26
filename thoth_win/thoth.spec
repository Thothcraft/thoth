# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for Thoth Windows .exe build.

Usage:
    pyinstaller thoth.spec

Output: dist/Thoth/Thoth.exe  (one-dir mode for faster startup)
  or:  dist/Thoth.exe          (one-file mode, slower startup)
"""

import os
import sys

THOTH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(SPECPATH)))

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=[THOTH_ROOT],
    binaries=[],
    datas=[
        # Include thoth_core backend templates and static files
        (os.path.join(THOTH_ROOT, 'thoth_core', 'backend', 'templates'), 'thoth_core/backend/templates'),
        (os.path.join(THOTH_ROOT, 'thoth_core', 'backend', 'static'), 'thoth_core/backend/static'),
        # Include .env
        (os.path.join(THOTH_ROOT, '.env'), '.'),
        # Include icon
        (os.path.join(SPECPATH, 'icon.png'), '.'),
    ],
    hiddenimports=[
        'thoth_core',
        'thoth_core.backend',
        'thoth_core.backend.app',
        'thoth_core.backend.config',
        'thoth_core.backend.auth_manager',
        'thoth_core.backend.device_manager',
        'thoth_core.backend.file_manager',
        'thoth_core.backend.models',
        'thoth_core.backend.dependency_checker',
        'thoth_core.backend.routes',
        'thoth_core.backend.routes.files',
        'thoth_core.sensors',
        'thoth_core.sensors.base',
        'thoth_core.sensors.manager',
        'thoth_core.data_manager',
        'thoth_core.data_manager.manager',
        'thoth_core.data_manager.protocol',
        'thoth_core.data_manager.scanner',
        'thoth_core.fl_client',
        'thoth_win',
        'thoth_win.sensors',
        'thoth_win.sensors.camera',
        'thoth_win.sensors.microphone',
        'thoth_win.sensors.imu',
        'thoth_win.sensors.csi',
        'flask',
        'flask_socketio',
        'flask_cors',
        'engineio',
        'engineio.async_drivers.threading',
        'jinja2',
        'werkzeug',
        'pystray',
        'PIL',
        'cv2',
        'numpy',
        'pandas',
        'requests',
        'dotenv',
        'apscheduler',
        'netifaces',
        'psutil',
        'jwt',
        'serial',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy', 'torch'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Thoth',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # No console window — tray app only
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.png',         # Will need .ico for Windows — see build script
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Thoth',
)
