# Thoth for macOS

A status-bar application that runs a local Thoth server, exposes built-in
sensors (camera, microphone, IMU mock), and provides a browser-based
dashboard for data collection, labelling, and viewing deployed model
predictions.

## Quick Install

```bash
chmod +x install.sh
./install.sh
```

This will:

1. Create a Python virtual environment in `thoth_mac/.venv`
2. Install all dependencies (core + macOS)
3. Register a **LaunchAgent** so Thoth starts on every login
4. Launch the app immediately

After install, look for **𓁟** in your menu bar and click **Open Dashboard**.

## Menu Bar Options

| Item               | Action                                       |
|--------------------|----------------------------------------------|
| Open Dashboard     | Opens `http://localhost:8000` in your browser |
| Start / Stop Collection | Toggle sensor data collection           |
| Quit Thoth         | Stop the background server and exit          |

## Sensors

| Sensor     | Status        | Notes                                 |
|------------|---------------|---------------------------------------|
| Camera     | ✅ Working    | FaceTime HD, Continuity Camera        |
| Microphone | ✅ Working    | Built-in mic, requires sox or ffmpeg  |
| IMU        | 🔶 Mock mode  | Simulated; extend for external USB/BLE|
| WiFi CSI   | 🔶 Scaffolded | Detects ESP32 via `/dev/tty.usbserial*`; mock data when absent |

## Uninstall

```bash
./uninstall.sh
```

## Building a .dmg Installer for Distribution

To package Thoth as a `.dmg` that customers can install by dragging to Applications:

```bash
# One-command build
./build_dmg.sh
```

This will:
1. Build `Thoth.app` via **py2app** (self-contained `.app` bundle with embedded Python)
2. Create `dist/Thoth-Installer-1.0.0.dmg` with a drag-to-Applications layout

### Prerequisites

- `py2app` (installed automatically by the build script)
- Optional: `brew install create-dmg` for a branded DMG with custom background

### Manual Steps

```bash
# Step 1: Build .app
source .venv/bin/activate
python setup_app.py py2app

# Step 2: Create DMG (simple)
hdiutil create -volname Thoth -srcfolder dist/Thoth.app -ov -format UDZO dist/Thoth.dmg

# Step 2 (alt): Branded DMG
create-dmg --volname Thoth --volicon Thoth.icns \
  --icon "Thoth.app" 175 190 --app-drop-link 425 190 \
  dist/Thoth.dmg dist/Thoth.app
```

### Code Signing (for distribution outside the App Store)

```bash
codesign --deep --force --sign "Developer ID Application: YOUR_NAME" dist/Thoth.app
# Then notarize:
xcrun notarytool submit dist/Thoth.dmg --apple-id YOU@EMAIL --team-id TEAM_ID --password APP_PASSWORD
```

## File Structure

```
thoth_mac/
├── app.py              # rumps status bar app + Flask launcher
├── sensors/
│   ├── camera.py       # OpenCV camera
│   ├── microphone.py   # PyAudio / sox / ffmpeg
│   ├── imu.py          # Mock IMU
│   └── csi.py          # ESP32 USB CSI (scaffolded)
├── install.sh          # One-click installer
├── uninstall.sh
├── requirements.txt    # macOS-only deps
└── README.md
```
