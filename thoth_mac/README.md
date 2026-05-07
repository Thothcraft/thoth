# Thoth for macOS

A status-bar application that runs a local Thoth server, exposes built-in
sensors (camera, microphone, IMU mock), and provides a browser-based
dashboard for data collection, labelling, and viewing deployed model
predictions.

## Quick Install (DMG + GUI)

1. Download `Thoth-macOS-Installer.dmg` from the latest release.
2. Open the DMG and run `Install Thoth.command`.
3. Follow the GUI prompts to complete setup.

## Manual Install (CLI fallback)

```bash
chmod +x install.sh
./install.sh
```

This will:

1. Create a Python virtual environment in `thoth_mac/.venv`
2. Install all dependencies (core + macOS)
3. Register a **LaunchAgent** so Thoth starts on every login
4. Launch the app immediately

After install, look for **Thoth** in your menu bar and click **Open Dashboard**.

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

To package Thoth as a `.dmg` with a guided installer entrypoint:

```bash
# One-command build
./build_dmg.sh
```

This will:
1. Bundle `thoth_core`, `thoth_mac`, and installer assets.
2. Create `dist/Thoth-macOS-Installer.dmg` with `Install Thoth.command`.

### Prerequisites

- macOS with `hdiutil` available (default on macOS runners and local machines).

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
├── Install Thoth.command  # GUI-guided installer entrypoint
├── build_dmg.sh        # Build release DMG
├── install.sh          # One-click installer
├── uninstall.sh
├── requirements.txt    # macOS-only deps
└── README.md
```
