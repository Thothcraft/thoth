# Thoth for macOS

A status-bar application that runs a local Thoth server, exposes built-in
sensors (camera and microphone) plus external sensor support, and provides
the same minute-capture dashboard used by the Raspberry Pi build.

## Quick Install (DMG + GUI)

1. Download `Thoth-macOS-Installer.dmg` from the latest release.
2. Open the DMG and run `Install Thoth.command`.
3. Follow the GUI prompts to complete setup.

## Manual Install

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
The dashboard shows continuous capture state, live sensor previews, and
minute folders with download/upload controls.

## Menu Bar Options

| Item               | Action                                       |
|--------------------|----------------------------------------------|
| Open Dashboard     | Opens `http://localhost:5000` in your browser |
| Start / Stop Collection | Toggle sensor data collection           |
| Quit Thoth         | Stop the background server and exit          |

## Sensors

| Sensor     | Status        | Notes                                 |
|------------|---------------|---------------------------------------|
| Camera     | ✅ Working    | FaceTime HD, Continuity Camera        |
| Microphone | ✅ Working    | Built-in mic, requires sox or ffmpeg  |
| IMU        | ☐ External   | Extend for external USB/BLE hardware |
| WiFi CSI   | ☐ External   | Detects ESP32 via `/dev/tty.usbserial*` |

The dashboard uses the shared Thoth core, so minute-long capture views and
saved-minute browsing match the Raspberry Pi experience.

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

### Code Signing and Notarization (required for Gatekeeper-safe installs)

```bash
# Sign the installer script inside the DMG staging folder
codesign --force --options runtime --timestamp \
  --sign "Developer ID Application: YOUR_NAME" \
  "dist/macos-installer/Install Thoth.command"

# Sign and notarize the DMG
codesign --force --options runtime --timestamp \
  --sign "Developer ID Application: YOUR_NAME" \
  "dist/Thoth-macOS-Installer.dmg"
xcrun notarytool submit "dist/Thoth-macOS-Installer.dmg" \
  --apple-id YOU@EMAIL --team-id TEAM_ID --password APP_PASSWORD --wait
xcrun stapler staple "dist/Thoth-macOS-Installer.dmg"
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
