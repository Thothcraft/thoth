# Thoth for Windows

A system-tray application that runs a local Thoth server, exposes built-in
sensors (camera, microphone, IMU mock), and provides a browser-based
dashboard for data collection and model predictions.

## Quick Install

```powershell
# Right-click install.ps1 → Run with PowerShell
# Or run from a terminal:
powershell -ExecutionPolicy Bypass -File install.ps1
```

This will:

1. Create a Python virtual environment in `thoth_win\.venv`
2. Install all dependencies (core + Windows)
3. Register a **Scheduled Task** so Thoth starts on every login
4. Launch the app immediately

## System Tray Menu

| Item               | Action                                       |
|--------------------|----------------------------------------------|
| Open Dashboard     | Opens `http://localhost:8000` in your browser |
| Start / Stop Collection | Toggle sensor data collection           |
| Quit Thoth         | Stop the background server and exit          |

## Sensors

| Sensor     | Status        | Notes                                        |
|------------|---------------|----------------------------------------------|
| Camera     | ✅ Working    | DirectShow backend, built-in or USB          |
| Microphone | ✅ Working    | PyAudio                                      |
| IMU        | 🔶 Mock mode  | Simulated; extend for external USB/BLE       |
| WiFi CSI   | 🔶 Scaffolded | Detects ESP32 via COM ports; mock when absent|

## Uninstall

```powershell
.\uninstall.ps1
```

## Building a .exe Installer for Distribution

### Option A: Standalone .exe (PyInstaller only)

```powershell
.\build_exe.ps1
```

This produces `dist\Thoth\Thoth.exe` — a folder distribution that can be zipped and shared.

### Option B: Professional Installer Wizard (PyInstaller + Inno Setup)

1. Install [Inno Setup 6](https://jrsoftware.org/isinfo.php) (free)
2. Run the build script — it auto-detects Inno Setup:

```powershell
.\build_exe.ps1
```

This produces `dist\Thoth-Setup-1.0.0.exe` — a professional installer with:
- Welcome wizard with Thoth icon
- License agreement page
- Install location picker
- Start Menu + Desktop shortcuts
- "Run at startup" checkbox
- Full uninstaller in Add/Remove Programs

### Manual Build Steps

```powershell
# Step 1: Build with PyInstaller
pyinstaller --name Thoth --icon icon.ico --noconsole `
  --add-data "..\thoth_core\backend\templates;thoth_core\backend\templates" `
  --add-data "..\thoth_core\backend\static;thoth_core\backend\static" `
  --paths ".." app.py

# Step 2: Build installer (optional)
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" thoth_installer.iss
```

### Code Signing (recommended for distribution)

```powershell
# Sign the exe with your code signing certificate
signtool sign /f YourCert.pfx /p PASSWORD /tr http://timestamp.digicert.com /td sha256 dist\Thoth\Thoth.exe
# Sign the installer too
signtool sign /f YourCert.pfx /p PASSWORD /tr http://timestamp.digicert.com /td sha256 dist\Thoth-Setup-1.0.0.exe
```
