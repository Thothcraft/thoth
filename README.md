# Thoth — Edge Smart Home Sensor Platform

Thoth is the edge device component of the [Thothcraft](https://thothcraft.com)
smart home research ecosystem.  It collects multi-modal sensor data, runs
deployed ML models for real-time predictions, and communicates with the Brain
cloud server for training, federated learning, and Home Assistant integration.

## Repository Structure

```
thoth/
├── thoth_core/          # Shared platform-agnostic code
│   ├── backend/         #   Flask web app, auth, device manager
│   ├── data_manager/    #   Data protocol, storage, scanning
│   ├── fl_client/       #   Federated learning client
│   └── sensors/         #   Base sensor classes & manager
│
├── thoth_mac/           # macOS status-bar application
│   ├── app.py           #   rumps menu-bar app + Flask launcher
│   ├── sensors/         #   Camera, mic, IMU (mock), CSI (scaffolded)
│   ├── install.sh       #   One-click installer (LaunchAgent)
│   └── README.md
│
├── thoth_win/           # Windows system-tray application
│   ├── app.py           #   pystray tray app + Flask launcher
│   ├── sensors/         #   Camera, mic, IMU (mock), CSI (scaffolded)
│   ├── install.ps1      #   One-click installer (Scheduled Task)
│   └── README.md
│
├── thoth_rpi/           # Raspberry Pi headless edge device
│   ├── app.py           #   Entry point (reads Imager credentials)
│   ├── sensors/         #   RPi camera (picamera2), CSI
│   ├── setup/           #   systemd service, image build scripts
│   └── README.md
│
├── WS/                  # WiFi Sensing (ESP32 firmware, training, live)
├── .env                 # Environment variables (Brain URL, tokens)
└── LICENSE
```

## Quick Start

### macOS

```bash
cd thoth_mac
chmod +x install.sh
./install.sh
```

### Windows

```powershell
cd thoth_win
powershell -ExecutionPolicy Bypass -File install.ps1
```

### Raspberry Pi

1. Flash the Thoth image with **Raspberry Pi Imager**
2. Configure WiFi in Imager settings
3. Place `thoth_credentials.json` on the boot partition
4. Power on — Thoth starts automatically

For manual install:
```bash
cd thoth_rpi
sudo ./setup/install.sh
```

## Architecture

All three platform variants share **thoth_core** which provides:

- **Flask web dashboard** — sensor status, media gallery, model predictions
- **Data manager** — save/load sensor data with metadata
- **FL client** — federated learning participation
- **Sensor base classes** — extensible `BaseSensor` + `SensorRegistry`

Each platform adds:

| Platform | Status Bar | Sensors | Boot | WiFi CSI |
|----------|-----------|---------|------|----------|
| **macOS** | rumps (menu bar) | Camera, Mic, IMU (mock) | LaunchAgent | ESP32 via `/dev/tty.usbserial*` |
| **Windows** | pystray (tray) | Camera, Mic, IMU (mock) | Scheduled Task | ESP32 via COM ports |
| **RPi** | headless | picamera2, CSI | systemd | ESP32 via `/dev/ttyUSB*` |

## WiFi Sensing (ESP32 CSI)

The `WS/` directory contains ESP32 firmware for Channel State Information
collection.  Mac and Windows apps detect ESP32 devices connected via USB
and read the CSI stream.  See `WS/csi_recv/` and `WS/csi_send/` for the
firmware source.

## License

MIT — see [LICENSE](LICENSE)
