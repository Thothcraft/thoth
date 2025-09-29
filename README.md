# Thoth - Portable Raspberry Pi Research Device

Thoth is a portable, battery-powered research device built on Raspberry Pi with Sense HAT for IMU data collection, PiSugar for power management, and automatic WiFi/AP switching for remote data access.

## Features

- **Portable Data Collection**: Battery-powered with PiSugar UPS
- **IMU Sensors**: 9-axis motion sensing via Sense HAT (accelerometer, gyroscope, magnetometer)
- **Smart Networking**: Auto-switches between WiFi and AP mode with captive portal
- **Remote Control**: Flask backend with WebSocket live feeds and REST API
- **Button Control**: Programmable PiSugar button for collection start/stop, AP mode, shutdown
- **Data Upload**: Batch upload to cloud services or research servers

## Hardware Requirements

- Raspberry Pi 4 (recommended) or Zero 2 W
- Sense HAT (stacks on GPIO pins)
- PiSugar 2 Pro (5000mAh) for Pi 4, or PiSugar 2 for Pi Zero
- MicroSD card (32GB+)

## Quick Start

1. **Hardware Assembly**: Stack Sense HAT on Pi GPIO, PiSugar on bottom via pogo pins
2. **Software Setup**: 
   ```bash
   git clone <repo-url> thoth
   cd thoth
   ./setup/install_deps.sh
   ./setup/enable_services.sh
   ./setup/configure_pisugar.sh
   ```
3. **Test**: Press PiSugar button to start collection, check logs in `data/logs/`

## API Endpoints

- `GET /health` - System status and battery level
- `POST /control/start` - Start sensor collection
- `POST /control/stop` - Stop sensor collection
- `WS /data/stream` - Live IMU data feed
- `POST /upload` - Upload collected data
- `POST /config/button` - Configure button actions

## Directory Structure

```
thoth/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── setup/                    # Installation scripts
├── src/                      # Main source code
│   ├── backend/              # Flask app
│   ├── sensors/              # Sensor collection
│   ├── network/              # WiFi/AP handling
│   └── utils/                # Helpers
├── data/                     # Local storage (gitignored)
└── portal/                   # Research portal docs
```

## Usage

### Button Controls
- **Single tap**: Start/stop data collection
- **Double tap**: Switch to AP mode for WiFi configuration
- **Long press**: Safe shutdown

### Data Collection
Data is stored in JSON format in `data/logs/sensors.json` with timestamp, IMU orientation, acceleration, gyroscope, and magnetometer readings.

### Remote Access
Connect to the device's IP address on port 5000 to access the web interface and API.

## Development

```bash
source venv/bin/activate
cd src/backend
python app.py
```

## License

MIT License - See LICENSE file for details.
