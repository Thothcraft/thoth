# Thoth - Portable Raspberry Pi Research Device

Thoth is a portable, battery-powered research device built on Raspberry Pi with Sense HAT for IMU data collection, PiSugar for power management, and automatic WiFi/AP switching for remote data access.

## Features

- **Portable Data Collection**: Battery-powered with PiSugar UPS
- **IMU Sensors**: 9-axis motion sensing via Sense HAT (accelerometer, gyroscope, magnetometer)
- **Smart Networking**: Auto-switches between WiFi and AP mode with captive portal
- **Web Interface**: Modern, responsive web UI for device management and monitoring
  - **Captive Portal**: Easy WiFi configuration for first-time setup
  - **Status Dashboard**: Real-time system metrics and sensor data
  - **Authentication**: Secure login system for device access
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

## Web Interface

### Accessing the Web Interface
1. Connect to the Thoth device's WiFi network (default SSID: `Thoth-Device`)
2. Open a web browser and navigate to `http://192.168.4.1` (AP mode) or the device's IP address
3. Log in using the default credentials (username: `admin`, password: `admin`)

### Captive Portal
- Automatically appears when connecting to the device in AP mode
- Scan and connect to available WiFi networks
- Configure network credentials with a user-friendly interface

### Status Dashboard
- Real-time system monitoring (CPU, memory, disk usage)
- Battery level and power status
- Sensor data visualization
- Collection status and control
- System actions (restart services, shutdown)

## API Endpoints

### Authentication
- `GET /login` - Login page
- `POST /login` - Authenticate user
- `GET /logout` - Logout user

### System Control
- `GET /status` - Status dashboard
- `GET /captive-portal` - WiFi configuration portal
- `POST /wifi/config` - Configure WiFi settings
- `POST /api/collection/start` - Start data collection
- `POST /api/collection/stop` - Stop data collection
- `POST /api/system/restart-service` - Restart system services
- `POST /api/system/shutdown` - Shutdown the device

### Sensor Data
- `GET /health` - System status and battery level
- `WS /data/stream` - Live IMU data feed
- `POST /upload` - Upload collected data

## Directory Structure

```
thoth/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── setup/                    # Installation scripts
├── src/                      # Main source code
│   ├── backend/              # Flask app
│   │   ├── static/           # Static files (CSS, JS, images)
│   │   └── templates/        # HTML templates
│   │       ├── base.html     # Base template
│   │       ├── login.html    # Login page
│   │       ├── status.html   # Status dashboard
│   │       └── captive_portal.html  # WiFi setup
│   ├── sensors/              # Sensor collection
│   ├── network/              # WiFi/AP handling
│   └── utils/                # Helpers
├── data/                     # Local storage (gitignored)
└── portal/                   # Research portal docs
```

## Usage

### Web Interface Access
1. **First-time Setup**:
   - Connect to the device's WiFi network (default: Thoth-Device)
   - Open a web browser to access the captive portal
   - Configure your WiFi network settings
   - Log in with default credentials (admin/admin)
   - Change the default password after first login

2. **Status Dashboard**:
   - Monitor system health and sensor data
   - Start/stop data collection
   - View real-time sensor readings
   - Check battery status and power management

### Button Controls
- **Single tap**: Start/stop data collection
- **Double tap**: Switch to AP mode for WiFi configuration
- **Long press**: Safe shutdown

### Data Collection
Data is stored in JSON format in `data/logs/sensors.json` with timestamp, IMU orientation, acceleration, gyroscope, and magnetometer readings. The web interface provides visualization of this data in real-time.

### Security Notes
- Change the default admin password after first login
- The web interface uses session-based authentication
- All sensitive operations require authentication
- The device creates a secure access point with WPA2 encryption

## Development

### Prerequisites
- Python 3.8+
- Node.js 14+ (for frontend development)
- pip

### Setup
```bash
# Clone the repository
git clone <repo-url> thoth
cd thoth

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the development server
cd src/backend
python app.py
```

The web interface will be available at `http://localhost:5000`

### Testing
Run the test suite with:
```bash
pytest tests/
```

## License

MIT License - See LICENSE file for details.
