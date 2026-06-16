# Thoth for Raspberry Pi

A headless edge device application that runs the Thoth sensor platform,
collects data, runs deployed ML models, and communicates with the Brain
cloud server.

## How It Works

**WiFi and account setup are handled entirely by Raspberry Pi Imager** —
no captive portal, no hotspot, no NoDogSplash. After the first login, Thoth
creates a matching local SSH user so the dashboard Connect button can open a
real terminal session on the Pi.

### Burning the Image

1. Download the Thoth RPi image (`.img.gz`)
2. Open **Raspberry Pi Imager**
3. Select the Thoth image
4. Click the **gear icon** (⚙️) and configure:
   - **WiFi SSID** and **password**
   - **SSH** enabled
   - **Hostname**: `thoth`
   - **Username**: `pi`
   - Locale and timezone
5. Flash the SD card and insert into the Raspberry Pi

The GitHub release also ships a ready-to-run Raspberry Pi bundle that
contains `thoth_core/` and `thoth_rpi/` so the minute-capture dashboard and
collector service are available immediately after install.

### Credential File Format

If you want Brain registration on first boot, place
`/boot/firmware/thoth_credentials.json` on the boot partition:

```json
{
    "auth_token": "eyJhbGciOiJIUzI1NiIs...",
    "brain_server_url": "https://web-production-d7d37.up.railway.app"
}
```

The `auth_token` is a JWT generated when a user logs into the Research Portal
and creates a device entry.  If present, the RPi reads this file on first boot,
authenticates with the Brain server, stores the token locally, and **deletes
the file** from the boot partition for security.

### Boot Sequence

```
Power On
    │
    ▼
WiFi connects automatically (Imager config)
    │
    ▼
thoth.service starts
    │
    ▼
app.py optionally reads /boot/firmware/thoth_credentials.json
    │
    ├── Token found → authenticate + delete file
    │
    └── No file → keep running with the current local session
    │
    ▼
Flask dashboard available at http://thoth.local:5000
    │
    ▼
Login creates local SSH user and enables the dashboard Connect page
```

If mDNS is blocked by your laptop or router, open `http://<pi-ip>:5000`
instead. The first boot script enables NetworkManager and Avahi and disables
the old `hostapd`/`dnsmasq` hotspot path so the Imager WiFi settings control
the LAN connection.

## Manual Installation

```bash
git clone https://github.com/Thothcraft/thoth.git
cd thoth/thoth_rpi
sudo ./setup/install.sh
```

## Sensors

| Sensor     | Status     | Notes                              |
|------------|------------|------------------------------------|
| Camera     | ✅ Working | RPi camera module via picamera2    |
| WiFi CSI   | ✅ Working | ESP32 via /dev/ttyUSB*             |

## Creating a Custom Image

See `setup/prepare-image.sh` and the comments therein. It prepares the Pi
for cloning into a reusable SD card image after the software stack and
capture services are installed.
