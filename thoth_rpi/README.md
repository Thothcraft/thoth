# Thoth for Raspberry Pi

A headless edge device application that runs the Thoth sensor platform,
collects data, runs deployed ML models, and communicates with the Brain
cloud server.

## How It Works

**WiFi and account setup are handled entirely by Raspberry Pi Imager** —
no captive portal, no hotspot, no NoDogSplash.

### Burning the Image

1. Download the Thoth RPi image (`.img.gz`)
2. Open **Raspberry Pi Imager**
3. Select the Thoth image
4. Click the **gear icon** (⚙️) and configure:
   - **WiFi SSID** and **password**
   - **SSH** enabled
   - **Hostname**, locale, timezone
5. Place `thoth_credentials.json` on the boot partition:
   - Download from the **Research Portal** → Devices → Add Device → Download Key
   - Copy to `/boot/firmware/thoth_credentials.json`
6. Flash the SD card and insert into the Raspberry Pi

### Credential File Format

`/boot/firmware/thoth_credentials.json`:

```json
{
    "auth_token": "eyJhbGciOiJIUzI1NiIs...",
    "brain_server_url": "https://web-production-d7d37.up.railway.app"
}
```

The `auth_token` is a JWT generated when a user logs into the Research Portal
and creates a device entry.  The RPi reads this file on first boot,
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
app.py reads /boot/firmware/thoth_credentials.json
    │
    ├── Token found → authenticate + delete file
    │
    └── No file → use previously saved auth
    │
    ▼
Flask dashboard available at http://<pi-ip>:8000
```

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

See `setup/prepare-image.sh` and the comments therein.
