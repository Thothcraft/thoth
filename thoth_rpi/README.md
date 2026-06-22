# Thoth for Raspberry Pi

A headless edge device application that runs the Thoth sensor platform,
collects data, runs deployed ML models, and communicates with the Brain
cloud server.

## How It Works

WiFi and account setup should normally be handled by Raspberry Pi Imager.
If the image burner does not expose customization options for custom images,
Thoth also supports a boot-partition provisioning file. There is no captive
portal, hotspot, or NoDogSplash path. After the first login, Thoth creates a
matching local SSH user so the dashboard Connect button can open a real
terminal session on the Pi.

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

If your burner does not show the gear/customization options for this image:

1. Flash the Thoth image.
2. Remove and reinsert the SD card so the boot partition mounts on your
   computer.
3. Copy `setup/thoth_provisioning.json.example` to the boot partition as
   `thoth_provisioning.json`.
4. Edit the WiFi, SSH, hostname, timezone, and optional Brain credentials.
5. Eject the SD card and boot the Raspberry Pi.

On first boot Thoth reads `/boot/firmware/thoth_provisioning.json` or
`/boot/thoth_provisioning.json`, configures NetworkManager WiFi, applies the
hostname/timezone/SSH settings, writes Brain credentials for the app, and
deletes the provisioning file because it contains secrets.

The GitHub release image includes the full `thoth` folder at
`/home/pi/Desktop/thoth`. The dashboard and continuous minute collector use
that folder directly after first boot.

### Provisioning File Format

Use this all-in-one file when Raspberry Pi Imager cannot customize the image:

```json
{
  "wifi": {
    "ssid": "YOUR_WIFI_NAME",
    "password": "YOUR_WIFI_PASSWORD",
    "country": "US",
    "hidden": false
  },
  "ssh_enabled": true,
  "hostname": "thoth",
  "timezone": "America/Toronto",
  "thoth_credentials": {
    "auth_token": "eyJhbGciOiJIUzI1NiIs...",
    "brain_server_url": "https://web-production-d7d37.up.railway.app"
  }
}
```

For open WiFi networks, omit `password` or leave it empty. The file also
accepts flat compatibility keys such as `wifi_ssid`, `wifi_password`,
`auth_token`, and `brain_server_url`.

### Brain Credential File Format

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
thoth-firstboot.service applies optional thoth_provisioning.json
    │
    ▼
thoth-web.service starts
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
mkdir -p ~/Desktop
mv thoth ~/Desktop/thoth
cd ~/Desktop/thoth/thoth_rpi
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
