# Thoth Raspberry Pi Deployment Guide

## Overview

This guide explains how to create a custom Raspberry Pi image with Thoth pre-installed, enabling headless setup via WiFi hotspot.

## Quick Start (Pre-built Image)

1. Flash the Thoth image to an SD card
2. Insert SD card into Raspberry Pi and power on
3. On your phone/laptop, connect to WiFi: **Thoth-AP** (password: `thoth123`)
4. Open browser → automatically redirects to setup page
5. Select your home WiFi network and enter password
6. Login with your Thoth platform credentials
7. Device connects to your WiFi and shows its new IP address
8. Reconnect your phone/laptop to your home WiFi
9. Access Thoth at the displayed IP address

## Creating the Custom Image

### Prerequisites

- Raspberry Pi (3B+, 4, or Zero 2 W recommended)
- 16GB+ microSD card
- Another computer with SD card reader
- Raspberry Pi Imager or balenaEtcher

### Step 1: Flash Base Image

1. Download Raspberry Pi OS Lite (64-bit) from https://www.raspberrypi.com/software/
2. Flash to SD card using Raspberry Pi Imager
3. Enable SSH by creating empty file `ssh` in boot partition
4. (Optional) Set default user in `userconf.txt`

### Step 2: First Boot Setup

1. Insert SD card and power on Pi
2. Connect via Ethernet or find IP via router
3. SSH into Pi: `ssh pi@<ip-address>` (default password: `raspberry`)
4. Change password: `passwd`

### Step 3: Install Thoth

```bash
# Clone the repository
git clone https://github.com/your-repo/thoth.git
cd thoth

# Run installation script
sudo ./setup/install.sh
```

### Step 4: Create Custom Image

After installation, you can create a reusable image:

```bash
# On the Pi, clean up
sudo apt-get clean
sudo rm -rf /var/cache/apt/archives/*
sudo rm -rf /tmp/*
sudo rm -f /home/pi/thoth/data/config/wifi_configured.flag
sudo rm -f /home/pi/thoth/data/config/auth.json

# Shutdown
sudo shutdown -h now
```

Then on your computer:
1. Remove SD card from Pi
2. Use Win32DiskImager (Windows) or `dd` (Linux/Mac) to create image
3. Compress: `gzip thoth-image.img`

## File Structure

```
thoth/
├── setup/
│   ├── install.sh           # Main installation script
│   ├── thoth-web.service    # Systemd service for web app
│   ├── thoth-hotspot.service # Systemd service for hotspot
│   ├── hotspot-manager.sh   # Hotspot start/stop script
│   ├── connect-wifi.sh      # WiFi connection script
│   ├── hostapd.conf         # Hotspot configuration
│   └── dnsmasq.conf         # DNS/DHCP for captive portal
├── src/backend/
│   └── app.py               # Main Flask application
├── data/
│   └── config/
│       ├── wifi_configured.flag  # Created after WiFi setup
│       └── wifi_credentials.json # Saved WiFi passwords
└── logs/
    └── thoth.log
```

## How It Works

### Boot Sequence

1. **thoth-hotspot.service** starts first
   - Checks if `wifi_configured.flag` exists
   - If NO: Starts hotspot (Thoth-AP at 192.168.4.1)
   - If YES: Skips hotspot, normal WiFi connects

2. **thoth-web.service** starts after
   - Runs Flask app on port 5000
   - In hotspot mode: accessible at http://192.168.4.1:5000
   - In WiFi mode: accessible at assigned IP

### Setup Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    POWER ON                                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ wifi_configured.flag  │
              │       exists?         │
              └───────────────────────┘
                    │           │
                   NO          YES
                    │           │
                    ▼           ▼
         ┌──────────────┐  ┌──────────────┐
         │ Start Hotspot│  │ Connect WiFi │
         │  Thoth-AP    │  │              │
         └──────────────┘  └──────────────┘
                    │           │
                    ▼           ▼
         ┌──────────────┐  ┌──────────────┐
         │ User connects│  │ Start Web App│
         │ to Thoth-AP  │  │ on WiFi IP   │
         └──────────────┘  └──────────────┘
                    │
                    ▼
         ┌──────────────┐
         │ Setup Page   │
         │ Select WiFi  │
         │ Enter creds  │
         └──────────────┘
                    │
                    ▼
         ┌──────────────┐
         │ Connect WiFi │
         │ Stop Hotspot │
         │ Show new IP  │
         └──────────────┘
```

## Configuration

### Hotspot Settings

Edit `setup/hostapd.conf`:
- `ssid=Thoth-AP` - Network name
- `wpa_passphrase=thoth123` - Password

### Web App Settings

Edit `src/backend/.env`:
- `BRAIN_SERVER_URL` - Thoth platform URL
- `DEVICE_NAME` - Device display name

## Troubleshooting

### Can't see Thoth-AP network
```bash
sudo systemctl status hostapd
sudo journalctl -u hostapd -n 50
```

### Web app not starting
```bash
sudo systemctl status thoth-web
sudo journalctl -u thoth-web -n 50
```

### Reset to factory (start hotspot again)
```bash
sudo rm /home/pi/thoth/data/config/wifi_configured.flag
sudo systemctl restart thoth-hotspot
sudo systemctl restart thoth-web
```

## Security Notes

- Change default hotspot password before deployment
- The hotspot is only active during initial setup
- WiFi credentials are stored locally on the device
- Platform credentials are used to register with Brain server
