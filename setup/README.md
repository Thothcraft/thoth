# Thoth Raspberry Pi Deployment Guide

## Overview

This guide explains how to create a custom Raspberry Pi image with Thoth pre-installed, enabling headless setup via WiFi hotspot and captive portal.

## Quick Start (Pre-built Image)

1. Flash the Thoth image to an SD card
2. Insert SD card into Raspberry Pi and power on
3. On your phone/laptop, connect to WiFi: **Thoth-AP** (password: `thoth123`)
4. A captive portal will automatically open (or go to http://192.168.4.1:5000)
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

1. Download **Raspberry Pi OS Lite (64-bit)** from https://www.raspberrypi.com/software/
2. Flash to SD card using Raspberry Pi Imager
3. In Raspberry Pi Imager settings (gear icon):
   - Enable SSH
   - Set username: `pi` and password
   - Configure WiFi temporarily (for initial setup only)
   - Set locale/timezone

### Step 2: Initial Setup via SSH

1. Insert SD card and power on Pi
2. Find Pi's IP address via your router or use `ping raspberrypi.local`
3. SSH into Pi: `ssh pi@<ip-address>`

### Step 3: Install Thoth and Prepare Image

```bash
# Clone the repository
git clone https://github.com/Thothcraft/thoth.git
cd thoth

# Run the image preparation script (does everything automatically)
sudo ./setup/prepare-image.sh
```

The `prepare-image.sh` script will:
- Install all system dependencies (hostapd, dnsmasq, Python, etc.)
- Create Python virtual environment and install packages
- Configure systemd services for first-boot and web app
- Set up hotspot and captive portal configuration
- Clean up for imaging (remove logs, credentials, SSH keys)
- Enable first-boot service

### Step 4: Create the Image

```bash
# Shutdown the Pi
sudo shutdown -h now
```

Then on your computer:

**Windows (Win32DiskImager):**
1. Remove SD card from Pi, insert into computer
2. Open Win32DiskImager
3. Select a file path for the image (e.g., `thoth-v1.0.img`)
4. Select the SD card drive
5. Click "Read" to create the image
6. Compress: Right-click → Send to → Compressed folder

**Linux/Mac:**
```bash
# Find the SD card device
lsblk  # or diskutil list on Mac

# Create image (replace sdX with your device)
sudo dd if=/dev/sdX of=thoth-v1.0.img bs=4M status=progress

# Shrink image (optional, requires PiShrink)
wget https://raw.githubusercontent.com/Drewsif/PiShrink/master/pishrink.sh
chmod +x pishrink.sh
sudo ./pishrink.sh thoth-v1.0.img

# Compress
gzip thoth-v1.0.img
```

## Alternative: Manual Installation

If you prefer to install on an existing Pi without creating an image:

```bash
git clone https://github.com/Thothcraft/thoth.git
cd thoth
sudo ./setup/install.sh
```

## File Structure

```
thoth/
├── setup/
│   ├── prepare-image.sh      # Image preparation script (run this!)
│   ├── first-boot.sh         # First boot initialization
│   ├── install.sh            # Manual installation script
│   ├── thoth-firstboot.service # First boot systemd service
│   ├── thoth-web.service     # Systemd service for web app
│   ├── thoth-hotspot.service # Systemd service for hotspot
│   ├── hotspot-manager.sh    # Hotspot start/stop script
│   ├── connect-wifi.sh       # WiFi connection script
│   ├── hostapd.conf          # Hotspot configuration
│   └── dnsmasq.conf          # DNS/DHCP for captive portal
├── src/backend/
│   └── app.py                # Main Flask application
├── data/
│   └── config/
│       ├── wifi_configured.flag  # Created after WiFi setup
│       └── auth.json             # Saved login credentials
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
