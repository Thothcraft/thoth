# Thoth Raspberry Pi 5 Image Build Guide

This guide explains how to create a distributable Raspberry Pi OS image with Thoth pre-installed. End users can simply flash this image to an SD card, boot their Pi, and immediately see the Thoth captive portal for WiFi setup.

---

## Overview

**What you'll create:** A `.img.gz` file (~2-3GB) that customers can download and flash to their SD card.

**What customers experience:**
1. Flash image to SD card
2. Insert into Raspberry Pi 5 and power on
3. Connect phone/laptop to "Thoth-AP" WiFi (password: thoth123)
4. Captive portal automatically opens
5. Select home WiFi and login to Thoth platform
6. Device connects to home WiFi and is ready to use

---

## Prerequisites

### Hardware Needed (for building the image)
- Raspberry Pi 5 (4GB or 8GB)
- 16GB+ microSD card (32GB recommended)
- USB keyboard (for initial setup if needed)
- HDMI monitor (optional)
- Ethernet cable OR temporary WiFi access
- Another computer with SD card reader

### Software Needed
- [Raspberry Pi Imager](https://www.raspberrypi.com/software/) (Windows/Mac/Linux)
- [Win32DiskImager](https://sourceforge.net/projects/win32diskimager/) (Windows) OR `dd` (Linux/Mac)
- SSH client (Windows Terminal, PuTTY, or Terminal on Mac/Linux)

---

## Step 1: Flash Base Raspberry Pi OS

### 1.1 Download and Install Raspberry Pi Imager
- Go to https://www.raspberrypi.com/software/
- Download and install for your OS

### 1.2 Flash Raspberry Pi OS Lite (64-bit)

1. Insert your SD card into your computer
2. Open Raspberry Pi Imager
3. Click **"Choose Device"** → Select **Raspberry Pi 5**
4. Click **"Choose OS"** → **Raspberry Pi OS (other)** → **Raspberry Pi OS Lite (64-bit)**
   - ⚠️ Use **Lite** version (no desktop) - smaller and faster
5. Click **"Choose Storage"** → Select your SD card

### 1.3 Configure Initial Settings (IMPORTANT)

Before flashing, click the **gear icon ⚙️** (or "Edit Settings"):

**General tab:**
- ✅ Set hostname: `thoth`
- ✅ Set username and password:
  - Username: `pi`
  - Password: `thoth2025` (temporary, will be removed from final image)
- ✅ Configure wireless LAN (TEMPORARY - for initial setup only):
  - SSID: Your current WiFi network
  - Password: Your WiFi password
  - Country: Your country code (e.g., US)
- ✅ Set locale settings:
  - Time zone: Your timezone
  - Keyboard layout: Your layout

**Services tab:**
- ✅ Enable SSH
- ✅ Use password authentication

6. Click **"Save"** then **"Write"**
7. Wait for flashing to complete (~5-10 minutes)

---

## Step 2: First Boot and SSH Access

### 2.1 Boot the Raspberry Pi

1. Insert the SD card into your Raspberry Pi 5
2. Connect Ethernet cable (recommended) OR rely on WiFi configured above
3. Connect power - the Pi will boot

### 2.2 Find the Pi's IP Address

**Option A: Check your router**
- Log into your router admin page
- Look for a device named "thoth" or "raspberrypi"

**Option B: Use hostname (may not work on all networks)**
```bash
ping thoth.local
# or
ping raspberrypi.local
```

**Option C: Use network scanner**
- Windows: Advanced IP Scanner
- Mac/Linux: `nmap -sn 192.168.1.0/24`

### 2.3 SSH into the Pi

```bash
ssh pi@<IP_ADDRESS>
# Password: thoth2025 (or whatever you set)
```

Example:
```bash
ssh pi@192.168.1.105
```

---

## Step 3: Install Thoth

### 3.1 Update the System

```bash
sudo apt update && sudo apt upgrade -y
```

### 3.2 Install Git and Clone Thoth

```bash
sudo apt install -y git
git clone https://github.com/Thothcraft/thoth.git
cd thoth
```

### 3.3 Run the Image Preparation Script

```bash
sudo ./setup/prepare-image.sh
```

This script will:
- Install all dependencies (Python, hostapd, dnsmasq, etc.)
- Create Python virtual environment
- Install Python packages
- Configure systemd services
- Set up hotspot and captive portal
- Clean up for imaging

**This takes 10-20 minutes.** Wait for it to complete.

---

## Step 4: Final Cleanup Before Imaging

The `prepare-image.sh` script does most cleanup, but verify these manually:

### 4.1 Remove Your Temporary WiFi Credentials

```bash
# Remove the WiFi config you used for setup
sudo rm -f /etc/wpa_supplicant/wpa_supplicant.conf

# Create empty wpa_supplicant.conf
sudo bash -c 'cat > /etc/wpa_supplicant/wpa_supplicant.conf << EOF
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=US
EOF'
```

### 4.2 Verify First-Boot Service is Enabled

```bash
sudo systemctl is-enabled thoth-firstboot.service
# Should output: enabled
```

### 4.3 Verify Cleanup Flags

```bash
# These should NOT exist:
ls -la /etc/thoth-first-boot-done          # Should say "No such file"
ls -la /home/pi/thoth/data/config/wifi_configured.flag  # Should say "No such file"
ls -la /home/pi/thoth/data/config/auth.json             # Should say "No such file"
```

### 4.4 Clear Command History

```bash
cat /dev/null > ~/.bash_history
history -c
```

### 4.5 Shutdown

```bash
sudo shutdown -h now
```

**Wait for the Pi to fully shut down** (green LED stops blinking).

---

## Step 5: Create the Image

### 5.1 Remove SD Card

1. Disconnect power from Pi
2. Remove the SD card
3. Insert into your computer's SD card reader

### 5.2 Create Image File

#### Windows (Win32DiskImager)

1. Open Win32DiskImager
2. In "Image File" field, enter a path like: `C:\Users\YourName\Desktop\thoth-rpi5-v1.0.img`
3. Select your SD card drive letter
4. Click **"Read"** (NOT Write!)
5. Wait for completion (~15-30 minutes for 16GB card)

#### Linux

```bash
# Find your SD card device
lsblk

# Create the image (replace sdX with your device, e.g., sdb)
sudo dd if=/dev/sdX of=thoth-rpi5-v1.0.img bs=4M status=progress

# This creates a full-size image (same size as SD card)
```

#### macOS

```bash
# Find your SD card
diskutil list

# Unmount (but don't eject)
diskutil unmountDisk /dev/diskN

# Create image (replace N with your disk number)
sudo dd if=/dev/rdiskN of=thoth-rpi5-v1.0.img bs=4m status=progress
```

---

## Step 6: Shrink the Image (Recommended)

The raw image is the full size of your SD card (16GB+). Shrinking it reduces download size.

### Linux Only (using PiShrink)

```bash
# Download PiShrink
wget https://raw.githubusercontent.com/Drewsif/PiShrink/master/pishrink.sh
chmod +x pishrink.sh

# Shrink the image
sudo ./pishrink.sh thoth-rpi5-v1.0.img

# This reduces it to ~3-4GB
```

### Alternative: Use a Smaller SD Card

Use an 8GB SD card for building - the image will be smaller.

---

## Step 7: Compress the Image

```bash
# Linux/Mac
gzip -k thoth-rpi5-v1.0.img
# Creates: thoth-rpi5-v1.0.img.gz (~1-2GB)

# Or use zip for Windows compatibility
zip thoth-rpi5-v1.0.zip thoth-rpi5-v1.0.img
```

On Windows, right-click the .img file → "Send to" → "Compressed (zipped) folder"

---

## Step 8: Test the Image

**IMPORTANT:** Test on a different SD card before distributing!

1. Flash the compressed image to a NEW SD card using Raspberry Pi Imager
2. Insert into a Raspberry Pi 5
3. Power on
4. On your phone, look for "Thoth-AP" WiFi network
5. Connect (password: `thoth123`)
6. Captive portal should automatically open
7. Verify you can select WiFi and the setup flow works

---

## Step 9: Distribute to Customers

### Hosting Options

1. **GitHub Releases** - Free, up to 2GB per file
2. **Google Drive / Dropbox** - Easy sharing
3. **AWS S3 / DigitalOcean Spaces** - Professional, scalable
4. **Your own website** - Full control

### Customer Instructions

Provide customers with these simple steps:

```
THOTH DEVICE SETUP

1. Download the Thoth image: [your-download-link]
2. Download Raspberry Pi Imager: https://www.raspberrypi.com/software/
3. Flash the image to a microSD card (16GB+)
4. Insert SD card into your Thoth device and power on
5. Wait 1-2 minutes for the device to boot
6. On your phone/laptop, connect to WiFi: "Thoth-AP" (password: thoth123)
7. A setup page will automatically open
8. Select your home WiFi network and enter the password
9. Login with your Thoth account (or create one)
10. Done! Your Thoth device is now connected.

Need help? Contact support@thoth.com
```

---

## Troubleshooting

### Thoth-AP network doesn't appear
- Wait 2-3 minutes after powering on
- Make sure you're using a Raspberry Pi 5
- The Pi's green LED should be blinking occasionally

### Captive portal doesn't open automatically
- Manually go to http://192.168.4.1:5000 in your browser
- Try a different browser
- On iPhone, wait 10-15 seconds after connecting

### WiFi connection fails
- Verify the WiFi password is correct
- Make sure your router is 2.4GHz compatible (5GHz only won't work)
- Try moving closer to your router

### SSH access for debugging
Default credentials (before customer setup):
- Hostname: thoth.local
- Username: pi
- Password: thoth2025

---

## Security Checklist Before Distribution

- [ ] Removed your personal WiFi credentials
- [ ] Removed SSH keys (`/etc/ssh/ssh_host_*`)
- [ ] Cleared bash history
- [ ] No personal files in `/home/pi/`
- [ ] Changed default hotspot password from `thoth123` (optional but recommended)
- [ ] Tested on fresh hardware

---

## Updating the Image

When you release a new version:

1. Flash your existing image to an SD card
2. Boot and SSH in
3. Update Thoth:
   ```bash
   cd /home/pi/thoth
   git pull
   pip install -r requirements.txt
   ```
4. Re-run cleanup:
   ```bash
   sudo rm -f /etc/thoth-first-boot-done
   sudo rm -f /home/pi/thoth/data/config/*
   sudo rm -f /etc/ssh/ssh_host_*
   cat /dev/null > ~/.bash_history
   sudo shutdown -h now
   ```
5. Create new image

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | YYYY-MM-DD | Initial release |

