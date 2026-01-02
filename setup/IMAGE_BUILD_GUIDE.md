# Thoth Raspberry Pi 5 Image Build Guide

## Customer Experience

Customers just:
1. Power on the Thoth device
2. Connect to **"Thoth-AP"** WiFi (password: `thoth123`)
3. Setup page opens → select home WiFi → done!

---

# Building the Image

## Requirements

- Raspberry Pi 5
- microSD card (16GB+)
- Computer with SD card reader
- [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
- [Win32DiskImager](https://sourceforge.net/projects/win32diskimager/) (Windows) or `dd` (Linux/Mac)

---

## Step 1: Flash Raspberry Pi OS

1. Open **Raspberry Pi Imager**
2. Select **Raspberry Pi 5** → **Raspberry Pi OS Lite (64-bit)**
3. Click ⚙️ and configure:

| Setting | Value |
|---------|-------|
| Hostname | `thoth` |
| Username | `pi` |
| Password | `thoth2025` |
| WiFi SSID | Your WiFi network |
| WiFi Password | Your WiFi password |
| Enable SSH | ✅ Yes |

4. Flash to SD card

---

## Step 2: Install Thoth

1. Insert SD card into Pi and power on
2. Wait 1-2 minutes, then SSH in:

```bash
ssh pi@thoth.local
# Password: thoth2025
```

3. Run these commands:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git
git clone https://github.com/Thothcraft/thoth.git
cd thoth
sudo ./setup/prepare-image.sh
```

Wait 10-20 minutes for setup to complete.

---

## Step 3: Clean Up for Distribution

```bash
# Remove your WiFi credentials
sudo bash -c 'cat > /etc/wpa_supplicant/wpa_supplicant.conf << EOF
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=US
EOF'

# Clear history and shutdown
cat /dev/null > ~/.bash_history
history -c
sudo shutdown -h now
```

---

## Step 4: Create the Image

1. Remove SD card from Pi
2. Insert into your computer
3. Create image:

**Windows:** Use Win32DiskImager → Click **"Read"** → Save as `thoth-rpi5.img`

**Linux/Mac:**
```bash
sudo dd if=/dev/sdX of=thoth-rpi5.img bs=4M status=progress
```

4. Compress:
```bash
gzip -k thoth-rpi5.img
```

---

## Step 5: Test

1. Flash `thoth-rpi5.img.gz` to a **new** SD card
2. Boot Pi (no Ethernet needed)
3. Connect phone to **"Thoth-AP"** (password: `thoth123`)
4. Verify setup page works

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Thoth-AP" doesn't appear | Wait 2-3 minutes |
| Captive portal doesn't open | Go to `http://192.168.4.1:5000` |
| Need SSH access | `ssh pi@thoth.local` (password: `thoth2025`) |
