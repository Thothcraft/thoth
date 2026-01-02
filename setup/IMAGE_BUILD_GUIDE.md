# Thoth Raspberry Pi 5 Image Build Guide

## Customer Experience

Customers just:
1. Power on the Thoth device
2. Connect to **"Thoth-AP"** WiFi (password: `thoth123`)
3. Setup page opens → select home WiFi → done!

---

# Building the Image

## Requirements

- Raspberry Pi 5 (4GB or 8GB)
- microSD card (16GB+, 32GB recommended)
- Computer with SD card reader
- [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
- [HDD Raw Copy Tool](https://hddguru.com/software/HDD-Raw-Copy-Tool/) (Windows) or `dd` (Linux/Mac)

---

## Step 1: Flash Raspberry Pi OS

1. Open **Raspberry Pi Imager**
2. Select **Raspberry Pi 5** → **Raspberry Pi OS (other)** → **Raspberry Pi OS Lite (64-bit)**
   - ⚠️ Make sure it says **Lite** (no desktop)
3. Click ⚙️ and configure:

| Setting | Value |
|---------|-------|
| Hostname | `thoth` |
| Username | `pi` |
| Password | `thoth2025` |
| WiFi SSID | Your WiFi network |
| WiFi Password | Your WiFi password |
| Locale | Your country/timezone |
| Enable SSH | ✅ Yes |

4. Flash to SD card

---

## Step 2: Install Thoth

1. Insert SD card into Pi and power on
2. Wait 2-3 minutes for first boot
3. SSH in (try IP if hostname doesn't work):

```bash
ssh pi@thoth.local
# Or: ssh pi@<IP_ADDRESS>
# Password: thoth2025
```

4. Run these commands:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git
git clone https://github.com/Thothcraft/thoth.git
cd thoth
sudo bash setup/prepare-image.sh
```

⏱️ Wait 15-25 minutes for setup to complete.

You should see:
```
==========================================
  Image Preparation Complete!
==========================================
```

---

## Step 3: Shutdown for Imaging

The prepare-image.sh script already cleans up WiFi credentials. Just shutdown:

```bash
sudo shutdown -h now
```

Wait for the green LED to stop blinking.

---

## Step 4: Create the Image

1. Remove SD card from Pi
2. Insert into your computer

### Windows (HDD Raw Copy Tool - Recommended)

1. Download and run [HDD Raw Copy Tool](https://hddguru.com/software/HDD-Raw-Copy-Tool/)
2. Select your SD card as **SOURCE** (shows full size like 32GB)
3. Click **Continue**
4. Select **FILE** as target
5. Save as `thoth-rpi5.img`
6. Wait for completion (~15-30 minutes)

### Linux/Mac
```bash
# Find your SD card
lsblk  # Linux
diskutil list  # Mac

# Create image (replace sdX with your device)
sudo dd if=/dev/sdX of=thoth-rpi5.img bs=4M status=progress
```

---

## Step 5: Compress (Optional)

```bash
gzip -k thoth-rpi5.img
# Creates: thoth-rpi5.img.gz
```

Windows: Use 7-Zip to compress

---

## Step 6: Test

1. Flash `thoth-rpi5.img` to a SD card using Raspberry Pi Imager
2. Insert into Pi and power on
3. Wait 2-3 minutes
4. On your phone, look for **"Thoth-AP"** WiFi
5. Connect (password: `thoth123`)
6. Setup page should open at `http://192.168.4.1:5000`

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Thoth-AP" doesn't appear | Wait 3-5 minutes, check green LED is blinking |
| Captive portal doesn't open | Open browser, go to `http://192.168.4.1:5000` |
| hostapd fails to start | Check `/var/log/thoth-first-boot.log` |
| Need SSH access | Connect Ethernet, `ssh pi@<IP>` (password: `thoth2025`) |

### Debug Commands (via SSH or monitor)
```bash
# Check first-boot log
cat /var/log/thoth-first-boot.log

# Check service status
sudo systemctl status thoth-firstboot
sudo systemctl status thoth-web
sudo systemctl status hostapd

# Manually start hotspot
sudo /home/pi/thoth/setup/first-boot.sh
```
