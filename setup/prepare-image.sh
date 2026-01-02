#!/bin/bash
# Thoth Raspberry Pi Image Preparation Script
# Run this on a fresh Raspberry Pi OS to prepare it for imaging
# After running this, the SD card can be cloned as a Thoth image

set -e

echo "=========================================="
echo "  Thoth Image Preparation Script"
echo "=========================================="
echo ""
echo "This script prepares a Raspberry Pi for Thoth deployment."
echo "After completion, the SD card can be imaged and distributed."
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./prepare-image.sh)"
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THOTH_DIR="$(dirname "$SCRIPT_DIR")"
THOTH_USER="pi"

echo "[1/10] Updating system packages..."
apt-get update
apt-get upgrade -y

echo "[2/10] Installing system dependencies..."
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    hostapd \
    dnsmasq \
    dhcpcd5 \
    wireless-tools \
    net-tools \
    git \
    iptables \
    rfkill \
    dos2unix

echo "[3/10] Installing Thoth to /home/pi/thoth..."
# Copy thoth to standard location if not already there
if [ "$THOTH_DIR" != "/home/pi/thoth" ]; then
    mkdir -p /home/pi/thoth
    cp -r "$THOTH_DIR"/* /home/pi/thoth/
    THOTH_DIR="/home/pi/thoth"
    SCRIPT_DIR="$THOTH_DIR/setup"
fi

echo "[4/10] Creating Python virtual environment..."
cd "$THOTH_DIR"
python3 -m venv venv
source venv/bin/activate

echo "[5/10] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[6/10] Setting up Thoth directories..."
mkdir -p "$THOTH_DIR/data/config"
mkdir -p "$THOTH_DIR/logs"
chown -R "$THOTH_USER:$THOTH_USER" "$THOTH_DIR"

echo "[7/10] Installing systemd services..."
# Copy service files
cp "$SCRIPT_DIR/thoth-web.service" /etc/systemd/system/
cp "$SCRIPT_DIR/thoth-hotspot.service" /etc/systemd/system/
cp "$SCRIPT_DIR/thoth-firstboot.service" /etc/systemd/system/

# Ensure paths are correct
sed -i "s|/home/pi/thoth|$THOTH_DIR|g" /etc/systemd/system/thoth-web.service
sed -i "s|/home/pi/thoth|$THOTH_DIR|g" /etc/systemd/system/thoth-hotspot.service
sed -i "s|/home/pi/thoth|$THOTH_DIR|g" /etc/systemd/system/thoth-firstboot.service

# Fix Windows line endings (CRLF -> LF) on all scripts and configs
echo "Fixing line endings on scripts and configs..."
find "$SCRIPT_DIR" -type f \( -name "*.sh" -o -name "*.conf" -o -name "*.service" \) -exec dos2unix {} \; 2>/dev/null || \
    find "$SCRIPT_DIR" -type f \( -name "*.sh" -o -name "*.conf" -o -name "*.service" \) -exec sed -i 's/\r$//' {} \;

# Also fix the copied config files
dos2unix /etc/hostapd/hostapd.conf 2>/dev/null || sed -i 's/\r$//' /etc/hostapd/hostapd.conf
dos2unix /etc/dnsmasq.conf 2>/dev/null || sed -i 's/\r$//' /etc/dnsmasq.conf

# Make scripts executable
chmod +x "$SCRIPT_DIR/first-boot.sh"
chmod +x "$SCRIPT_DIR/hotspot-manager.sh"
chmod +x "$SCRIPT_DIR/connect-wifi.sh"

# Reload systemd
systemctl daemon-reload

echo "[8/10] Configuring hostapd and dnsmasq..."
# Backup original configs
[ -f /etc/hostapd/hostapd.conf ] && cp /etc/hostapd/hostapd.conf /etc/hostapd/hostapd.conf.backup
[ -f /etc/dnsmasq.conf ] && cp /etc/dnsmasq.conf /etc/dnsmasq.conf.backup

# Copy our configs
cp "$SCRIPT_DIR/hostapd.conf" /etc/hostapd/hostapd.conf
cp "$SCRIPT_DIR/dnsmasq.conf" /etc/dnsmasq.conf

# Point hostapd to config
echo 'DAEMON_CONF="/etc/hostapd/hostapd.conf"' > /etc/default/hostapd

# Unmask hostapd and dnsmasq (required on Pi OS - they come masked by default)
systemctl unmask hostapd
systemctl unmask dnsmasq

# Remove any mask symlinks that might persist
rm -f /etc/systemd/system/hostapd.service
rm -f /etc/systemd/system/dnsmasq.service
systemctl daemon-reload
systemctl unmask hostapd
systemctl unmask dnsmasq

# Disable hostapd and dnsmasq from starting automatically
# (first-boot service will control them)
systemctl disable hostapd
systemctl disable dnsmasq
systemctl stop hostapd 2>/dev/null || true
systemctl stop dnsmasq 2>/dev/null || true

echo "[9/10] Enabling first-boot service..."
# Enable the first-boot service to run on next boot
systemctl enable thoth-firstboot.service

# Enable thoth-web to start after first-boot completes
systemctl enable thoth-web.service

echo "[10/10] Cleaning up for imaging..."
# Remove first-boot flag if it exists (so it runs on next boot)
rm -f /etc/thoth-first-boot-done

# Remove WiFi configuration flag
rm -f "$THOTH_DIR/data/config/wifi_configured.flag"

# Clear any saved credentials
rm -f "$THOTH_DIR/data/config/auth.json"
rm -f "$THOTH_DIR/data/config/device_config.json"

# Clear logs
rm -f "$THOTH_DIR/logs/"*.log
rm -f /var/log/thoth*.log
rm -f /var/log/dnsmasq.log
journalctl --vacuum-time=1s 2>/dev/null || true

# Clear bash history
cat /dev/null > /home/pi/.bash_history 2>/dev/null || true
cat /dev/null > /root/.bash_history 2>/dev/null || true

# Clear apt cache
apt-get clean
rm -rf /var/cache/apt/archives/*

# Remove temporary files
rm -rf /tmp/*
rm -rf /var/tmp/*

# Remove SSH host keys (will be regenerated on first boot)
rm -f /etc/ssh/ssh_host_*

# Remove any WiFi credentials used during image building
cat > /etc/wpa_supplicant/wpa_supplicant.conf << 'WPAEOF'
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=US
WPAEOF

# Create flag to regenerate SSH keys on first boot
touch /etc/ssh/sshd_not_to_be_run
cat > /etc/rc.local << 'EOF'
#!/bin/bash
# Regenerate SSH host keys on first boot
if [ -f /etc/ssh/sshd_not_to_be_run ]; then
    rm -f /etc/ssh/sshd_not_to_be_run
    dpkg-reconfigure openssh-server
    systemctl restart ssh
fi
exit 0
EOF
chmod +x /etc/rc.local

# Sync filesystem
sync

echo ""
echo "=========================================="
echo "  Image Preparation Complete!"
echo "=========================================="
echo ""
echo "The system is now ready to be imaged."
echo ""
echo "To create an image:"
echo "  1. Shutdown the Pi: sudo shutdown -h now"
echo "  2. Remove the SD card"
echo "  3. Use Win32DiskImager, dd, or Raspberry Pi Imager to create an image"
echo ""
echo "When the image boots on a new Pi:"
echo "  1. Thoth will automatically create WiFi hotspot: Thoth-AP"
echo "  2. Connect to Thoth-AP (password: thoth123)"
echo "  3. A captive portal will open at http://192.168.4.1:5000"
echo "  4. Select your WiFi network and login to Brain"
echo ""
echo "To shutdown now for imaging: sudo shutdown -h now"
echo ""
