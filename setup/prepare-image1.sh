#!/bin/bash
# Thoth Raspberry Pi Image Preparation Script - Part 1
# Non-Network-Disrupting Setup
# This part installs system dependencies, Python environment, and Thoth services
# Run this while connected to WiFi for updates and downloads

set -e

echo "=========================================="
echo "  Thoth Image Preparation - Part 1"
echo "  (Non-Network-Disrupting Setup)"
echo "=========================================="
echo ""
echo "This script prepares system dependencies and Thoth services."
echo "You will remain connected to WiFi during this process."
echo "Run prepare-image2.sh afterwards to set up the hotspot."
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./prepare-image1.sh)"
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THOTH_DIR="$(dirname "$SCRIPT_DIR")"
THOTH_USER="pi"

echo "[1/7] Updating system packages..."
# Fix any interrupted dpkg installations
echo "Checking for interrupted package installations..."

# Remove corrupted packages that can't be reinstalled
echo "Removing any corrupted packages..."
dpkg --remove --force-remove-reinstreq rpi-eeprom 2>/dev/null || true

dpkg --configure -a

# Clean up any stale apt locks
rm -f /var/lib/apt/lists/lock
rm -f /var/cache/apt/archives/lock
rm -f /var/lib/dpkg/lock*

apt-get update
apt-get upgrade -y

# Reinstall rpi-eeprom if it was removed
echo "Ensuring rpi-eeprom is installed..."
apt-get install -y rpi-eeprom 2>/dev/null || echo "Note: rpi-eeprom installation skipped (not critical for image preparation)"

echo "[2/7] Installing system dependencies..."
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
    dos2unix \
    libmicrohttpd-dev \
    libjson-c-dev \
    build-essential

echo "[3/7] Installing Thoth to /home/pi/thoth..."
# Copy thoth to standard location if not already there
if [ "$THOTH_DIR" != "/home/pi/thoth" ]; then
    mkdir -p /home/pi/thoth
    cp -r "$THOTH_DIR"/* /home/pi/thoth/
    THOTH_DIR="/home/pi/thoth"
    SCRIPT_DIR="$THOTH_DIR/setup"
fi

echo "[4/7] Creating Python virtual environment..."
cd "$THOTH_DIR"
python3 -m venv venv
source venv/bin/activate

# Set matplotlib backend for headless operation (no display)
export MPLBACKEND=Agg

echo "[5/7] Installing Python dependencies..."
pip install --upgrade pip
# Use Pi-specific requirements (excludes heavy ML libraries like PyTorch)
if [ -f "$THOTH_DIR/requirements-pi.txt" ]; then
    echo "Installing lightweight Pi requirements..."
    pip install -r requirements-pi.txt
else
    echo "Warning: requirements-pi.txt not found, using full requirements.txt (may take hours)..."
    pip install -r requirements.txt
fi

echo "[6/7] Setting up Thoth directories..."
mkdir -p "$THOTH_DIR/data/config"
mkdir -p "$THOTH_DIR/logs"
chown -R "$THOTH_USER:$THOTH_USER" "$THOTH_DIR"

echo "[7/7] Installing systemd services..."
# Copy service files
cp "$SCRIPT_DIR/thoth-web.service" /etc/systemd/system/
cp "$SCRIPT_DIR/thoth-hotspot.service" /etc/systemd/system/
cp "$SCRIPT_DIR/thoth-firstboot.service" /etc/systemd/system/

# Ensure paths are correct
sed -i "s|/home/pi/thoth|$THOTH_DIR|g" /etc/systemd/system/thoth-web.service
sed -i "s|/home/pi/thoth|$THOTH_DIR|g" /etc/systemd/system/thoth-hotspot.service
sed -i "s|/home/pi/thoth|$THOTH_DIR|g" /etc/systemd/system/thoth-firstboot.service

# Verify services are enabled
systemctl enable thoth-firstboot.service
systemctl enable thoth-web.service

# Reload systemd
systemctl daemon-reload

# Fix Windows line endings (CRLF -> LF) on all scripts and configs
echo "Fixing line endings on scripts and configs..."
find "$SCRIPT_DIR" -type f \( -name "*.sh" -o -name "*.conf" -o -name "*.service" \) -exec dos2unix {} \; 2>/dev/null || \
    find "$SCRIPT_DIR" -type f \( -name "*.sh" -o -name "*.conf" -o -name "*.service" \) -exec sed -i 's/\r$//' {} \;

# Ensure all scripts are executable and have correct permissions
find "$SCRIPT_DIR" -type f \( -name "*.sh" \) -exec chmod +x {} \;
chown -R "$THOTH_USER:$THOTH_USER" "$SCRIPT_DIR"

# Create completion flag
touch /etc/thoth-image-part1-complete

echo ""
echo "=========================================="
echo "  Part 1 Complete!"
echo "=========================================="
echo ""
echo "System dependencies and Thoth services are installed."
echo "You are still connected to WiFi."
echo ""
echo "Next steps:"
echo "  1. Run: sudo ./prepare-image2.sh"
echo "     (This will set up the hotspot and disconnect from WiFi)"
echo ""
echo "Or run both parts together:"
echo "  sudo ./prepare-image.sh"
echo ""
