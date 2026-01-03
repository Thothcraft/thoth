#!/bin/bash
# Thoth Device Installation Script for Raspberry Pi
# This script installs all dependencies and configures the system

set -e

echo "=========================================="
echo "  Thoth Device Installation Script"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./install.sh)"
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THOTH_DIR="$(dirname "$SCRIPT_DIR")"
THOTH_USER="${SUDO_USER:-pi}"

echo "[1/8] Updating system packages..."
apt-get update
apt-get upgrade -y

echo "[2/8] Installing system dependencies..."
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    hostapd \
    dnsmasq \
    dhcpcd5 \
    wireless-tools \
    net-tools \
    git

echo "[3/8] Creating Python virtual environment..."
cd "$THOTH_DIR"
python3 -m venv venv
source venv/bin/activate

echo "[4/8] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[5/8] Setting up Thoth directories..."
mkdir -p "$THOTH_DIR/data/config"
mkdir -p "$THOTH_DIR/logs"
chown -R "$THOTH_USER:$THOTH_USER" "$THOTH_DIR"

echo "[6/8] Installing systemd services..."
# Copy service files
cp "$SCRIPT_DIR/thoth-web.service" /etc/systemd/system/
cp "$SCRIPT_DIR/thoth-hotspot.service" /etc/systemd/system/

# Update paths in service files
sed -i "s|/home/pi/thoth|$THOTH_DIR|g" /etc/systemd/system/thoth-web.service
sed -i "s|/home/pi/thoth|$THOTH_DIR|g" /etc/systemd/system/thoth-hotspot.service
sed -i "s|User=pi|User=$THOTH_USER|g" /etc/systemd/system/thoth-web.service

# Reload systemd
systemctl daemon-reload

echo "[7/8] Configuring hostapd and dnsmasq..."
# Backup original configs
[ -f /etc/hostapd/hostapd.conf ] && cp /etc/hostapd/hostapd.conf /etc/hostapd/hostapd.conf.backup
[ -f /etc/dnsmasq.conf ] && cp /etc/dnsmasq.conf /etc/dnsmasq.conf.backup

# Copy our configs
cp "$SCRIPT_DIR/hostapd.conf" /etc/hostapd/hostapd.conf
cp "$SCRIPT_DIR/dnsmasq.conf" /etc/dnsmasq.conf

# Point hostapd to config
echo 'DAEMON_CONF="/etc/hostapd/hostapd.conf"' > /etc/default/hostapd

# Disable hostapd and dnsmasq from starting automatically
# (we control them via our service)
systemctl disable hostapd
systemctl disable dnsmasq
systemctl stop hostapd 2>/dev/null || true
systemctl stop dnsmasq 2>/dev/null || true

echo "[8/8] Enabling Thoth services..."
systemctl enable thoth-web.service

echo ""
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""
echo "Thoth is installed at: $THOTH_DIR"
echo ""
echo "On first boot:"
echo "  1. Thoth will create a WiFi hotspot: Thoth"
echo "  2. Connect to Thoth (password: thoth123)"
echo "  3. Open http://192.168.4.1 in your browser"
echo "  4. Select your WiFi network and login"
echo ""
echo "To start Thoth now: sudo systemctl start thoth-web"
echo "To view logs: journalctl -u thoth-web -f"
echo ""
