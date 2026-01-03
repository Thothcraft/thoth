#!/bin/bash
# Nodogsplash Installation Script for Thoth Device
# This script installs and configures Nodogsplash for automatic captive portal detection

set -e

echo "=========================================="
echo "Thoth Device - Nodogsplash Installation"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Update package list
echo "[1/6] Updating package list..."
apt update

# Install dependencies
echo "[2/6] Installing dependencies..."
apt install -y git libmicrohttpd-dev build-essential

# Copy local Nodogsplash repository
echo "[3/6] Copying local Nodogsplash repository..."
cd /tmp
if [ -d "nodogsplash" ]; then
    rm -rf nodogsplash
fi
cp -r "$SCRIPT_DIR/nodogsplash" /tmp/
cd nodogsplash

# Build and install
echo "[4/6] Building Nodogsplash..."
make

echo "[5/6] Installing Nodogsplash..."
make install

# Create configuration directory if it doesn't exist
mkdir -p /etc/nodogsplash/htdocs

# Copy configuration file
echo "[6/6] Setting up configuration..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/nodogsplash.conf" ]; then
    cp "$SCRIPT_DIR/nodogsplash.conf" /etc/nodogsplash/nodogsplash.conf
    echo "Configuration file copied from $SCRIPT_DIR/nodogsplash.conf"
else
    echo "Warning: nodogsplash.conf not found in $SCRIPT_DIR"
    echo "You'll need to configure /etc/nodogsplash/nodogsplash.conf manually"
fi

if [ -f "$SCRIPT_DIR/splash.html" ]; then
    cp "$SCRIPT_DIR/splash.html" /etc/nodogsplash/htdocs/splash.html
    echo "Splash page copied from $SCRIPT_DIR/splash.html"
else
    echo "Warning: splash.html not found in $SCRIPT_DIR"
    echo "You'll need to create /etc/nodogsplash/htdocs/splash.html manually"
fi

# Enable and start service
echo ""
echo "Enabling Nodogsplash service..."
systemctl enable nodogsplash

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review the configuration at /etc/nodogsplash/nodogsplash.conf"
echo "2. Customize the splash page at /etc/nodogsplash/htdocs/splash.html"
echo "3. Start the service with: sudo systemctl start nodogsplash"
echo "4. Or reboot the system: sudo reboot"
echo ""
echo "Connect to 'Thoth' WiFi network to test the captive portal."
echo "To check status: sudo systemctl status nodogsplash"
echo "To view logs: sudo journalctl -u nodogsplash -f"
echo ""
