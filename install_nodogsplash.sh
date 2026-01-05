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

echo "Copying nodogsplash from $SCRIPT_DIR/nodogsplash..."

# Debug: Check source directory
echo "DEBUG: SCRIPT_DIR = $SCRIPT_DIR"
echo "DEBUG: Source path = $SCRIPT_DIR/nodogsplash"
if [ ! -d "$SCRIPT_DIR/nodogsplash" ]; then
    echo "ERROR: Source directory $SCRIPT_DIR/nodogsplash does not exist!"
    echo "Contents of $SCRIPT_DIR:"
    ls -la "$SCRIPT_DIR"
    exit 1
fi

echo "DEBUG: Source directory exists, checking for Makefile in source..."
if [ ! -f "$SCRIPT_DIR/nodogsplash/Makefile" ]; then
    echo "ERROR: Makefile not found in source directory $SCRIPT_DIR/nodogsplash"
    echo "Contents of source directory:"
    ls -la "$SCRIPT_DIR/nodogsplash"
    exit 1
fi
echo "DEBUG: Makefile found in source directory"

# Use rsync if available, otherwise cp with manual .git exclusion
if command -v rsync &> /dev/null; then
    echo "DEBUG: Using rsync to copy files..."
    rsync -av --exclude='.git' "$SCRIPT_DIR/nodogsplash/" /tmp/nodogsplash/
else
    echo "DEBUG: Using cp to copy files..."
    cp -rv "$SCRIPT_DIR/nodogsplash" /tmp/
    rm -rf /tmp/nodogsplash/.git 2>/dev/null || true
fi

echo "DEBUG: Copy complete, checking destination..."
cd /tmp/nodogsplash
echo "Current directory: $(pwd)"
echo "Contents of /tmp/nodogsplash:"
ls -la

echo "Checking for Makefile in destination..."
if [ ! -f "Makefile" ]; then
    echo "ERROR: Makefile not found in $(pwd) after copy"
    exit 1
fi
echo "DEBUG: Makefile successfully copied to destination"

# Fix line endings on Makefile and source files (Windows CRLF -> Unix LF)
echo "Fixing line endings in nodogsplash files..."
find . -type f \( -name "Makefile" -o -name "*.c" -o -name "*.h" -o -name "*.sh" \) -exec dos2unix {} \; 2>/dev/null || \
    find . -type f \( -name "Makefile" -o -name "*.c" -o -name "*.h" -o -name "*.sh" \) -exec sed -i 's/\r$//' {} \;

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
