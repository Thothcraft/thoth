#!/bin/bash
# Thoth Dependencies Installation Script

set -e

echo "=== Thoth Installation Script ==="
echo "Updating system packages..."

# Update system
sudo apt update && sudo apt upgrade -y

echo "Installing core dependencies..."
# Core dependencies
sudo apt install -y python3-pip python3-venv git hostapd dnsmasq iptables libmicrohttpd-dev libjson-c-dev build-essential i2c-tools

echo "Installing Sense HAT support..."
# Sense HAT
sudo apt install -y python3-sense-hat sense-hat-emulator

echo "Installing PiSugar support..."
# PiSugar (from GitHub)
cd /tmp
if [ ! -d "PiSugar" ]; then
    git clone https://github.com/PiSugar/PiSugar.git
fi
cd PiSugar
sudo ./install.sh
sudo systemctl enable pisugar-power-manager

echo "Installing WiFi AP/Captive portal support..."
# WiFi AP/Captive (nodogsplash) - use local copy
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THOTH_DIR="$(dirname "$SCRIPT_DIR")"
cd /tmp
if [ -d "nodogsplash" ]; then
    rm -rf nodogsplash
fi

echo "Copying nodogsplash from $THOTH_DIR/nodogsplash..."

# Debug: Check source directory
echo "DEBUG: THOTH_DIR = $THOTH_DIR"
echo "DEBUG: Source path = $THOTH_DIR/nodogsplash"
if [ ! -d "$THOTH_DIR/nodogsplash" ]; then
    echo "ERROR: Source directory $THOTH_DIR/nodogsplash does not exist!"
    echo "Contents of $THOTH_DIR:"
    ls -la "$THOTH_DIR"
    exit 1
fi

echo "DEBUG: Source directory exists, checking for Makefile in source..."
if [ ! -f "$THOTH_DIR/nodogsplash/Makefile" ]; then
    echo "ERROR: Makefile not found in source directory $THOTH_DIR/nodogsplash"
    echo "Contents of source directory:"
    ls -la "$THOTH_DIR/nodogsplash"
    exit 1
fi
echo "DEBUG: Makefile found in source directory"

# Use rsync if available, otherwise cp with manual .git exclusion
if command -v rsync &> /dev/null; then
    echo "DEBUG: Using rsync to copy files..."
    rsync -av --exclude='.git' "$THOTH_DIR/nodogsplash/" /tmp/nodogsplash/
else
    echo "DEBUG: Using cp to copy files..."
    cp -rv "$THOTH_DIR/nodogsplash" /tmp/
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

make
sudo make install

# Install systemd service file
echo "Installing nodogsplash systemd service..."
if [ -f "debian/nodogsplash.service" ]; then
    sudo cp debian/nodogsplash.service /etc/systemd/system/nodogsplash.service
    sudo dos2unix /etc/systemd/system/nodogsplash.service 2>/dev/null || sudo sed -i 's/\r$//' /etc/systemd/system/nodogsplash.service
    sudo systemctl daemon-reload
    echo "Nodogsplash service file installed"
else
    echo "Warning: nodogsplash.service not found in debian directory"
fi

sudo systemctl enable nodogsplash

echo "Setting up Python virtual environment..."
# Python virtual environment and dependencies
cd "$THOTH_DIR"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Creating data directories..."
# Create data directories
mkdir -p "$THOTH_DIR/data/logs"
chmod 755 "$THOTH_DIR/data"
chmod 755 "$THOTH_DIR/data/logs"

echo "Setting up log rotation..."
# Setup log rotation
sudo tee /etc/logrotate.d/thoth > /dev/null <<EOF
$THOTH_DIR/data/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 pi pi
}
EOF

echo "Installation complete!"
echo "Next steps:"
echo "1. Run './setup/enable_services.sh' to enable systemd services"
echo "2. Run './setup/configure_pisugar.sh' to configure PiSugar"
echo "3. Copy .env.example to .env and configure your settings"
echo "4. Reboot the system"
