#!/bin/bash
# Thoth Dependencies Installation Script

set -e

echo "=== Thoth Installation Script ==="
echo "Updating system packages..."

# Update system
sudo apt update && sudo apt upgrade -y

echo "Installing core dependencies..."
# Core dependencies
sudo apt install -y python3-pip python3-venv git hostapd dnsmasq iptables libmicrohttpd-dev build-essential i2c-tools

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
mkdir -p nodogsplash
cd "$THOTH_DIR/nodogsplash"
tar --exclude='.git' -cf - . | (cd /tmp/nodogsplash && tar -xf -)
cd /tmp/nodogsplash

echo "Current directory: $(pwd)"
echo "Checking for Makefile..."
if [ ! -f "Makefile" ]; then
    echo "ERROR: Makefile not found in $(pwd)"
    ls -la
    exit 1
fi

# Fix line endings on Makefile and source files (Windows CRLF -> Unix LF)
echo "Fixing line endings in nodogsplash files..."
find . -type f \( -name "Makefile" -o -name "*.c" -o -name "*.h" -o -name "*.sh" \) -exec dos2unix {} \; 2>/dev/null || \
    find . -type f \( -name "Makefile" -o -name "*.c" -o -name "*.h" -o -name "*.sh" \) -exec sed -i 's/\r$//' {} \;

make
sudo make install
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
