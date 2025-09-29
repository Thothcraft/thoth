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
# WiFi AP/Captive (nodogsplash)
cd /tmp
if [ ! -d "nodogsplash" ]; then
    git clone https://github.com/nodogsplash/nodogsplash.git
fi
cd nodogsplash
make
sudo make install
sudo systemctl enable nodogsplash

echo "Setting up Python virtual environment..."
# Python virtual environment and dependencies
cd /opt/thoth
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Creating data directories..."
# Create data directories
mkdir -p /opt/thoth/data/logs
chmod 755 /opt/thoth/data
chmod 755 /opt/thoth/data/logs

echo "Setting up log rotation..."
# Setup log rotation
sudo tee /etc/logrotate.d/thoth > /dev/null <<EOF
/opt/thoth/data/logs/*.log {
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
