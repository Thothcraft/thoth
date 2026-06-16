#!/usr/bin/env bash
# ============================================================================
# Thoth RPi — Installation Script
#
# Installs dependencies, creates a venv, and enables the systemd service.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RPI_DIR="$(dirname "$SCRIPT_DIR")"
THOTH_ROOT="$(dirname "$RPI_DIR")"
VENV_DIR="$RPI_DIR/.venv"
RUNNING_SYSTEMD=0

if [ -d /run/systemd/system ]; then
    RUNNING_SYSTEMD=1
fi

enable_service() {
    sudo systemctl enable "$1"
}

start_service() {
    if [ "$RUNNING_SYSTEMD" -eq 1 ]; then
        sudo systemctl start "$1"
    else
        echo "  systemd is not running; $1 will start on boot"
    fi
}

echo "╔══════════════════════════════════════════╗"
echo "║    Thoth Raspberry Pi Installer          ║"
echo "╚══════════════════════════════════════════╝"

# --- System dependencies ---
echo "➤ Installing system dependencies …"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-venv \
    python3-pip \
    libopencv-dev \
    ffmpeg \
    sox \
    openssh-server \
    network-manager \
    avahi-daemon

# Hardware-specific packages vary across Raspberry Pi OS releases. Install
# them when available, but do not fail the whole image if one package moved.
sudo apt-get install -y -qq python3-picamera2 || true
sudo apt-get install -y -qq python3-rpi.gpio || true

# --- Python venv ---
echo "➤ Creating virtual environment …"
python3 -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip -q
pip install -r "$RPI_DIR/requirements.txt" -q

# --- systemd service ---
echo "➤ Installing systemd service …"
sudo cp "$SCRIPT_DIR/thoth.service" /etc/systemd/system/thoth.service
sudo cp "$SCRIPT_DIR/thoth-collector.service" /etc/systemd/system/thoth-collector.service
sudo cp "$SCRIPT_DIR/thoth-firstboot.service" /etc/systemd/system/thoth-firstboot.service

# Patch paths in the service file
sudo sed -i "s|__PYTHON__|$VENV_DIR/bin/python|g" /etc/systemd/system/thoth.service
sudo sed -i "s|__APP__|$RPI_DIR/app.py|g" /etc/systemd/system/thoth.service
sudo sed -i "s|__WORKDIR__|$RPI_DIR|g" /etc/systemd/system/thoth.service
sudo sed -i "s|__THOTH_ROOT__|$THOTH_ROOT|g" /etc/systemd/system/thoth.service

sudo sed -i "s|__PYTHON__|$VENV_DIR/bin/python|g" /etc/systemd/system/thoth-collector.service
sudo sed -i "s|__COLLECTOR__|$RPI_DIR/collector.py|g" /etc/systemd/system/thoth-collector.service
sudo sed -i "s|__WORKDIR__|$RPI_DIR|g" /etc/systemd/system/thoth-collector.service
sudo sed -i "s|__THOTH_ROOT__|$THOTH_ROOT|g" /etc/systemd/system/thoth-collector.service
sudo sed -i "s|__CAPTURE_SCRIPT__|/home/pi/Desktop/capture_dreamhat_minute.py|g" /etc/systemd/system/thoth-collector.service

sudo sed -i "s|__FIRST_BOOT__|$SCRIPT_DIR/first-boot.sh|g" /etc/systemd/system/thoth-firstboot.service

if [ "$RUNNING_SYSTEMD" -eq 1 ]; then
    sudo systemctl daemon-reload
fi
sudo systemctl unmask NetworkManager 2>/dev/null || true
enable_service NetworkManager
enable_service NetworkManager-wait-online.service
enable_service avahi-daemon
enable_service ssh
start_service ssh
enable_service thoth-firstboot.service
enable_service thoth.service
start_service thoth.service
enable_service thoth-collector.service
start_service thoth-collector.service

echo ""
echo "✅  Thoth installed and running on port 5000"
echo "    Access at: http://$(hostname -I | awk '{print $1}'):5000"
