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

echo "╔══════════════════════════════════════════╗"
echo "║    Thoth Raspberry Pi Installer          ║"
echo "╚══════════════════════════════════════════╝"

# --- System dependencies ---
echo "➤ Installing system dependencies …"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip libopencv-dev ffmpeg sox

# --- Python venv ---
echo "➤ Creating virtual environment …"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip -q
pip install -r "$THOTH_ROOT/thoth_core/requirements.txt" -q
pip install -r "$RPI_DIR/requirements.txt" -q

# --- systemd service ---
echo "➤ Installing systemd service …"
sudo cp "$SCRIPT_DIR/thoth.service" /etc/systemd/system/thoth.service
sudo cp "$SCRIPT_DIR/thoth-collector.service" /etc/systemd/system/thoth-collector.service

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

sudo systemctl daemon-reload
sudo systemctl enable thoth.service
sudo systemctl start thoth.service
sudo systemctl enable thoth-collector.service
sudo systemctl start thoth-collector.service

echo ""
echo "✅  Thoth installed and running on port 8000"
echo "    Access at: http://$(hostname -I | awk '{print $1}'):8000"
