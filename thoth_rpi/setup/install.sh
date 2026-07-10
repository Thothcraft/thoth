#!/usr/bin/env bash
# ============================================================================
# Thoth RPi — Installation Script
#
# Installs dependencies, creates a venv, and enables the systemd service.
# ============================================================================

set -euo pipefail

if [ "${EUID}" -ne 0 ]; then
    exec sudo -E bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RPI_DIR="$(dirname "$SCRIPT_DIR")"
THOTH_ROOT="$(dirname "$RPI_DIR")"
VENV_DIR="$RPI_DIR/.venv"
RUNNING_SYSTEMD=0
SERVICE_USER="${SUDO_USER:-}"

if [ -z "$SERVICE_USER" ] || [ "$SERVICE_USER" = "root" ]; then
    SERVICE_USER="$(stat -c '%U' "$THOTH_ROOT")"
fi
if [ -z "$SERVICE_USER" ] || [ "$SERVICE_USER" = "UNKNOWN" ] || ! id "$SERVICE_USER" >/dev/null 2>&1; then
    echo "Unable to determine the non-root account that should run Thoth." >&2
    exit 1
fi
SERVICE_GROUP="$(id -gn "$SERVICE_USER")"

if [ -d /run/systemd/system ]; then
    RUNNING_SYSTEMD=1
fi

enable_service() {
    systemctl enable "$1"
}

start_service() {
    if [ "$RUNNING_SYSTEMD" -eq 1 ]; then
        systemctl restart "$1"
    else
        echo "  systemd is not running; $1 will start on boot"
    fi
}

echo "╔══════════════════════════════════════════╗"
echo "║    Thoth Raspberry Pi Installer          ║"
echo "╚══════════════════════════════════════════╝"

# --- System dependencies ---
echo "➤ Installing system dependencies …"
apt-get update -qq
apt-get install -y -qq \
    python3-venv \
    python3-pip \
    python3-dev \
    python3-spidev \
    python3-gpiozero \
    libopencv-dev \
    ffmpeg \
    v4l-utils \
    sox \
    openssh-server \
    avahi-daemon

# Hardware-specific packages vary across Raspberry Pi OS releases. Install
# them when available, but do not fail the whole image if one package moved.
apt-get install -y -qq python3-picamera2 || true
apt-get install -y -qq python3-rpi.gpio || true

# DreamHat uses SPI0. This persists across boots; the first enable requires one
# reboot before /dev/spidev0.0 is created by the kernel.
if command -v raspi-config >/dev/null 2>&1; then
    raspi-config nonint do_spi 0
else
    BOOT_CONFIG="/boot/firmware/config.txt"
    [ -f "$BOOT_CONFIG" ] || BOOT_CONFIG="/boot/config.txt"
    if [ -f "$BOOT_CONFIG" ] && ! grep -q '^dtparam=spi=on' "$BOOT_CONFIG"; then
        printf '\ndtparam=spi=on\n' >> "$BOOT_CONFIG"
    fi
fi

for group in dialout video render spi gpio; do
    if getent group "$group" >/dev/null 2>&1; then
        usermod -aG "$group" "$SERVICE_USER"
    fi
done

# --- Python venv ---
echo "➤ Creating virtual environment …"
python3 -m venv --system-site-packages "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip -q
"$VENV_DIR/bin/python" -m pip install -r "$RPI_DIR/requirements.txt" -q
"$VENV_DIR/bin/python" -c 'import gpiozero, numba, pyfftw, scipy, serial, spidev'

# --- systemd service ---
echo "➤ Installing systemd service …"
cp "$SCRIPT_DIR/thoth.service" /etc/systemd/system/thoth.service
cp "$SCRIPT_DIR/thoth-collector.service" /etc/systemd/system/thoth-collector.service
cp "$SCRIPT_DIR/thoth-firstboot.service" /etc/systemd/system/thoth-firstboot.service

# Patch paths in the service file
sed -i "s|__USER__|$SERVICE_USER|g; s|__GROUP__|$SERVICE_GROUP|g" /etc/systemd/system/thoth.service
sed -i "s|__PYTHON__|$VENV_DIR/bin/python|g" /etc/systemd/system/thoth.service
sed -i "s|__APP__|$RPI_DIR/app.py|g" /etc/systemd/system/thoth.service
sed -i "s|__WORKDIR__|$RPI_DIR|g" /etc/systemd/system/thoth.service
sed -i "s|__THOTH_ROOT__|$THOTH_ROOT|g" /etc/systemd/system/thoth.service

sed -i "s|__USER__|$SERVICE_USER|g; s|__GROUP__|$SERVICE_GROUP|g" /etc/systemd/system/thoth-collector.service
sed -i "s|__PYTHON__|$VENV_DIR/bin/python|g" /etc/systemd/system/thoth-collector.service
sed -i "s|__COLLECTOR__|$RPI_DIR/collector.py|g" /etc/systemd/system/thoth-collector.service
sed -i "s|__WORKDIR__|$RPI_DIR|g" /etc/systemd/system/thoth-collector.service
sed -i "s|__THOTH_ROOT__|$THOTH_ROOT|g" /etc/systemd/system/thoth-collector.service
sed -i "s|__CAPTURE_SCRIPT__|$THOTH_ROOT/capture_dreamhat_minute.py|g" /etc/systemd/system/thoth-collector.service

sed -i "s|__FIRST_BOOT__|$SCRIPT_DIR/first-boot.sh|g" /etc/systemd/system/thoth-firstboot.service

install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$THOTH_ROOT/data"

if [ "$RUNNING_SYSTEMD" -eq 1 ]; then
    systemctl daemon-reload
fi
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
if [ ! -e /dev/spidev0.0 ]; then
    echo "⚠️   SPI was enabled. Reboot once to activate the DreamHat radar."
fi
