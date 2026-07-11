#!/usr/bin/env bash
# One-command Thoth installer for a freshly imaged Raspberry Pi OS system.

set -euo pipefail

if [ "${EUID}" -ne 0 ]; then
    echo "Run with sudo: sudo bash setup/first-boot.sh" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THOTH_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$THOTH_ROOT/.venv"
SERVICE_USER="${SUDO_USER:-}"

if [ -z "$SERVICE_USER" ] || [ "$SERVICE_USER" = "root" ]; then
    SERVICE_USER="$(stat -c '%U' "$THOTH_ROOT")"
fi
if ! id "$SERVICE_USER" >/dev/null 2>&1; then
    echo "Unable to determine the non-root user that owns $THOTH_ROOT" >&2
    exit 1
fi
SERVICE_GROUP="$(id -gn "$SERVICE_USER")"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Installing Raspberry Pi system dependencies"
apt-get update -qq
apt-get install -y -qq \
    python3-venv python3-pip python3-dev python3-spidev python3-gpiozero \
    libopencv-dev ffmpeg v4l-utils sox openssh-server avahi-daemon
apt-get install -y -qq python3-picamera2 || true
apt-get install -y -qq python3-rpi.gpio || true

log "Enabling SPI for the DreamHat radar"
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
    getent group "$group" >/dev/null 2>&1 && usermod -aG "$group" "$SERVICE_USER"
done

log "Creating Python environment"
python3 -m venv --system-site-packages "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip -q
"$VENV_DIR/bin/python" -m pip install -q \
    flask flask-socketio flask-cors requests python-dotenv eventlet netifaces \
    APScheduler psutil 'PyJWT>=2.8.0' numpy numba scipy pyfftw spidev \
    gpiozero matplotlib pyserial pexpect
"$VENV_DIR/bin/python" -c 'import flask, gpiozero, numba, pyfftw, scipy, serial, spidev'

install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$THOTH_ROOT/data" "$THOTH_ROOT/config" "$THOTH_ROOT/logs"

log "Installing systemd services"
cat > /etc/systemd/system/thoth.service <<EOF
[Unit]
Description=Thoth Raspberry Pi Dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$THOTH_ROOT
Environment=THOTH_ROOT=$THOTH_ROOT
ExecStart=$VENV_DIR/bin/python $THOTH_ROOT/src/app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
Alias=thoth-web.service
EOF

cat > /etc/systemd/system/thoth-collector.service <<EOF
[Unit]
Description=Thoth Continuous Minute Collector
After=thoth.service network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$THOTH_ROOT
Environment=THOTH_ROOT=$THOTH_ROOT
Environment=THOTH_CAPTURE_SCRIPT=$THOTH_ROOT/src/backend/minute_collector.py
ExecStart=$VENV_DIR/bin/python $THOTH_ROOT/src/collector.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl disable --now thoth-firstboot.service 2>/dev/null || true
rm -f /etc/systemd/system/thoth-firstboot.service
systemctl daemon-reload
systemctl enable avahi-daemon ssh thoth.service thoth-collector.service
systemctl restart avahi-daemon ssh thoth.service thoth-collector.service

hostnamectl set-hostname thoth || true
if grep -q '^127.0.1.1' /etc/hosts; then
    sed -i 's/^127.0.1.1.*/127.0.1.1\tthoth/' /etc/hosts
else
    printf '127.0.1.1\tthoth\n' >> /etc/hosts
fi

touch /etc/thoth-first-boot-done
log "Thoth installation complete: http://thoth.local:5000"
if [ ! -e /dev/spidev0.0 ]; then
    log "Reboot once to activate the newly enabled SPI radar interface"
fi
