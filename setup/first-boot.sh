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

# ---------------------------------------------------------------------------
# Install profile: "full" (Pi 4/5) or "lite" (Pi 3 and other 1 GB boards).
# Auto-detect a Pi 3 from the device-tree model unless THOTH_PROFILE is set.
# The lite profile skips the heavy/optional Python packages (torch, numba,
# scipy, pyfftw, matplotlib, eventlet) — all are lazily imported or unused, so
# the app runs fine without them; TorchScript models simply report "PyTorch
# required" instead of running.
# ---------------------------------------------------------------------------
THOTH_PROFILE="${THOTH_PROFILE:-}"
if [ -z "$THOTH_PROFILE" ]; then
    MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || true)"
    case "$MODEL" in
        *"Raspberry Pi 3"*|*"Raspberry Pi Zero"*|*"Raspberry Pi 2"*) THOTH_PROFILE="lite" ;;
        *) THOTH_PROFILE="full" ;;
    esac
fi
log "Install profile: $THOTH_PROFILE"

log "Installing Raspberry Pi system dependencies"
apt-get update -qq
# Note: libopencv-dev and docker.io are intentionally NOT installed — nothing
# in the codebase uses cv2, and Home Assistant is a manual opt-in (see below).
apt-get install -y -qq \
    python3-venv python3-pip python3-dev python3-spidev python3-gpiozero \
    ffmpeg v4l-utils sox openssh-server avahi-daemon avahi-utils
apt-get install -y -qq python3-picamera2 || true
apt-get install -y -qq python3-rpi.gpio || true
# Sense HAT support (optional — only present on some devices)
apt-get install -y -qq python3-sense-hat || true

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

log "Enabling I2C for the Sense HAT"
if command -v raspi-config >/dev/null 2>&1; then
    raspi-config nonint do_i2c 0
else
    BOOT_CONFIG="/boot/firmware/config.txt"
    [ -f "$BOOT_CONFIG" ] || BOOT_CONFIG="/boot/config.txt"
    if [ -f "$BOOT_CONFIG" ] && ! grep -q '^dtparam=i2c_arm=on' "$BOOT_CONFIG"; then
        printf '\ndtparam=i2c_arm=on\n' >> "$BOOT_CONFIG"
    fi
fi

for group in dialout video render spi gpio; do
    getent group "$group" >/dev/null 2>&1 && usermod -aG "$group" "$SERVICE_USER"
done

log "Creating Python environment"
python3 -m venv --system-site-packages "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip -q

# Core packages every profile needs.
"$VENV_DIR/bin/python" -m pip install -q \
    flask flask-socketio flask-cors requests python-dotenv netifaces \
    APScheduler psutil 'PyJWT>=2.8.0' numpy spidev \
    gpiozero pyserial pexpect

if [ "$THOTH_PROFILE" = "lite" ]; then
    # Lite (Pi 3 / 1 GB): skip the heavy scientific + inference wheels. They are
    # all lazily imported or unused, so the app runs without them.
    log "Lite profile: skipping torch, numba, scipy, pyfftw, matplotlib, eventlet"
else
    "$VENV_DIR/bin/python" -m pip install -q \
        eventlet numba scipy pyfftw matplotlib
    # torch is the heaviest dependency (~90 MB ARM wheel) and only needed for
    # TorchScript model inference. Skip it on constrained devices with
    # THOTH_NO_TORCH=1 — models then report "PyTorch required" instead of running.
    if [ "${THOTH_NO_TORCH:-0}" != "1" ]; then
        "$VENV_DIR/bin/python" -m pip install -q \
            torch --extra-index-url https://download.pytorch.org/whl/cpu
    fi
fi
"$VENV_DIR/bin/python" -c 'import flask, gpiozero, serial, spidev'

install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$THOTH_ROOT/data" "$THOTH_ROOT/config" "$THOTH_ROOT/logs"

# Seed the two optional example classifiers when the checkout is next to Desktop/models.
EXAMPLE_MODELS="$(dirname "$THOTH_ROOT")/models"
if [[ -d "$EXAMPLE_MODELS" ]]; then
  sudo -u "$SERVICE_USER" "$VENV_DIR/bin/python" "$THOTH_ROOT/setup/seed-models.py" "$EXAMPLE_MODELS" || echo "Example model import skipped; upload models from the dashboard." >&2
fi

# Home Assistant is intentionally NOT installed here — it is heavy (a ~1 GB
# docker image plus a always-on container) and overwhelms small Pis. To add it
# manually later:
#   sudo apt-get install -y docker.io
#   sudo docker run -d --name homeassistant --restart unless-stopped \
#       --privileged --network host -e TZ=America/Toronto \
#       -v "$THOTH_ROOT/config/homeassistant:/config" \
#       ghcr.io/home-assistant/home-assistant:stable

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
Environment=THOTH_PROFILE=$THOTH_PROFILE
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
Environment=THOTH_PROFILE=$THOTH_PROFILE
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

# Unique hostname per device: thoth-<serial-suffix> avoids mDNS collisions
# when several Thoth devices share a network. THOTH_HOSTNAME overrides.
if [ -z "${THOTH_HOSTNAME:-}" ]; then
    SERIAL="$(awk -F': ' '/^Serial/ {print $2}' /proc/cpuinfo 2>/dev/null | tr -d ' \t' | tail -c 7)"
    DEVICE_HOSTNAME="thoth${SERIAL:+-$SERIAL}"
else
    DEVICE_HOSTNAME="$THOTH_HOSTNAME"
fi
DEVICE_HOSTNAME="$(echo "$DEVICE_HOSTNAME" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9-')"
[ -n "$DEVICE_HOSTNAME" ] || DEVICE_HOSTNAME="thoth"
hostnamectl set-hostname "$DEVICE_HOSTNAME" || true
if grep -q '^127.0.1.1' /etc/hosts; then
    sed -i "s/^127.0.1.1.*/127.0.1.1\t$DEVICE_HOSTNAME/" /etc/hosts
else
    printf '127.0.1.1\t%s\n' "$DEVICE_HOSTNAME" >> /etc/hosts
fi

touch /etc/thoth-first-boot-done
log "Thoth installation complete: http://$DEVICE_HOSTNAME.local:5000"
if [ ! -e /dev/spidev0.0 ]; then
    log "Reboot once to activate the newly enabled SPI radar interface"
fi
