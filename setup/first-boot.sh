#!/usr/bin/env bash
# One-command Thoth installer for a freshly imaged Raspberry Pi OS system.
#
# Installs the packaged thoth-node daemon (python -m thoth daemon) as the
# single authoritative service. The legacy src/app.py + src/collector.py
# units are migrated away on rerun — src/ is a deprecated path kept for
# reference only.

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
# bluez + rfkill: BLE observer/central/peripheral roles and BLE-first
# provisioning. network-manager + policykit-1: Wi-Fi join via nmcli from
# the unprivileged daemon (polkit rule below grants netdev).
apt-get install -y -qq \
    python3-venv python3-pip python3-dev python3-spidev python3-gpiozero \
    ffmpeg v4l-utils sox openssh-server avahi-daemon avahi-utils \
    git bluez rfkill network-manager polkitd
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

# bluetooth → org.bluez D-Bus access (observer/central/peripheral);
# netdev → NetworkManager control for provisioning (polkit rule below).
for group in dialout video render spi gpio bluetooth netdev; do
    getent group "$group" >/dev/null 2>&1 && usermod -aG "$group" "$SERVICE_USER"
done

# Let netdev manage NetworkManager without an interactive session — the
# daemon joins Wi-Fi during BLE/AP provisioning as the service user.
if [ -d /etc/polkit-1/rules.d ]; then
    cat > /etc/polkit-1/rules.d/50-thoth-net.rules <<'EOF'
polkit.addRule(function(action, subject) {
    if (subject.isInGroup("netdev") &&
        action.id.indexOf("org.freedesktop.NetworkManager.") === 0) {
        return polkit.Result.YES;
    }
});
EOF
fi

log "Creating Python environment"
python3 -m venv --system-site-packages "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip -q

# whispy: shared contracts/drivers package. Resolution order:
#   1. sibling checkout at ../whispy/packages/* (developer machines)
#   2. git+https (subdirectory installs for plugin packages)
#   3. PyPI fallback for the core package alone.
WORKSPACE_ROOT="$(dirname "$THOTH_ROOT")"
WHISPY_LOCAL="$WORKSPACE_ROOT/whispy/packages"
if [ -d "$WHISPY_LOCAL/whispy" ]; then
    log "Installing whispy from sibling checkout: $WHISPY_LOCAL"
    "$VENV_DIR/bin/python" -m pip install -q "$WHISPY_LOCAL/whispy"
    for pkg in "$WHISPY_LOCAL"/whispy-*; do
        [ -d "$pkg" ] || continue
        "$VENV_DIR/bin/python" -m pip install -q "$pkg" || true
    done
else
    "$VENV_DIR/bin/python" -m pip install -q \
        "whispy @ git+https://github.com/Thothcraft/whispy#subdirectory=packages/whispy" \
        || "$VENV_DIR/bin/python" -m pip install -q whispy
    for pkg in whispy-sensor-dreamhat whispy-sensor-mmwhat whispy-sensor-csi \
               whispy-sensor-opencv-camera whispy-sensor-microphone \
               whispy-sensor-sensehat whispy-model-occ \
               whispy-model-opencv-person whispy-actuator-homeassistant \
               whispy-actuator-sensehat whispy-actuator-speaker \
               whispy-model-whisper-stt whispy-model-face; do
        "$VENV_DIR/bin/python" -m pip install -q \
            "$pkg @ git+https://github.com/Thothcraft/whispy#subdirectory=packages/$pkg" \
            || true
    done
fi

# The packaged node application — install from this checkout (non-editable
# keeps parity with the git+https installer path).
"$VENV_DIR/bin/python" -m pip install -q "$THOTH_ROOT"

# Hardware/runtime deps the daemon's optional drivers need at runtime.
"$VENV_DIR/bin/python" -m pip install -q \
    requests 'PyJWT>=2.8.0' numpy spidev gpiozero pyserial pexpect \
    bleak dbus-next

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
"$VENV_DIR/bin/python" -c 'import thoth, whispy, bleak, gpiozero, serial, spidev'

install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$THOTH_ROOT/data" "$THOTH_ROOT/config" "$THOTH_ROOT/logs"
SERVICE_HOME="$(getent passwd "$SERVICE_USER" | cut -d: -f6)"
install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0700 "$SERVICE_HOME/.thoth"

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

# ---------------------------------------------------------------------------
# Legacy migration: the pre-package runtime ran src/app.py (dashboard) +
# src/collector.py (minute collector) as two units. The packaged daemon
# supersedes both — remove the collector unit entirely and repoint
# thoth.service. Config/data under ~/.thoth and the checkout's data/, logs/
# are preserved untouched.
# ---------------------------------------------------------------------------
log "Migrating any legacy src/ units"
if [ -f /etc/systemd/system/thoth-collector.service ]; then
    systemctl disable --now thoth-collector.service 2>/dev/null || true
    rm -f /etc/systemd/system/thoth-collector.service
    log "Removed thoth-collector.service (superseded by thoth daemon minutes)"
fi

log "Installing systemd services"
cat > /etc/systemd/system/thoth.service <<EOF
[Unit]
Description=Thoth Node Daemon
After=network-online.target bluetooth.target
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$THOTH_ROOT
Environment=THOTH_ROOT=$THOTH_ROOT
Environment=THOTH_PROFILE=$THOTH_PROFILE
# CAP_NET_BIND_SERVICE lets the dashboard bind :80 without running root;
# ignored harmlessly when the port is already taken (daemon falls back).
AmbientCapabilities=CAP_NET_BIND_SERVICE
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
ExecStart=$VENV_DIR/bin/python -m thoth daemon
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
Alias=thoth-web.service
EOF

systemctl disable --now thoth-firstboot.service 2>/dev/null || true
rm -f /etc/systemd/system/thoth-firstboot.service
systemctl daemon-reload
systemctl enable avahi-daemon ssh thoth.service bluetooth
systemctl restart avahi-daemon ssh bluetooth thoth.service

# Friendly per-device hostname: thoth-<name>.local where <name> is a random
# month, person, or city name. THOTH_HOSTNAME overrides for a fixed name.
# Shared logic lives in setup/device-hostname.sh so existing devices can be
# renamed with the same scheme.
bash "$SCRIPT_DIR/device-hostname.sh"
DEVICE_HOSTNAME="$(hostname)"

touch /etc/thoth-first-boot-done
log "Thoth installation complete: http://$DEVICE_HOSTNAME.local"
log "  daemon: systemctl status thoth.service | journalctl -u thoth.service -f"
log "  pair:   thoth pair   (or BLE commissioning when unprovisioned)"
if [ ! -e /dev/spidev0.0 ]; then
    log "Reboot once to activate the newly enabled SPI radar interface"
fi
if ! id -nG "$SERVICE_USER" | tr ' ' '\n' | grep -qx bluetooth; then
    log "NOTE: $SERVICE_USER not in bluetooth group — BLE features disabled"
fi
