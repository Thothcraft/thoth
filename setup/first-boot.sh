#!/usr/bin/env bash
# Install and configure Thoth from a fresh Raspberry Pi OS clone.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THOTH_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/var/log/thoth-first-boot.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

if [ "${EUID}" -ne 0 ]; then
    echo "Run this script with sudo: sudo bash first-boot.sh" >&2
    exit 1
fi

log "=========================================="
log "  Thoth First Boot Initialization"
log "=========================================="

log "Installing Thoth services and hardware dependencies..."
bash "$THOTH_DIR/thoth_rpi/setup/install.sh"

log "Setting hostname to thoth..."
hostnamectl set-hostname thoth || log "Warning: could not set hostname"
if grep -q '^127.0.1.1' /etc/hosts 2>/dev/null; then
    sed -i 's/^127.0.1.1.*/127.0.1.1\tthoth/' /etc/hosts
else
    printf '127.0.1.1\tthoth\n' >> /etc/hosts
fi

log "Starting Thoth web application..."
systemctl enable thoth.service
systemctl restart thoth.service

log "Starting Thoth continuous collector..."
systemctl enable thoth-collector.service
systemctl restart thoth-collector.service

touch /etc/thoth-first-boot-done

log "First boot initialization complete"
log "Access the web app at http://thoth.local:5000"

if [ ! -e /dev/spidev0.0 ]; then
    log "SPI was enabled for the DreamHat radar; reboot once before radar capture will work."
fi
