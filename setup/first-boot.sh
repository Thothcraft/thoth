#!/bin/bash
# Thoth First Boot Setup Script
# Minimal boot path:
# - set hostname
# - remove captive portal leftovers
# - start web app and collector

set +e

THOTH_DIR="/home/pi/Desktop/thoth"
FIRST_BOOT_FLAG="/etc/thoth-first-boot-done"
LOG_FILE="/var/log/thoth-first-boot.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

if [ -f "$FIRST_BOOT_FLAG" ]; then
    log "First boot already completed, exiting"
    exit 0
fi

log "=========================================="
log "  Thoth First Boot Initialization"
log "=========================================="

log "Setting hostname to thoth..."
hostnamectl set-hostname thoth || log "Warning: could not set hostname"
if grep -q '^127.0.1.1' /etc/hosts 2>/dev/null; then
    sed -i 's/^127.0.1.1.*/127.0.1.1\tthoth/' /etc/hosts
else
    printf '127.0.1.1\tthoth\n' >> /etc/hosts
fi

log "Removing legacy hotspot and captive portal state..."
sudo systemctl unmask NetworkManager 2>/dev/null || true
sudo systemctl enable NetworkManager 2>/dev/null || true
sudo systemctl enable NetworkManager-wait-online.service 2>/dev/null || true
sudo systemctl restart NetworkManager 2>/dev/null || true
sudo systemctl disable --now nodogsplash 2>/dev/null || true
sudo systemctl disable --now thoth-hotspot 2>/dev/null || true
sudo systemctl disable --now hostapd 2>/dev/null || true
sudo systemctl disable --now dnsmasq 2>/dev/null || true
sudo systemctl mask hostapd 2>/dev/null || true
sudo systemctl mask dnsmasq 2>/dev/null || true
sudo systemctl enable avahi-daemon 2>/dev/null || true
sudo systemctl restart avahi-daemon 2>/dev/null || true
rm -f /etc/systemd/system/hostapd.service 2>/dev/null || true
rm -f /etc/systemd/system/dnsmasq.service 2>/dev/null || true
rm -f /etc/nodogsplash/nodogsplash.conf 2>/dev/null || true
rm -rf /etc/nodogsplash 2>/dev/null || true
rm -f /var/lib/nodogsplash/* 2>/dev/null || true
rm -f /var/run/thoth-firstboot 2>/dev/null || true

log "Starting Thoth web application..."
sudo systemctl enable thoth-web.service 2>/dev/null || true
sudo systemctl restart thoth-web.service || log "Warning: thoth-web restart failed"

log "Starting Thoth continuous collector..."
sudo systemctl enable thoth-collector.service 2>/dev/null || true
sudo systemctl restart thoth-collector.service || log "Warning: thoth-collector restart failed"

touch "$FIRST_BOOT_FLAG"

log "First boot initialization complete"
log "Access the web app at http://thoth.local:5000"
