#!/usr/bin/env bash
# ============================================================================
# Thoth RPi — First Boot Script
#
# Runs once after the image is flashed to a new SD card.
# Regenerates SSH keys, restores LAN WiFi services, advertises thoth.local,
# and ensures the web app and collector are running.
# ============================================================================

set -euo pipefail

FIRST_BOOT_FLAG="/etc/thoth-first-boot-done"
HOSTNAME="thoth"

if [ -f "$FIRST_BOOT_FLAG" ]; then
    exit 0
fi

# Ensure the image never boots into the old hotspot/captive-portal path.
sudo systemctl disable --now nodogsplash 2>/dev/null || true
sudo systemctl disable --now thoth-hotspot 2>/dev/null || true
sudo systemctl disable --now hostapd 2>/dev/null || true
sudo systemctl disable --now dnsmasq 2>/dev/null || true
sudo systemctl mask nodogsplash 2>/dev/null || true
sudo systemctl mask thoth-hotspot 2>/dev/null || true
sudo systemctl mask hostapd 2>/dev/null || true
sudo systemctl mask dnsmasq 2>/dev/null || true

# Raspberry Pi Imager writes WiFi settings for NetworkManager on current Pi OS.
sudo systemctl unmask NetworkManager 2>/dev/null || true
sudo systemctl enable NetworkManager
sudo systemctl enable NetworkManager-wait-online.service 2>/dev/null || true
sudo systemctl restart NetworkManager || sudo systemctl start NetworkManager

# thoth.local is provided by Avahi/mDNS.
sudo hostnamectl set-hostname "$HOSTNAME" || true
if grep -q '^127.0.1.1' /etc/hosts 2>/dev/null; then
    sudo sed -i "s/^127.0.1.1.*/127.0.1.1\t$HOSTNAME/" /etc/hosts
else
    printf '127.0.1.1\t%s\n' "$HOSTNAME" | sudo tee -a /etc/hosts >/dev/null
fi
sudo systemctl enable avahi-daemon
sudo systemctl restart avahi-daemon || sudo systemctl start avahi-daemon

# Regenerate SSH host keys and ensure the SSH daemon is enabled
sudo dpkg-reconfigure openssh-server 2>/dev/null || true
sudo systemctl enable ssh
sudo systemctl start ssh

# Ensure thoth web service is enabled and started
sudo systemctl enable thoth.service
sudo systemctl restart thoth.service

# Ensure continuous collector is enabled and started
sudo systemctl enable thoth-collector.service
sudo systemctl restart thoth-collector.service

sudo touch "$FIRST_BOOT_FLAG"
sudo rm -f /var/run/thoth-firstboot

echo "First boot complete. Thoth is running at http://thoth.local:5000"
