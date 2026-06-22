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
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

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
sudo systemctl restart NetworkManager-wait-online.service 2>/dev/null || true

# Apply optional boot-partition provisioning for image burners that cannot
# customize custom Raspberry Pi images.
if [ -x "$SCRIPT_DIR/provision-boot.py" ]; then
    sudo "$SCRIPT_DIR/provision-boot.py" || true
else
    sudo python3 "$SCRIPT_DIR/provision-boot.py" || true
fi

# thoth.local is provided by Avahi/mDNS unless provisioning supplied a hostname.
CURRENT_HOSTNAME="$(hostnamectl --static 2>/dev/null || hostname || true)"
if [ -z "$CURRENT_HOSTNAME" ] || [ "$CURRENT_HOSTNAME" = "raspberrypi" ]; then
    sudo hostnamectl set-hostname "$HOSTNAME" || true
    CURRENT_HOSTNAME="$HOSTNAME"
fi
if grep -q '^127.0.1.1' /etc/hosts 2>/dev/null; then
    sudo sed -i "s/^127.0.1.1.*/127.0.1.1\t$CURRENT_HOSTNAME/" /etc/hosts
else
    printf '127.0.1.1\t%s\n' "$CURRENT_HOSTNAME" | sudo tee -a /etc/hosts >/dev/null
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

echo "First boot complete. Thoth is running at http://$CURRENT_HOSTNAME.local:5000"
