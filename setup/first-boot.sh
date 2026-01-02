#!/bin/bash
# Thoth First Boot Setup Script
# This script runs once on first boot to initialize the device
# and start the captive portal for WiFi configuration

# Don't use set -e as we want to continue even if some commands fail

THOTH_DIR="/home/pi/thoth"
FIRST_BOOT_FLAG="/etc/thoth-first-boot-done"
LOG_FILE="/var/log/thoth-first-boot.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if first boot already completed
if [ -f "$FIRST_BOOT_FLAG" ]; then
    log "First boot already completed, exiting"
    exit 0
fi

log "=========================================="
log "  Thoth First Boot Initialization"
log "=========================================="

# Wait for system to fully boot
sleep 10

log "Setting up network interfaces..."

# Stop and disable NetworkManager if it exists (conflicts with hotspot)
if systemctl is-active NetworkManager &>/dev/null; then
    log "Stopping NetworkManager..."
    systemctl stop NetworkManager
    systemctl disable NetworkManager
    systemctl mask NetworkManager
fi

# Stop wpa_supplicant
systemctl stop wpa_supplicant 2>/dev/null || true

# Ensure WiFi is unblocked
rfkill unblock wifi 2>/dev/null || true

# Bring up wlan0
ip link set wlan0 up 2>/dev/null || true

# Wait for interface
for i in {1..10}; do
    if ip link show wlan0 &>/dev/null; then
        log "wlan0 interface is ready"
        break
    fi
    sleep 1
done

log "Configuring hotspot mode..."

# Ensure hostapd and dnsmasq are unmasked
log "Unmasking hostapd and dnsmasq..."
rm -f /etc/systemd/system/hostapd.service 2>/dev/null || true
rm -f /etc/systemd/system/dnsmasq.service 2>/dev/null || true
systemctl daemon-reload
systemctl unmask hostapd 2>/dev/null || true
systemctl unmask dnsmasq 2>/dev/null || true

# Configure static IP for AP mode
ip addr flush dev wlan0 2>/dev/null || true
ip addr add 192.168.4.1/24 dev wlan0
ip link set wlan0 up

log "Starting hostapd (WiFi Access Point)..."
# Retry hostapd start a few times as it can fail on first attempt
for i in 1 2 3; do
    if systemctl start hostapd; then
        log "hostapd started successfully"
        break
    else
        log "hostapd start attempt $i failed, retrying..."
        sleep 2
    fi
done

log "Starting dnsmasq (DHCP/DNS for captive portal)..."
systemctl start dnsmasq || log "Warning: dnsmasq failed to start"

# Enable IP forwarding for captive portal
echo 1 > /proc/sys/net/ipv4/ip_forward

# Set up iptables for captive portal redirect
iptables -t nat -A PREROUTING -i wlan0 -p tcp --dport 80 -j DNAT --to-destination 192.168.4.1:5000
iptables -t nat -A PREROUTING -i wlan0 -p tcp --dport 443 -j DNAT --to-destination 192.168.4.1:5000
iptables -t nat -A POSTROUTING -j MASQUERADE

log "Starting Thoth web application..."
systemctl start thoth-web || log "Warning: thoth-web failed to start"

log "=========================================="
log "  Thoth Hotspot Active!"
log "=========================================="
log "SSID: Thoth-AP"
log "Password: thoth123"
log "Portal: http://192.168.4.1:5000"
log "=========================================="

# Mark first boot as complete (but not WiFi configured yet)
touch "$FIRST_BOOT_FLAG"

log "First boot initialization complete"
log "Waiting for user to configure WiFi via captive portal..."
