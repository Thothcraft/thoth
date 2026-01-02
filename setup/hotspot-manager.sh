#!/bin/bash
# Thoth Hotspot Manager
# Manages the WiFi hotspot for initial device setup

THOTH_DIR="/home/pi/thoth"
CONFIG_FLAG="$THOTH_DIR/data/config/wifi_configured.flag"
INTERFACE="wlan0"

start_hotspot() {
    echo "Starting Thoth hotspot..."
    
    # Check if WiFi is already configured
    if [ -f "$CONFIG_FLAG" ]; then
        echo "WiFi already configured, skipping hotspot"
        return 0
    fi
    
    # Unblock WiFi
    rfkill unblock wifi 2>/dev/null || true
    
    # Stop any existing WiFi connection
    systemctl stop wpa_supplicant 2>/dev/null || true
    systemctl stop NetworkManager 2>/dev/null || true
    
    # Wait for interface to be ready
    sleep 2
    
    # Configure static IP for AP mode
    ip addr flush dev $INTERFACE 2>/dev/null || true
    ip addr add 192.168.4.1/24 dev $INTERFACE 2>/dev/null || true
    ip link set $INTERFACE up 2>/dev/null || true
    
    # Start hostapd and dnsmasq with retries
    for i in 1 2 3; do
        if systemctl start hostapd; then
            echo "hostapd started"
            break
        fi
        echo "hostapd attempt $i failed, retrying..."
        sleep 2
    done
    
    systemctl start dnsmasq 2>/dev/null || true
    
    echo "Hotspot started: Thoth-AP (192.168.4.1)"
}

stop_hotspot() {
    echo "Stopping Thoth hotspot..."
    
    # Stop hostapd and dnsmasq
    systemctl stop hostapd 2>/dev/null || true
    systemctl stop dnsmasq 2>/dev/null || true
    
    # Restore normal WiFi
    ip addr flush dev $INTERFACE
    systemctl restart dhcpcd
    systemctl restart wpa_supplicant 2>/dev/null || true
    
    echo "Hotspot stopped"
}

case "$1" in
    start)
        start_hotspot
        ;;
    stop)
        stop_hotspot
        ;;
    restart)
        stop_hotspot
        sleep 2
        start_hotspot
        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
        ;;
esac
