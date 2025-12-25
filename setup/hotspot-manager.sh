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
    
    # Stop any existing WiFi connection
    systemctl stop wpa_supplicant 2>/dev/null || true
    
    # Configure static IP for AP mode
    ip addr flush dev $INTERFACE
    ip addr add 192.168.4.1/24 dev $INTERFACE
    ip link set $INTERFACE up
    
    # Start hostapd and dnsmasq
    systemctl start hostapd
    systemctl start dnsmasq
    
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
