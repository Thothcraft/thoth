#!/bin/bash
# Thoth WiFi Connection Script
# Called by the web app to connect to a WiFi network

SSID="$1"
PASSWORD="$2"
THOTH_DIR="/home/pi/thoth"
CONFIG_FLAG="$THOTH_DIR/data/config/wifi_configured.flag"

if [ -z "$SSID" ]; then
    echo "Usage: $0 <ssid> [password]"
    exit 1
fi

echo "Connecting to WiFi: $SSID"

# Stop hotspot
systemctl stop hostapd 2>/dev/null || true
systemctl stop dnsmasq 2>/dev/null || true

# Create wpa_supplicant config
WPA_CONF="/etc/wpa_supplicant/wpa_supplicant.conf"

# Backup existing config
cp "$WPA_CONF" "${WPA_CONF}.backup" 2>/dev/null || true

# Write new config
cat > "$WPA_CONF" << EOF
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=US

network={
    ssid="$SSID"
    psk="$PASSWORD"
    key_mgmt=WPA-PSK
}
EOF

# If no password, use open network config
if [ -z "$PASSWORD" ]; then
    cat > "$WPA_CONF" << EOF
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=US

network={
    ssid="$SSID"
    key_mgmt=NONE
}
EOF
fi

# Restart networking
ip addr flush dev wlan0
systemctl restart dhcpcd
wpa_cli -i wlan0 reconfigure

# Wait for connection
echo "Waiting for connection..."
for i in {1..30}; do
    if ping -c 1 8.8.8.8 &>/dev/null; then
        echo "Connected to $SSID"
        
        # Get assigned IP
        IP=$(ip -4 addr show wlan0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
        echo "IP Address: $IP"
        
        # Save configuration flag
        echo "$SSID" > "$CONFIG_FLAG"
        
        exit 0
    fi
    sleep 1
done

echo "Failed to connect to $SSID"
exit 1
