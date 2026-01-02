#!/bin/bash
# Thoth Image Test Script
# Tests if prepare-image.sh worked correctly without rebooting

set -e

echo "=========================================="
echo "  Thoth Image Test Script"
echo "=========================================="
echo ""

THOTH_DIR="/home/pi/thoth"
ERRORS=0

# Function to check and report
check() {
    if [ "$2" = "0" ]; then
        echo "✅ $1"
    else
        echo "❌ $1"
        ERRORS=$((ERRORS + 1))
    fi
}

echo "[1/8] Checking systemd services..."
systemctl is-enabled thoth-firstboot >/dev/null 2>&1
check "thoth-firstboot service enabled" $?
systemctl is-enabled thoth-web >/dev/null 2>&1
check "thoth-web service enabled" $?

echo ""
echo "[2/8] Checking config files..."
[ -f /etc/hostapd/hostapd.conf ]
check "hostapd.conf exists" $?
[ -f /etc/dnsmasq.conf ]
check "dnsmasq.conf exists" $?
[ -f /etc/default/hostapd ]
check "hostapd default config exists" $?

echo ""
echo "[3/8] Checking config file contents..."
grep -q "Thoth-AP" /etc/hostapd/hostapd.conf 2>/dev/null
check "hostapd.conf has Thoth-AP SSID" $?
grep -q "192.168.4.1" /etc/dnsmasq.conf 2>/dev/null
check "dnsmasq.conf has correct IP" $?
grep -q "/etc/hostapd/hostapd.conf" /etc/default/hostapd 2>/dev/null
check "hostapd daemon conf points correctly" $?

echo ""
echo "[4/8] Checking scripts are executable..."
[ -x "$THOTH_DIR/setup/first-boot.sh" ]
check "first-boot.sh is executable" $?
[ -x "$THOTH_DIR/setup/hotspot-manager.sh" ]
check "hotspot-manager.sh is executable" $?

echo ""
echo "[5/8] Checking directories..."
[ -d "$THOTH_DIR/data/config" ]
check "Thoth config directory exists" $?
[ -d "$THOTH_DIR/logs" ]
check "Thoth logs directory exists" $?

echo ""
echo "[6/8] Checking WiFi interface..."
ip link show wlan0 >/dev/null 2>&1
check "wlan0 interface exists" $?
rfkill list wifi | grep -q "unblocked" >/dev/null 2>&1 || rfkill list | grep -q "wifi" >/dev/null 2>&1
check "WiFi not blocked by rfkill" $?

echo ""
echo "[7/8] Checking hostapd/dnsmasq status..."
systemctl is-masked hostapd >/dev/null 2>&1 && check "hostapd unmasked" 1 || check "hostapd unmasked" 0
systemctl is-masked dnsmasq >/dev/null 2>&1 && check "dnsmasq unmasked" 1 || check "dnsmasq unmasked" 0

echo ""
echo "[8/8] Checking for first-boot flags..."
[ ! -f /etc/thoth-first-boot-done ]
check "first-boot flag NOT present (good for testing)" $?
[ ! -f "$THOTH_DIR/data/config/wifi_configured.flag" ]
check "WiFi configured flag NOT present (good for testing)" $?

echo ""
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "✅ All tests passed! Image should work correctly."
    echo ""
    echo "You can now safely reboot to test:"
    echo "  sudo reboot"
    echo ""
    echo "After reboot, check for 'Thoth-AP' WiFi network"
else
    echo "❌ $ERRORS tests failed. Fix issues before rebooting."
    echo ""
    echo "Common fixes:"
    echo "  - Re-run prepare-image.sh"
    echo "  - Check service logs: journalctl -u thoth-firstboot"
fi
echo "=========================================="

exit $ERRORS
