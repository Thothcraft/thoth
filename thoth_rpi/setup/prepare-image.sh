#!/usr/bin/env bash
# ============================================================================
# Thoth RPi — Image Preparation Script
#
# Run this on a Raspberry Pi to prepare a clean image for distribution.
# After running, shut down the Pi and create an image from the SD card.
#
# Prerequisites:
#   - Raspberry Pi OS Lite (64-bit) flashed via Imager
#   - WiFi configured via Imager (for initial setup only)
#   - SSH enabled via Imager
#
# NOTE: WiFi credentials are provided by Imager at burn time.
#       Account association uses /boot/firmware/thoth_credentials.json
#       which is downloaded from the Research Portal per device.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RPI_DIR="$(dirname "$SCRIPT_DIR")"
THOTH_ROOT="$(dirname "$RPI_DIR")"

echo "╔══════════════════════════════════════════╗"
echo "║   Thoth RPi — Image Preparation          ║"
echo "╚══════════════════════════════════════════╝"

# --- 1. Install Thoth ---
"$SCRIPT_DIR/install.sh"

# --- 2. Create example credentials file (user replaces at burn time) ---
echo "➤ Creating example credentials template …"
cat > /boot/firmware/thoth_credentials.json.example << 'EOF'
{
    "auth_token": "PASTE_YOUR_JWT_TOKEN_HERE",
    "brain_server_url": "https://web-production-d7d37.up.railway.app"
}
EOF

# --- 3. Clean up for imaging ---
echo "➤ Cleaning up for imaging …"
# Remove logs
sudo rm -rf "$THOTH_ROOT/logs/"*
# Remove local auth (user will provide via credentials file)
rm -rf "$THOTH_ROOT/data/config/auth.json"
# Clear bash history
cat /dev/null > ~/.bash_history
# Remove SSH keys (regenerated on first boot)
sudo rm -f /etc/ssh/ssh_host_*

echo ""
echo "✅  Image preparation complete!"
echo ""
echo "Next steps:"
echo "  1. sudo shutdown -h now"
echo "  2. Remove SD card and create image with dd or Win32DiskImager"
echo "  3. Distribute the image"
echo ""
echo "Users burn the image with Raspberry Pi Imager which provides:"
echo "  - WiFi credentials (SSID + password)"
echo "  - /boot/firmware/thoth_credentials.json (downloaded from Research Portal)"
