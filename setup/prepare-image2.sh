#!/bin/bash
# Thoth Raspberry Pi Image Preparation Script - Part 2
# Network-Disrupting Setup (Hotspot & Captive Portal)
# This part configures the hotspot and captive portal
# WARNING: This will disconnect you from WiFi!

set -e

echo "=========================================="
echo "  Thoth Image Preparation - Part 2"
echo "  (Hotspot & Captive Portal Setup)"
echo "=========================================="
echo ""
echo "WARNING: This script will disconnect you from WiFi!"
echo "It will configure the Raspberry Pi as a WiFi hotspot."
echo ""
echo "Make sure you have completed Part 1 first."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./prepare-image2.sh)"
    exit 1
fi

# Check if Part 1 was completed
if [ ! -f /etc/thoth-image-part1-complete ]; then
    echo "ERROR: Part 1 not completed!"
    echo "Please run prepare-image1.sh first."
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THOTH_DIR="$(dirname "$SCRIPT_DIR")"
THOTH_USER="pi"

echo "[8/11] Disabling NetworkManager and configuring hotspot..."

# Disable NetworkManager completely (it conflicts with our hotspot)
if systemctl is-active NetworkManager &>/dev/null; then
    echo "Disabling NetworkManager..."
    systemctl stop NetworkManager
    systemctl disable NetworkManager
    systemctl mask NetworkManager
fi

# Also disable wpa_supplicant auto-start (we control it manually)
systemctl disable wpa_supplicant 2>/dev/null || true

# Ensure dhcpcd is enabled (critical for interface management on Pi)
systemctl enable dhcpcd 2>/dev/null || true
systemctl start dhcpcd 2>/dev/null || true

# Backup original configs
[ -f /etc/hostapd/hostapd.conf ] && cp /etc/hostapd/hostapd.conf /etc/hostapd/hostapd.conf.backup
[ -f /etc/dnsmasq.conf ] && cp /etc/dnsmasq.conf /etc/dnsmasq.conf.backup

# Copy our configs
cp "$SCRIPT_DIR/hostapd.conf" /etc/hostapd/hostapd.conf
cp "$SCRIPT_DIR/dnsmasq.conf" /etc/dnsmasq.conf

# Fix line endings on the copied config files
dos2unix /etc/hostapd/hostapd.conf 2>/dev/null || sed -i 's/\r$//' /etc/hostapd/hostapd.conf
dos2unix /etc/dnsmasq.conf 2>/dev/null || sed -i 's/\r$//' /etc/dnsmasq.conf

# Point hostapd to config
echo 'DAEMON_CONF="/etc/hostapd/hostapd.conf"' > /etc/default/hostapd

# Unmask hostapd and dnsmasq (required on Pi OS - they come masked by default)
# First remove any existing service files that might be mask symlinks
rm -f /etc/systemd/system/hostapd.service 2>/dev/null || true
rm -f /etc/systemd/system/dnsmasq.service 2>/dev/null || true
systemctl daemon-reload

# Now unmask them
systemctl unmask hostapd
systemctl unmask dnsmasq

# Verify they are unmasked
echo "Verifying hostapd status:"
systemctl is-enabled hostapd || true

# Disable hostapd and dnsmasq from starting automatically
# (first-boot service will control them)
systemctl disable hostapd 2>/dev/null || true
systemctl disable dnsmasq 2>/dev/null || true
systemctl stop hostapd 2>/dev/null || true
systemctl stop dnsmasq 2>/dev/null || true

echo "[9/11] Installing and configuring Nodogsplash captive portal..."
# Copy local Nodogsplash to /tmp and build (exclude .git directory)
cd /tmp
if [ -d "nodogsplash" ]; then
    rm -rf nodogsplash
fi

echo "Copying nodogsplash from $THOTH_DIR/nodogsplash..."

# Debug: Check source directory
echo "DEBUG: THOTH_DIR = $THOTH_DIR"
echo "DEBUG: Source path = $THOTH_DIR/nodogsplash"
if [ ! -d "$THOTH_DIR/nodogsplash" ]; then
    echo "ERROR: Source directory $THOTH_DIR/nodogsplash does not exist!"
    echo "Contents of $THOTH_DIR:"
    ls -la "$THOTH_DIR"
    exit 1
fi

echo "DEBUG: Source directory exists, checking for Makefile in source..."
if [ ! -f "$THOTH_DIR/nodogsplash/Makefile" ]; then
    echo "ERROR: Makefile not found in source directory $THOTH_DIR/nodogsplash"
    echo "Contents of source directory:"
    ls -la "$THOTH_DIR/nodogsplash"
    exit 1
fi
echo "DEBUG: Makefile found in source directory"

# Use rsync if available, otherwise cp with manual .git exclusion
if command -v rsync &> /dev/null; then
    echo "DEBUG: Using rsync to copy files..."
    rsync -av --exclude='.git' "$THOTH_DIR/nodogsplash/" /tmp/nodogsplash/
else
    echo "DEBUG: Using cp to copy files..."
    cp -rv "$THOTH_DIR/nodogsplash" /tmp/
    rm -rf /tmp/nodogsplash/.git 2>/dev/null || true
fi

echo "DEBUG: Copy complete, checking destination..."
cd /tmp/nodogsplash
echo "Current directory: $(pwd)"
echo "Contents of /tmp/nodogsplash:"
ls -la

echo "Checking for Makefile in destination..."
if [ ! -f "Makefile" ]; then
    echo "ERROR: Makefile not found in $(pwd) after copy"
    exit 1
fi
echo "DEBUG: Makefile successfully copied to destination"

# Fix line endings on Makefile and source files (Windows CRLF -> Unix LF)
echo "Fixing line endings in nodogsplash files..."
find . -type f \( -name "Makefile" -o -name "*.c" -o -name "*.h" -o -name "*.sh" \) -exec dos2unix {} \; 2>/dev/null || \
    find . -type f \( -name "Makefile" -o -name "*.c" -o -name "*.h" -o -name "*.sh" \) -exec sed -i 's/\r$//' {} \;

echo "Building nodogsplash..."
make
echo "Installing nodogsplash..."
make install

# Install systemd service file
echo "Installing nodogsplash systemd service..."
if [ -f "debian/nodogsplash.service" ]; then
    cp debian/nodogsplash.service /etc/systemd/system/nodogsplash.service
    dos2unix /etc/systemd/system/nodogsplash.service 2>/dev/null || sed -i 's/\r$//' /etc/systemd/system/nodogsplash.service
    systemctl daemon-reload
    echo "Nodogsplash service file installed"
else
    echo "Warning: nodogsplash.service not found in debian directory"
fi

# Create configuration directory
mkdir -p /etc/nodogsplash/htdocs

# Copy configuration files
if [ -f "$THOTH_DIR/nodogsplash.conf" ]; then
    cp "$THOTH_DIR/nodogsplash.conf" /etc/nodogsplash/nodogsplash.conf
    dos2unix /etc/nodogsplash/nodogsplash.conf 2>/dev/null || sed -i 's/\r$//' /etc/nodogsplash/nodogsplash.conf
    echo "Nodogsplash configuration installed"
else
    echo "Warning: nodogsplash.conf not found in $THOTH_DIR"
fi

if [ -f "$THOTH_DIR/splash.html" ]; then
    cp "$THOTH_DIR/splash.html" /etc/nodogsplash/htdocs/splash.html
    dos2unix /etc/nodogsplash/htdocs/splash.html 2>/dev/null || sed -i 's/\r$//' /etc/nodogsplash/htdocs/splash.html
    echo "Nodogsplash splash page installed"
else
    echo "Warning: splash.html not found in $THOTH_DIR"
fi

# Enable Nodogsplash service (but don't start it yet - first-boot will handle it)
systemctl enable nodogsplash
systemctl stop nodogsplash 2>/dev/null || true

cd "$THOTH_DIR"

echo "[10/11] Enabling first-boot service..."
# Enable the first-boot service to run on next boot
systemctl enable thoth-firstboot.service

# Enable thoth-web to start after first-boot completes
systemctl enable thoth-web.service

echo "[11/11] Cleaning up for imaging..."
# Remove first-boot flag if it exists (so it runs on next boot)
rm -f /etc/thoth-first-boot-done

# Remove WiFi configuration flag
rm -f "$THOTH_DIR/data/config/wifi_configured.flag"

# Clear any saved credentials
rm -f "$THOTH_DIR/data/config/auth.json"
rm -f "$THOTH_DIR/data/config/device_config.json"

# Clear logs
rm -f "$THOTH_DIR/logs/"*.log
rm -f /var/log/thoth*.log
rm -f /var/log/dnsmasq.log
journalctl --vacuum-time=1s 2>/dev/null || true

# Clear bash history
cat /dev/null > /home/pi/.bash_history 2>/dev/null || true
cat /dev/null > /root/.bash_history 2>/dev/null || true

# Clear apt cache
apt-get clean
rm -rf /var/cache/apt/archives/*

# Remove temporary files (but keep system-critical ones)
find /tmp -type f -name "*.tmp" -delete 2>/dev/null || true
find /tmp -type f -name "temp*" -delete 2>/dev/null || true
# Don't delete /tmp/* - it contains critical system files for boot

# NOTE: SSH key removal and regeneration is only for final image creation
# Commented out for testing purposes
# Remove SSH host keys (will be regenerated on first boot)
# rm -f /etc/ssh/ssh_host_*

# Remove any WiFi credentials used during image building
cat > /etc/wpa_supplicant/wpa_supplicant.conf << 'WPAEOF'
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=US
WPAEOF

# NOTE: SSH regeneration is only for final image creation
# Create flag to regenerate SSH keys on first boot
# touch /etc/ssh/sshd_not_to_be_run
# cat > /etc/rc.local << 'EOF'
# #!/bin/bash
# # Regenerate SSH host keys on first boot
# if [ -f /etc/ssh/sshd_not_to_be_run ]; then
#     rm -f /etc/ssh/sshd_not_to_be_run
#     dpkg-reconfigure openssh-server
#     systemctl restart ssh
# fi
# exit 0
# EOF
# chmod +x /etc/rc.local

# Create completion flag
touch /etc/thoth-image-part2-complete

# Sync filesystem
sync

echo ""
echo "=========================================="
echo "  Part 2 Complete!"
echo "=========================================="
echo ""
echo "Hotspot and captive portal are configured."
echo "You are now disconnected from WiFi."
echo ""
echo "The system is ready to be imaged."
echo ""
echo "To create an image:"
echo "  1. Shutdown the Pi: sudo shutdown -h now"
echo "  2. Remove the SD card"
echo "  3. Use Win32DiskImager, dd, or Raspberry Pi Imager to create an image"
echo ""
echo "When the image boots on a new Pi:"
echo "  1. Thoth will automatically create WiFi hotspot: Thoth"
echo "  2. Connect to Thoth (password: thoth123)"
echo "  3. A captive portal will redirect to the Thoth web interface"
echo "  4. Select your WiFi network and login to Brain"
echo ""
echo "To shutdown now for imaging: sudo shutdown -h now"
echo ""
