#!/bin/bash
# Thoth Raspberry Pi Image Preparation Script
# Run this on a fresh Raspberry Pi OS to prepare it for imaging
# After running this, the SD card can be cloned as a Thoth image

set -e

echo "=========================================="
echo "  Thoth Image Preparation Script"
echo "=========================================="
echo ""
echo "This script prepares a Raspberry Pi for Thoth deployment."
echo "After completion, the SD card can be imaged and distributed."
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./prepare-image.sh)"
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THOTH_DIR="$(dirname "$SCRIPT_DIR")"
THOTH_USER="pi"

echo "[1/11] Updating system packages..."
# Fix any interrupted dpkg installations
echo "Checking for interrupted package installations..."

# Remove corrupted packages that can't be reinstalled
echo "Removing any corrupted packages..."
dpkg --remove --force-remove-reinstreq rpi-eeprom 2>/dev/null || true

dpkg --configure -a

# Clean up any stale apt locks
rm -f /var/lib/apt/lists/lock
rm -f /var/cache/apt/archives/lock
rm -f /var/lib/dpkg/lock*

apt-get update
apt-get upgrade -y

# Reinstall rpi-eeprom if it was removed
echo "Ensuring rpi-eeprom is installed..."
apt-get install -y rpi-eeprom 2>/dev/null || echo "Note: rpi-eeprom installation skipped (not critical for image preparation)"

echo "[2/11] Installing system dependencies..."
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    hostapd \
    dnsmasq \
    dhcpcd5 \
    wireless-tools \
    net-tools \
    git \
    iptables \
    rfkill \
    dos2unix \
    libmicrohttpd-dev \
    libjson-c-dev \
    build-essential

echo "[3/11] Installing Thoth to /home/pi/thoth..."
# Copy thoth to standard location if not already there
if [ "$THOTH_DIR" != "/home/pi/thoth" ]; then
    mkdir -p /home/pi/thoth
    cp -r "$THOTH_DIR"/* /home/pi/thoth/
    THOTH_DIR="/home/pi/thoth"
    SCRIPT_DIR="$THOTH_DIR/setup"
fi

echo "[4/11] Creating Python virtual environment..."
cd "$THOTH_DIR"
python3 -m venv venv
source venv/bin/activate

# Set matplotlib backend for headless operation (no display)
export MPLBACKEND=Agg

echo "[5/11] Installing Python dependencies..."
pip install --upgrade pip
# Use Pi-specific requirements (excludes heavy ML libraries like PyTorch)
if [ -f "$THOTH_DIR/requirements-pi.txt" ]; then
    echo "Installing lightweight Pi requirements..."
    pip install -r requirements-pi.txt
else
    echo "Warning: requirements-pi.txt not found, using full requirements.txt (may take hours)..."
    pip install -r requirements.txt
fi

echo "[6/11] Setting up Thoth directories..."
mkdir -p "$THOTH_DIR/data/config"
mkdir -p "$THOTH_DIR/logs"
chown -R "$THOTH_USER:$THOTH_USER" "$THOTH_DIR"

echo "[7/11] Installing systemd services..."
# Copy service files
cp "$SCRIPT_DIR/thoth-web.service" /etc/systemd/system/
cp "$SCRIPT_DIR/thoth-hotspot.service" /etc/systemd/system/
cp "$SCRIPT_DIR/thoth-firstboot.service" /etc/systemd/system/

# Ensure paths are correct
sed -i "s|/home/pi/thoth|$THOTH_DIR|g" /etc/systemd/system/thoth-web.service
sed -i "s|/home/pi/thoth|$THOTH_DIR|g" /etc/systemd/system/thoth-hotspot.service
sed -i "s|/home/pi/thoth|$THOTH_DIR|g" /etc/systemd/system/thoth-firstboot.service

# Verify services are enabled
systemctl enable thoth-firstboot.service
systemctl enable thoth-web.service

# Reload systemd
systemctl daemon-reload

# Fix Windows line endings (CRLF -> LF) on all scripts and configs
echo "Fixing line endings on scripts and configs..."
find "$SCRIPT_DIR" -type f \( -name "*.sh" -o -name "*.conf" -o -name "*.service" \) -exec dos2unix {} \; 2>/dev/null || \
    find "$SCRIPT_DIR" -type f \( -name "*.sh" -o -name "*.conf" -o -name "*.service" \) -exec sed -i 's/\r$//' {} \;

# Ensure all scripts are executable and have correct permissions
find "$SCRIPT_DIR" -type f \( -name "*.sh" \) -exec chmod +x {} \;
chown -R "$THOTH_USER:$THOTH_USER" "$SCRIPT_DIR"

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

# Sync filesystem
sync

echo ""
echo "=========================================="
echo "  Image Preparation Complete!"
echo "=========================================="
echo ""
echo "The system is now ready to be imaged."
echo ""
echo "To create an image:"
echo "  1. Shutdown the Pi: sudo shutdown -h now"
echo "  2. Remove the SD card"
echo "  3. Use Win32DiskImager, dd, or Raspberry Pi Imager to create an image"
echo ""
echo "When the image boots on a new Pi:"
echo "  1. Thoth will automatically create WiFi hotspot: Thoth"
echo "  2. Connect to Thoth (password: thoth123)"
echo "  3. A captive portal will automatically pop up and redirect to http://192.168.4.1:5000"
echo "  4. Select your WiFi network and login to Brain"
echo ""
echo "To shutdown now for imaging: sudo shutdown -h now"
echo ""
