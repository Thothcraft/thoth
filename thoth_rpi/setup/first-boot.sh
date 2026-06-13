#!/usr/bin/env bash
# ============================================================================
# Thoth RPi — First Boot Script
#
# Runs once after the image is flashed to a new SD card.
# Regenerates SSH keys and ensures the service is running.
# ============================================================================

set -euo pipefail

# Regenerate SSH host keys
sudo dpkg-reconfigure openssh-server 2>/dev/null || true

# Ensure thoth service is enabled and started
sudo systemctl enable thoth.service
sudo systemctl start thoth.service

# Ensure continuous collector is enabled and started
sudo systemctl enable thoth-collector.service
sudo systemctl start thoth-collector.service

# Remove the first-boot flag
sudo rm -f /var/run/thoth-firstboot

echo "First boot complete — Thoth is running."
