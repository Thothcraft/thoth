#!/bin/bash
# Thoth Raspberry Pi Image Preparation Script
# Run this on a fresh Raspberry Pi OS to prepare it for imaging
# After running this, the SD card can be cloned as a Thoth image
#
# This script runs both parts:
#   Part 1: Non-network-disrupting setup (system deps, Python, services)
#   Part 2: Network-disrupting setup (hotspot, captive portal)
#
# You can also run the parts separately:
#   sudo ./prepare-image1.sh  (while connected to WiFi)
#   sudo ./prepare-image2.sh  (will disconnect from WiFi)

set -e

echo "=========================================="
echo "  Thoth Image Preparation Script"
echo "  (Complete Setup - Parts 1 & 2)"
echo "=========================================="
echo ""
echo "This script prepares a Raspberry Pi for Thoth deployment."
echo "After completion, the SD card can be imaged and distributed."
echo ""
echo "This will run both parts:"
echo "  Part 1: System dependencies and Thoth services"
echo "  Part 2: Hotspot and captive portal (disconnects WiFi)"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./prepare-image.sh)"
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running Part 1: Non-network-disrupting setup..."
echo ""

# Run Part 1
"$SCRIPT_DIR/prepare-image1.sh"

if [ $? -ne 0 ]; then
    echo "ERROR: Part 1 failed!"
    exit 1
fi

echo ""
echo "Part 1 completed successfully."
echo ""
echo "Running Part 2: Hotspot and captive portal setup..."
echo "WARNING: This will disconnect you from WiFi!"
echo ""
sleep 3

# Run Part 2 (auto-confirm with yes)
yes | "$SCRIPT_DIR/prepare-image2.sh"

if [ $? -ne 0 ]; then
    echo "ERROR: Part 2 failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "  Complete Image Preparation Done!"
echo "=========================================="
echo ""
echo "Both parts completed successfully."
echo "The system is ready to be imaged."
echo ""
