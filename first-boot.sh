#!/usr/bin/env bash
# One-command Raspberry Pi bootstrap for a fresh Thoth clone.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${EUID}" -ne 0 ]; then
    echo "Run this script with sudo: sudo bash first-boot.sh" >&2
    exit 1
fi

exec bash "$SCRIPT_DIR/setup/first-boot.sh"
