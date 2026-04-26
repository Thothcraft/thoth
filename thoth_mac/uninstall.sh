#!/usr/bin/env bash
# ============================================================================
# Thoth macOS — Uninstaller
# ============================================================================

set -euo pipefail

PLIST_NAME="com.thothcraft.thoth.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Uninstalling Thoth macOS …"

# Stop the agent
launchctl unload "$PLIST_DST" 2>/dev/null || true

# Remove the plist
rm -f "$PLIST_DST"

# Optionally remove venv
if [ -d "$SCRIPT_DIR/.venv" ]; then
    read -rp "Remove virtual environment? [y/N] " yn
    if [[ "$yn" =~ ^[Yy]$ ]]; then
        rm -rf "$SCRIPT_DIR/.venv"
        echo "Virtual environment removed."
    fi
fi

echo "✅  Thoth has been uninstalled."
