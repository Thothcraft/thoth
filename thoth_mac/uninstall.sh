#!/usr/bin/env bash
# ============================================================================
# Thoth macOS — Uninstaller
# ============================================================================

set -euo pipefail

PLIST_NAME="com.thothcraft.thoth.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Uninstalling Thoth macOS …"

# Signal the Brain server that this device is going offline before stopping it
THOTH_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$THOTH_ROOT/data/config/device_config.json"
if [ -f "$CONFIG_FILE" ]; then
    DEVICE_ID=$(python3 -c "import json,sys; d=json.load(open('$CONFIG_FILE')); print(d.get('device_id',''))" 2>/dev/null || true)
    SERVER_URL=$(python3 -c "import json,sys; d=json.load(open('$CONFIG_FILE')); print(d.get('brain_server_url',''))" 2>/dev/null || true)
    if [ -n "$DEVICE_ID" ] && [ -n "$SERVER_URL" ]; then
        curl -s -X POST "$SERVER_URL/device/$DEVICE_ID/offline" \
             -H "Content-Type: application/json" -d '{}' --max-time 3 >/dev/null 2>&1 || true
        echo "  Sent offline signal to Brain server."
    fi
fi

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
