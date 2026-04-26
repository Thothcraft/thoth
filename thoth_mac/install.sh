#!/usr/bin/env bash
# ============================================================================
# Thoth macOS — One-Click Installer
#
# Usage:  ./install.sh
#
# What it does:
#   1. Copies Thoth.app to /Applications
#   2. Removes any pre-saved authentication data (ensures logged out on first run)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
THOTH_ROOT="$(dirname "$SCRIPT_DIR")"
APP_NAME="Thoth.app"
APP_SRC="$SCRIPT_DIR/$APP_NAME"
APP_DST="/Applications/$APP_NAME"

echo "╔══════════════════════════════════════════╗"
echo "║        Thoth macOS Installer             ║"
echo "╚══════════════════════════════════════════╝"

# --- 1. Remove any pre-saved auth data (ensure logged out on first run) ---
echo ""
echo "➤ Clearing any pre-saved authentication data …"
AUTH_DATA_DIR="$THOTH_ROOT/thoth_core/data/config"
AUTH_FILE="$AUTH_DATA_DIR/auth.json"
if [ -f "$AUTH_FILE" ]; then
    rm -f "$AUTH_FILE"
    echo "  ✓ Removed existing auth.json"
fi

# --- 2. Copy app to Applications folder ---
echo ""
echo "➤ Installing Thoth.app to /Applications …"
if [ ! -d "$APP_SRC" ]; then
    echo "❌ Error: Thoth.app not found at $APP_SRC"
    echo "   Please build the app first using: python setup_app.py py2app"
    exit 1
fi

# Remove existing installation if present
if [ -d "$APP_DST" ]; then
    echo "  Removing existing installation …"
    rm -rf "$APP_DST"
fi

# Copy the app
cp -R "$APP_SRC" "$APP_DST"
echo "  ✓ Thoth.app installed to /Applications"

echo ""
echo "✅  Thoth installed!"
echo "    To start:  Open Thoth.app from Applications or double-click it"
echo "    Dashboard:  http://localhost:8000"
echo "    Status bar:  look for 𓁟 in your menu bar"
echo ""
echo "To uninstall:  Drag Thoth.app from /Applications to Trash"
