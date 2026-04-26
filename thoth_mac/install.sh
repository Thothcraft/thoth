#!/usr/bin/env bash
# ============================================================================
# Thoth macOS — One-Click Installer
#
# Usage:  ./install.sh
#
# What it does:
#   1. Creates a Python virtual-environment in thoth_mac/.venv
#   2. Installs core + macOS dependencies
#   3. Removes any pre-saved authentication data (ensures logged out on first run)
#   4. Copies the LaunchAgent plist to ~/Library/LaunchAgents/
#   5. Loads the agent so Thoth starts on every login
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
THOTH_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$SCRIPT_DIR/.venv"
PLIST_NAME="com.thothcraft.thoth.plist"
PLIST_SRC="$SCRIPT_DIR/$PLIST_NAME"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"

echo "╔══════════════════════════════════════════╗"
echo "║        Thoth macOS Installer             ║"
echo "╚══════════════════════════════════════════╝"

# --- 1. Python venv ---
echo ""
echo "➤ Creating virtual environment …"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "➤ Upgrading pip …"
pip install --upgrade pip -q

echo "➤ Installing core dependencies …"
pip install -r "$THOTH_ROOT/thoth_core/requirements.txt" -q

echo "➤ Installing macOS dependencies …"
pip install -r "$SCRIPT_DIR/requirements.txt" -q

# --- 2. Remove any pre-saved auth data (ensure logged out on first run) ---
echo "➤ Clearing any pre-saved authentication data …"
AUTH_DATA_DIR="$THOTH_ROOT/thoth_core/data/config"
AUTH_FILE="$AUTH_DATA_DIR/auth.json"
if [ -f "$AUTH_FILE" ]; then
    rm -f "$AUTH_FILE"
    echo "  ✓ Removed existing auth.json"
fi

# --- 3. Generate LaunchAgent plist with correct paths ---
echo "➤ Configuring LaunchAgent …"
PYTHON_BIN="$VENV_DIR/bin/python"
APP_SCRIPT="$SCRIPT_DIR/app.py"

cat > "$PLIST_SRC" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.thothcraft.thoth</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON_BIN}</string>
        <string>${APP_SCRIPT}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>${THOTH_ROOT}/logs/thoth_mac.log</string>
    <key>StandardErrorPath</key>
    <string>${THOTH_ROOT}/logs/thoth_mac_err.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>THOTH_ROOT</key>
        <string>${THOTH_ROOT}</string>
    </dict>
</dict>
</plist>
EOF

# --- 4. Install LaunchAgent ---
mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$THOTH_ROOT/logs"
cp "$PLIST_SRC" "$PLIST_DST"

# Unload first if already loaded (idempotent)
launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"

echo ""
echo "✅  Thoth installed!"
echo "    Dashboard:  http://localhost:8000"
echo "    Status bar:  look for 𓁟 in your menu bar"
echo ""
echo "To uninstall, run:  ./uninstall.sh"
