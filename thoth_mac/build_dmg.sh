#!/usr/bin/env bash
# ============================================================================
# Thoth macOS — Build .dmg Installer
#
# This script:
#   1. Builds Thoth.app via py2app
#   2. Creates a pretty .dmg with drag-to-Applications layout
#
# Prerequisites:
#   pip install py2app          (already in requirements below)
#   brew install create-dmg     (optional — for branded DMG)
#
# Usage:
#   ./build_dmg.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
THOTH_ROOT="$(dirname "$SCRIPT_DIR")"
DIST_DIR="$SCRIPT_DIR/dist"
BUILD_DIR="$SCRIPT_DIR/build"
APP_NAME="Thoth"
DMG_NAME="Thoth-Installer"
VERSION="1.0.0"

echo "╔══════════════════════════════════════════╗"
echo "║    Thoth macOS — Build DMG Installer     ║"
echo "╚══════════════════════════════════════════╝"

# --- 0. Ensure venv and deps ---
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "➤ Creating build venv …"
    python3 -m venv "$SCRIPT_DIR/.venv"
fi
source "$SCRIPT_DIR/.venv/bin/activate"
pip install --upgrade pip -q
pip install -r "$THOTH_ROOT/thoth_core/requirements.txt" -q
pip install -r "$SCRIPT_DIR/requirements.txt" -q
pip install py2app -q

# --- 1. Clean previous builds ---
echo "➤ Cleaning previous builds …"
rm -rf "$DIST_DIR" "$BUILD_DIR"

# --- 2. Build .app bundle ---
echo "➤ Building $APP_NAME.app …"
cd "$SCRIPT_DIR"
python setup_app.py py2app 2>&1 | tail -5

if [ ! -d "$DIST_DIR/$APP_NAME.app" ]; then
    echo "❌ Build failed — $APP_NAME.app not found in dist/"
    exit 1
fi

echo "✅ $APP_NAME.app built successfully"

# --- 3. Create DMG ---
echo "➤ Creating DMG installer …"

if command -v create-dmg &>/dev/null; then
    # Branded DMG with background and icon layout
    create-dmg \
        --volname "$APP_NAME" \
        --volicon "$SCRIPT_DIR/Thoth.icns" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "$APP_NAME.app" 175 190 \
        --hide-extension "$APP_NAME.app" \
        --app-drop-link 425 190 \
        --no-internet-enable \
        "$DIST_DIR/${DMG_NAME}-${VERSION}.dmg" \
        "$DIST_DIR/$APP_NAME.app"
else
    # Fallback: simple DMG via hdiutil
    echo "  (install 'brew install create-dmg' for a branded DMG)"
    hdiutil create \
        -volname "$APP_NAME" \
        -srcfolder "$DIST_DIR/$APP_NAME.app" \
        -ov -format UDZO \
        "$DIST_DIR/${DMG_NAME}-${VERSION}.dmg"
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  ✅  DMG ready!                          ║"
echo "║  $DIST_DIR/${DMG_NAME}-${VERSION}.dmg"
echo "╚══════════════════════════════════════════╝"
