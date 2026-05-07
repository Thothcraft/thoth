#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/dist/macos-installer"
DMG_PATH="${ROOT_DIR}/dist/Thoth-macOS-Installer.dmg"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}/thoth"
mkdir -p "${ROOT_DIR}/dist"

cp -R "${ROOT_DIR}/thoth_core" "${BUILD_DIR}/thoth/"
cp -R "${ROOT_DIR}/thoth_mac" "${BUILD_DIR}/thoth/"
cp "${ROOT_DIR}/README.md" "${BUILD_DIR}/thoth/"
cp "${ROOT_DIR}/LICENSE" "${BUILD_DIR}/thoth/"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  cp "${ROOT_DIR}/.env" "${BUILD_DIR}/thoth/.env"
fi

cp "${ROOT_DIR}/thoth_mac/Install Thoth.command" "${BUILD_DIR}/Install Thoth.command"
chmod +x "${BUILD_DIR}/Install Thoth.command"
chmod +x "${BUILD_DIR}/thoth/thoth_mac/install.sh"

rm -f "${DMG_PATH}"
hdiutil create \
  -volname "Thoth Installer" \
  -srcfolder "${BUILD_DIR}" \
  -ov \
  -format UDZO \
  "${DMG_PATH}"

echo "Created ${DMG_PATH}"
