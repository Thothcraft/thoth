#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGE_ROOT="${SCRIPT_DIR}/thoth"
INSTALLER="${PACKAGE_ROOT}/thoth_mac/install.sh"

osascript <<'APPLESCRIPT'
display dialog "Thoth installer will set up the app and register it to launch on login. Continue?" buttons {"Cancel", "Install"} default button "Install" with icon note
APPLESCRIPT

if [[ ! -x "${INSTALLER}" ]]; then
  osascript -e 'display dialog "Installer script is missing or not executable." buttons {"OK"} default button "OK" with icon stop'
  exit 1
fi

cd "${PACKAGE_ROOT}/thoth_mac"
./install.sh

osascript <<'APPLESCRIPT'
display dialog "Thoth installation completed successfully. Look for the Thoth icon in the macOS menu bar." buttons {"Open Dashboard", "Done"} default button "Open Dashboard" with icon note
set clicked to button returned of result
if clicked is "Open Dashboard" then
  do shell script "open http://localhost:5000"
end if
APPLESCRIPT
