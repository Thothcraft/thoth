#!/usr/bin/env bash
# Thoth installer for Raspberry Pi 3 (and other 1 GB boards).
#
# This is the lightweight counterpart to install.sh (which targets Pi 4/5).
# It clones the repo and runs first-boot.sh with THOTH_PROFILE=lite, which:
#   * skips the heavy Python wheels (torch, numba, scipy, pyfftw, matplotlib,
#     eventlet) — all lazily imported or unused, so the app runs without them;
#   * backs off the Brain registration/heartbeat intervals to keep a 1 GB Pi
#     responsive while it is online.
#
# Usage:  curl -fsSL <raw-url>/setup/install-rpi3.sh | sudo bash
set -euo pipefail

REPOSITORY_URL="https://github.com/Thothcraft/thoth.git"
INVOKING_USER="${SUDO_USER:-$(id -un)}"
INVOKING_HOME="$(getent passwd "$INVOKING_USER" | cut -d: -f6)"
INSTALL_ROOT="$INVOKING_HOME/thoth"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run this installer with sudo." >&2
  exit 1
fi
if [[ -z "$INVOKING_HOME" || "$INVOKING_HOME" == "/" ]]; then
  echo "Unable to determine the invoking user's home directory." >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Installing git..."
  apt-get update -qq
  apt-get install -y -qq git
fi

if [[ -d "$INSTALL_ROOT/.git" ]]; then
  sudo -u "$INVOKING_USER" git -C "$INSTALL_ROOT" fetch origin main
  sudo -u "$INVOKING_USER" git -C "$INSTALL_ROOT" checkout main
  sudo -u "$INVOKING_USER" git -C "$INSTALL_ROOT" pull --ff-only origin main
elif [[ -e "$INSTALL_ROOT" ]]; then
  echo "$INSTALL_ROOT exists but is not a Thoth git checkout; move it aside and rerun." >&2
  exit 1
else
  sudo -u "$INVOKING_USER" git clone --branch main --single-branch --depth 1 "$REPOSITORY_URL" "$INSTALL_ROOT"
fi

# Force the lightweight profile for Pi 3.
export THOTH_PROFILE=lite
exec bash "$INSTALL_ROOT/setup/first-boot.sh"
