#!/usr/bin/env bash
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

if [[ -d "$INSTALL_ROOT/.git" ]]; then
  sudo -u "$INVOKING_USER" git -C "$INSTALL_ROOT" fetch origin main
  sudo -u "$INVOKING_USER" git -C "$INSTALL_ROOT" checkout main
  sudo -u "$INVOKING_USER" git -C "$INSTALL_ROOT" pull --ff-only origin main
elif [[ -e "$INSTALL_ROOT" ]]; then
  echo "$INSTALL_ROOT exists but is not a Thoth git checkout; move it aside and rerun." >&2
  exit 1
else
  sudo -u "$INVOKING_USER" git clone --branch main --single-branch "$REPOSITORY_URL" "$INSTALL_ROOT"
fi

exec bash "$INSTALL_ROOT/setup/first-boot.sh"
