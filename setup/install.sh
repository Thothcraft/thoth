#!/usr/bin/env bash
# Thoth node installer — macOS + Linux (Pi).
#
#   curl -fsSL https://raw.githubusercontent.com/Thothcraft/thoth/main/setup/install.sh | bash
#
# Optional env:
#   THOTH_HOSTNAME=thoth-denver   mDNS/portal name (thoth-<name>.local)
#   THOTH_DASHBOARD_PORT=8080     dashboard UI port (default 80 → falls back
#                               to the API port when unprivileged)
#   THOTH_API_PORT=5000           local API port
set -euo pipefail

REPOSITORY_URL="https://github.com/Thothcraft/thoth.git"
PKG_SPEC="thoth-node @ git+https://github.com/Thothcraft/thoth"
OS="$(uname -s)"

# ---------------------------------------------------------------------------
# config.json writer — shared by both flows; runs as the real (non-root) user.
# ---------------------------------------------------------------------------
write_config() {
  local cfg_home="${THOTH_HOME:-$HOME/.thoth}"
  mkdir -p "$cfg_home"
  python3 - "$cfg_home/config.json" <<'PYEOF'
import json, os, sys
path = sys.argv[1]
cfg = {}
if os.path.exists(path):
    try:
        cfg = json.load(open(path))
    except Exception:
        cfg = {}
name = os.environ.get("THOTH_HOSTNAME")
if name:
    cfg["device_name"] = name
if os.environ.get("THOTH_DASHBOARD_PORT"):
    cfg["dashboard_port"] = int(os.environ["THOTH_DASHBOARD_PORT"])
if os.environ.get("THOTH_API_PORT"):
    cfg["local_port"] = int(os.environ["THOTH_API_PORT"])
json.dump(cfg, open(path, "w"), indent=2)
os.chmod(path, 0o600)
print(f"config → {path}: {cfg}")
PYEOF
}

print_next() {
  local name="${THOTH_HOSTNAME:-$(hostname -s | tr '[:upper:]' '[:lower:]')}"
  local dash_port="${THOTH_DASHBOARD_PORT:-80}"
  local api_port="${THOTH_API_PORT:-5000}"
  echo
  echo "== thoth-node installed =="
  echo "Next: pair this node to your portal account:"
  echo "    thoth pair            # prompts for portal username + password"
  echo "Dashboard: http://${name}.local:${dash_port}  "
  echo "  (falls back to :${api_port} when :${dash_port} is unavailable,"
  echo "   or use the device's LAN IP)"
}

# ---------------------------------------------------------------------------
# macOS — user venv + launchd agent + Bonjour (no sudo, no port-80 bind)
# ---------------------------------------------------------------------------
if [[ "$OS" == "Darwin" ]]; then
  VENV="$HOME/.thoth-node"
  PYBIN="$(command -v python3 || true)"
  [[ -n "$PYBIN" ]] || { echo "python3 not found — install Xcode CLT: xcode-select --install" >&2; exit 1; }

  [[ -x "$VENV/bin/python" ]] || "$PYBIN" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet --force-reinstall --no-deps "$PKG_SPEC"

  write_config

  # launchd user agent — starts at login, respawns on crash.
  PLIST="$HOME/Library/LaunchAgents/com.thothcraft.node.plist"
  mkdir -p "$HOME/Library/LaunchAgents" "$HOME/.thoth"
  cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.thothcraft.node</string>
  <key>ProgramArguments</key>
  <array>
    <string>$VENV/bin/python</string>
    <string>-m</string><string>thoth</string>
    <string>daemon</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$HOME/.thoth/daemon.log</string>
  <key>StandardErrorPath</key><string>$HOME/.thoth/daemon.log</string>
</dict></plist>
PLIST
  launchctl bootout "gui/$(id -u)/com.thothcraft.node" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
  launchctl kickstart "gui/$(id -u)/com.thothcraft.node" 2>/dev/null || true

  # Friendly .local name via Bonjour — hostname change needs sudo; print the
  # one-liner instead of failing the whole install when not root.
  if [[ -n "${THOTH_HOSTNAME:-}" ]]; then
    if sudo -n scutil --set LocalHostName "$THOTH_HOSTNAME" 2>/dev/null; then
      echo "Bonjour name set: ${THOTH_HOSTNAME}.local"
    else
      echo "To advertise ${THOTH_HOSTNAME}.local, run:"
      echo "    sudo scutil --set LocalHostName $THOTH_HOSTNAME"
    fi
  fi

  # CLI on PATH without activating the venv.
  mkdir -p "$HOME/.local/bin"
  ln -sf "$VENV/bin/thoth" "$HOME/.local/bin/thoth"
  echo "thoth CLI → ~/.local/bin/thoth (add to PATH if needed)"

  print_next
  exit 0
fi

# ---------------------------------------------------------------------------
# Linux / Pi — original flow: clone + first-boot.sh (systemd, avahi, deps)
# ---------------------------------------------------------------------------
INVOKING_USER="${SUDO_USER:-$(id -un)}"
INVOKING_HOME="$(getent passwd "$INVOKING_USER" | cut -d: -f6)"
INSTALL_ROOT="$INVOKING_HOME/thoth"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run this installer with sudo on Linux." >&2
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

exec bash "$INSTALL_ROOT/setup/first-boot.sh"
