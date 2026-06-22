#!/usr/bin/env python3
"""Apply optional Thoth boot-partition provisioning.

This is a fallback for image burners that cannot customize Raspberry Pi OS
settings for custom images. Users can place thoth_provisioning.json on the
boot partition after flashing; first boot consumes it and removes secrets.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


BOOT_FILES = (
    Path("/boot/firmware/thoth_provisioning.json"),
    Path("/boot/thoth_provisioning.json"),
)
THOTH_CREDENTIALS_FILE = Path("/boot/firmware/thoth_credentials.json")
HOSTNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,62}$")


def run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def first_present(values: list[Any]) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def read_config() -> tuple[Path | None, dict[str, Any]]:
    for path in BOOT_FILES:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return path, json.load(handle)
    return None, {}


def configure_hostname(config: dict[str, Any]) -> None:
    hostname = str(config.get("hostname") or "").strip()
    if not hostname:
        return
    if not HOSTNAME_RE.match(hostname):
        print(f"Skipping invalid hostname: {hostname}")
        return

    run(["hostnamectl", "set-hostname", hostname])
    hosts_path = Path("/etc/hosts")
    try:
        lines = hosts_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []

    replacement = f"127.0.1.1\t{hostname}"
    changed = False
    for idx, line in enumerate(lines):
        if line.startswith("127.0.1.1"):
            lines[idx] = replacement
            changed = True
            break
    if not changed:
        lines.append(replacement)
    hosts_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_timezone(config: dict[str, Any]) -> None:
    timezone = str(config.get("timezone") or "").strip()
    if timezone:
        run(["timedatectl", "set-timezone", timezone])


def configure_ssh(config: dict[str, Any]) -> None:
    ssh_enabled = config.get("ssh_enabled", config.get("ssh"))
    if ssh_enabled is False:
        run(["systemctl", "disable", "--now", "ssh"])
    elif ssh_enabled is True:
        run(["systemctl", "enable", "ssh"])
        run(["systemctl", "start", "ssh"])


def configure_wifi(config: dict[str, Any]) -> None:
    wifi = config.get("wifi") if isinstance(config.get("wifi"), dict) else {}
    ssid = first_present([wifi.get("ssid"), config.get("wifi_ssid"), config.get("ssid")])
    if not ssid:
        return

    ssid = str(ssid)
    password = first_present([wifi.get("password"), config.get("wifi_password"), config.get("password")])
    country = first_present([wifi.get("country"), config.get("wifi_country"), config.get("country")])
    hidden = bool(wifi.get("hidden", config.get("wifi_hidden", False)))

    run(["systemctl", "unmask", "NetworkManager"])
    run(["systemctl", "enable", "NetworkManager"])
    run(["systemctl", "start", "NetworkManager"])

    if country:
        country = str(country).upper()
        run(["raspi-config", "nonint", "do_wifi_country", country])
        run(["iw", "reg", "set", country])

    if shutil.which("nmcli"):
        run(["nmcli", "radio", "wifi", "on"])
        existing = run(["nmcli", "-t", "-f", "NAME", "connection", "show"])
        for name in existing.stdout.splitlines():
            if name == ssid:
                run(["nmcli", "connection", "delete", ssid])

        run(["nmcli", "connection", "add", "type", "wifi", "ifname", "wlan0", "con-name", ssid, "ssid", ssid])
        if password:
            run(["nmcli", "connection", "modify", ssid, "wifi-sec.key-mgmt", "wpa-psk", "wifi-sec.psk", str(password)])
        else:
            run(["nmcli", "connection", "modify", ssid, "wifi-sec.key-mgmt", "none"])
        if hidden:
            run(["nmcli", "connection", "modify", ssid, "802-11-wireless.hidden", "yes"])
        run(["nmcli", "connection", "modify", ssid, "connection.autoconnect", "yes"])
        run(["nmcli", "connection", "up", ssid])
        return

    write_wpa_supplicant(ssid, str(password) if password else None, str(country) if country else None)


def write_wpa_supplicant(ssid: str, password: str | None, country: str | None) -> None:
    conf = Path("/etc/wpa_supplicant/wpa_supplicant.conf")
    conf.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"country={country or 'US'}",
        "ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev",
        "update_config=1",
        "",
        "network={",
        f'    ssid="{ssid}"',
    ]
    if password:
        lines.extend(["    key_mgmt=WPA-PSK", f'    psk="{password}"'])
    else:
        lines.append("    key_mgmt=NONE")
    lines.append("}")
    conf.write_text("\n".join(lines) + "\n", encoding="utf-8")
    conf.chmod(0o600)
    run(["systemctl", "restart", "wpa_supplicant"])


def write_thoth_credentials(config: dict[str, Any]) -> None:
    auth_token = first_present([config.get("auth_token"), config.get("brain_auth_token")])
    brain_server_url = config.get("brain_server_url")
    credentials = config.get("thoth_credentials")
    if isinstance(credentials, dict):
        auth_token = first_present([credentials.get("auth_token"), auth_token])
        brain_server_url = first_present([credentials.get("brain_server_url"), brain_server_url])

    if not auth_token:
        return

    THOTH_CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    THOTH_CREDENTIALS_FILE.write_text(
        json.dumps(
            {
                "auth_token": auth_token,
                "brain_server_url": brain_server_url or "https://web-production-d7d37.up.railway.app",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    THOTH_CREDENTIALS_FILE.chmod(0o600)


def main() -> int:
    source, config = read_config()
    if not source:
        print("No Thoth provisioning file found on boot partition.")
        return 0

    print(f"Applying Thoth provisioning from {source}")
    configure_hostname(config)
    configure_timezone(config)
    configure_ssh(config)
    configure_wifi(config)
    write_thoth_credentials(config)

    try:
        source.unlink()
        print(f"Deleted provisioning file: {source}")
    except OSError as exc:
        print(f"Could not delete provisioning file {source}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
