"""Wi-Fi control via NetworkManager — the provisioning path's egress.

Thin ``nmcli`` wrapper with an injectable backend so the provisioning
state machine is testable off-hardware. Everything the daemon needs is
``status() / scan() / connect() / disconnect()``; AP-mode support
(``ap_start``/``ap_stop``) backs Phase 4's fallback.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_IFACE = "wlan0"


def ssid_hash(ssid: str) -> str:
    """SSIDs never leave the node raw — the wire carries this hash."""
    return hashlib.sha256(str(ssid).encode()).hexdigest()[:12]


@dataclass
class WifiStatus:
    present: bool          # wlan iface exists / NM sees it
    connected: bool
    ssid: Optional[str] = None
    iface: str = DEFAULT_IFACE
    ip: Optional[str] = None


@dataclass
class WifiNetwork:
    ssid: str
    signal: int = 0
    security: str = ""


class NmcliWifiManager:
    """NetworkManager via nmcli — the real backend on Raspberry Pi OS."""

    def __init__(self, iface: str = DEFAULT_IFACE):
        self.iface = iface

    @staticmethod
    def available() -> bool:
        return shutil.which("nmcli") is not None

    def _run(self, *args: str, timeout: float = 15.0) -> str:
        out = subprocess.run(
            ["nmcli", *args], capture_output=True, text=True,
            timeout=timeout)
        if out.returncode != 0:
            raise RuntimeError(
                f"nmcli {' '.join(args)}: {out.stderr.strip() or out.stdout.strip()}")
        return out.stdout

    # -- status --------------------------------------------------------------
    def status(self) -> WifiStatus:
        try:
            dev_line = self._run("-t", "-f", "DEVICE,TYPE,STATE,CONNECTION",
                                 "device", "status")
        except Exception:
            return WifiStatus(present=False, connected=False)
        present = False
        connected = False
        ssid: Optional[str] = None
        for line in dev_line.splitlines():
            fields = line.split(":")
            if len(fields) < 4 or fields[1] != "wifi":
                continue
            if fields[0] == self.iface or not present:
                present = True
                if fields[2] == "connected":
                    connected = True
                    ssid = fields[3].strip() or None
        ip = None
        if connected:
            try:
                ip_line = self._run("-t", "-f", "IP4.ADDRESS", "device",
                                    "show", self.iface)
                for line in ip_line.splitlines():
                    if line.startswith("IP4.ADDRESS"):
                        ip = line.split(":", 1)[1].split("/")[0].strip()
                        break
            except Exception:
                pass
        return WifiStatus(present=present, connected=connected,
                          ssid=ssid, iface=self.iface, ip=ip)

    # -- scan ------------------------------------------------------------------
    def scan(self) -> List[WifiNetwork]:
        try:
            self._run("device", "wifi", "rescan", "ifname", self.iface,
                      timeout=10.0)
        except Exception:
            pass
        try:
            out = self._run("-t", "-f", "SSID,SIGNAL,SECURITY",
                            "device", "wifi", "list", "ifname", self.iface)
        except Exception:
            return []
        nets: List[WifiNetwork] = []
        for line in out.splitlines():
            fields = line.split(":")
            if len(fields) >= 3 and fields[0]:
                try:
                    sig = int(fields[1])
                except ValueError:
                    sig = 0
                nets.append(WifiNetwork(ssid=fields[0], signal=sig,
                                        security=fields[2]))
        return nets

    # -- connect -----------------------------------------------------------------
    def connect(self, ssid: str, psk: str, *,
                hidden: bool = False) -> WifiStatus:
        """Create/activate a WPA-PSK connection for ``ssid``."""
        name = f"thoth-{ssid_hash(ssid)}"
        try:
            self._run("connection", "delete", name)
        except Exception:
            pass
        args = ["connection", "add", "type", "wifi",
                "ifname", self.iface, "con-name", name,
                "ssid", ssid]
        self._run(*args, timeout=20.0)
        if psk:
            self._run("connection", "modify", name,
                      "wifi-sec.key-mgmt", "wpa-psk",
                      "wifi-sec.psk", psk)
        else:
            self._run("connection", "modify", name,
                      "wifi-sec.key-mgmt", "")
        if hidden:
            try:
                self._run("connection", "modify", name,
                          "802-11-wireless.hidden", "yes")
            except Exception:
                pass
        self._run("connection", "up", name, timeout=45.0)
        return self.status()

    def disconnect(self) -> None:
        try:
            self._run("device", "disconnect", self.iface)
        except Exception:
            pass

    # -- AP mode (Phase 4 fallback) ---------------------------------------------
    def ap_start(self, ssid: str, psk: str) -> bool:
        """Broadcast an AP via NM's hotspot support."""
        try:
            self._run("device", "wifi", "hotspot", "ifname", self.iface,
                      "con-name", "thoth-ap", "ssid", ssid,
                      "password", psk, timeout=30.0)
            return True
        except Exception as exc:
            logger.debug("ap_start failed: %s", exc)
            return False

    def ap_stop(self) -> None:
        try:
            self._run("connection", "down", "thoth-ap")
            self._run("connection", "delete", "thoth-ap")
        except Exception:
            pass


class NullWifiManager:
    """No Wi-Fi hardware — provisioning reports 'no iface'."""

    iface = DEFAULT_IFACE

    @staticmethod
    def available() -> bool:
        return False

    def status(self) -> WifiStatus:
        return WifiStatus(present=False, connected=False)

    def scan(self) -> List[WifiNetwork]:
        return []

    def connect(self, ssid, psk, *, hidden=False) -> WifiStatus:
        raise RuntimeError("no wifi backend")

    def disconnect(self) -> None:
        return None

    def ap_start(self, ssid, psk) -> bool:
        return False

    def ap_stop(self) -> None:
        return None


def default_wifi_manager(iface: str = DEFAULT_IFACE):
    if NmcliWifiManager.available():
        return NmcliWifiManager(iface)
    return NullWifiManager()


__all__ = [
    "DEFAULT_IFACE", "NmcliWifiManager", "NullWifiManager",
    "WifiNetwork", "WifiStatus", "default_wifi_manager", "ssid_hash",
]
