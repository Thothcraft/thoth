"""Best-effort mDNS advertisement for the local dashboard/API.

Linux/Pi nodes get ``<hostname>.local`` from Avahi; Windows has no
mDNS responder, so nodes like ``thoth-denver`` are unreachable by
name without this. We register ``_thoth._tcp`` + ``_http._tcp``
services whose ``server=`` records point at ``<name>.local``.

When a system responder (Avahi) already owns the hostname we publish
**only** the service records (PTR/SRV/TXT) — no A record — so we never
fight it for the name and the hostname survives daemon restarts. On
hosts with no responder we publish the primary LAN address ourselves.

Soft dependency: missing ``zeroconf`` just disables advertisement.
"""
from __future__ import annotations

import logging
import os
import re
import socket
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

_SERVICE_TYPES = ("_thoth._tcp.local.", "_http._tcp.local.")


def _sanitise(name: str) -> str:
    """hostname-safe mDNS label: lowercase, [a-z0-9-], no edge dashes."""
    label = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return label[:63]


def _system_responder_active() -> bool:
    """A system mDNS responder already publishes ``<hostname>.local``.

    Avahi keeps its socket at ``/run/avahi-daemon/socket`` while running;
    systemd-resolved mDNS is rarer and skipped. When present, publishing
    our own A record for the hostname conflicts (Avahi yields and renames
    itself to ``<host>-2``, and our records die with the daemon).
    """
    return os.path.exists("/run/avahi-daemon/socket")


def _lan_addresses() -> List[str]:
    """Primary LAN IPv4(s) — never loopback/link-local.

    ``getaddrinfo(hostname)`` is useless for this: Debian/cloud-init maps
    the hostname to ``127.0.1.1`` in /etc/hosts. A UDP "connect" to an
    unroutable address reveals the primary outbound interface instead.
    """
    addrs: List[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))  # no traffic is sent
        addrs.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    for ai in socket.getaddrinfo(socket.gethostname(), None):
        a = ai[4][0]
        if "." in a and not a.startswith(("127.", "169.254.")):
            addrs.append(a)
    out = list(dict.fromkeys(addrs))
    if not out:
        try:
            fallback = socket.gethostbyname(socket.gethostname())
            if not fallback.startswith("127."):
                out.append(fallback)
        except OSError:
            pass
    return out


class MdnsAdvertiser:
    def __init__(self) -> None:
        self._zc = None
        self._infos: list = []

    def start(self, names: Iterable[str], port: int,
              device_id: str = "") -> Optional["MdnsAdvertiser"]:
        try:
            from zeroconf import ServiceInfo, Zeroconf
        except ImportError:
            logger.debug("zeroconf not installed — mDNS advertisement off")
            return None
        try:
            own_addr = not _system_responder_active()
            addrs = _lan_addresses() if own_addr else []
            if own_addr and not addrs:
                logger.debug("mDNS: no LAN address found — skipping")
                return None
            props = {b"id": device_id.encode(), b"path": b"/"}
            self._zc = Zeroconf()
            for raw in names:
                label = _sanitise(raw)
                if not label:
                    continue
                for stype in _SERVICE_TYPES:
                    info = ServiceInfo(
                        stype,
                        f"{label}.{stype}",
                        addresses=[socket.inet_aton(a) for a in addrs],
                        port=port,
                        properties=props,
                        server=f"{label}.local.",
                    )
                    try:
                        self._zc.register_service(info)
                        self._infos.append(info)
                    except Exception as exc:
                        logger.debug("mDNS register %s failed: %s",
                                     info.name, exc)
            if self._infos:
                logger.info("mDNS: advertising %s on port %d",
                            ", ".join(
                                i.server for i in self._infos
                                if i.type == _SERVICE_TYPES[0]),
                            port)
            return self
        except Exception as exc:
            logger.debug("mDNS advertisement disabled: %s", exc)
            self.close()
            return None

    def close(self) -> None:
        if self._zc is not None:
            try:
                for info in self._infos:
                    self._zc.unregister_service(info)
            except Exception:
                pass
            try:
                self._zc.close()
            except Exception:
                pass
        self._zc = None
        self._infos = []
