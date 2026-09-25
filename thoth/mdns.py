"""Best-effort mDNS advertisement for the local dashboard/API.

Linux/Pi nodes get ``<hostname>.local`` from Avahi; Windows has no
mDNS responder, so nodes like ``thoth-denver`` are unreachable by
name without this. We register ``_thoth._tcp`` + ``_http._tcp``
services whose ``server=`` records publish the ``<name>.local``
address, so ``http://<name>.local`` resolves on any node.

Soft dependency: missing ``zeroconf`` just disables advertisement.
"""
from __future__ import annotations

import logging
import re
import socket
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_SERVICE_TYPES = ("_thoth._tcp.local.", "_http._tcp.local.")


def _sanitise(name: str) -> str:
    """hostname-safe mDNS label: lowercase, [a-z0-9-], no edge dashes."""
    label = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return label[:63]


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
            addrs = {
                ai[4][0]
                for ai in socket.getaddrinfo(socket.gethostname(), None)
                if "." in ai[4][0]  # IPv4 only; v6 link-local is noisy
            }
            if not addrs:
                addrs = {socket.gethostbyname(socket.gethostname())}
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
