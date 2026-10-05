"""BlueZ backend for the BLE subsystem — bleak (observer + central)
with dbus-next reserved for the peripheral/GATT role (Phase 3).

Everything here is import-guarded: ``bleak``/``dbus_next`` are optional
hardware deps, so the module loads on any platform and reports
``available() == False`` instead of raising.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class BackendUnavailable(RuntimeError):
    """Raised when the requested BLE backend role can't run here."""


@dataclass
class Advertisement:
    """Normalized BLE advertisement (backend-agnostic)."""
    address: str                                  # "AA:BB:.." — local only
    rssi: int
    ts: float
    tx_power: Optional[int] = None
    addr_type: str = "anonymous"                  # public|random|anonymous
    name: Optional[str] = None
    service_uuids: List[str] = field(default_factory=list)


@dataclass
class CentralHandle:
    """A live central-role connection (backend-agnostic)."""
    client: Any
    address: str

    @property
    def connected(self) -> bool:
        return bool(getattr(self.client, "is_connected", False))


def _bleak_importable() -> bool:
    try:
        import bleak  # noqa: F401
        return True
    except Exception:
        return False


def _dbus_next_importable() -> bool:
    try:
        import dbus_next  # noqa: F401
        return True
    except Exception:
        return False


def _bluez_tools() -> bool:
    return shutil.which("bluetoothctl") is not None or \
        shutil.which("btmgmt") is not None


class BleakBackend:
    """Real BlueZ access via bleak.

    ``adapter`` selects the host controller ("hci0"…); None lets BlueZ
    pick. All methods are async — the subsystem owns the loop.
    """

    def __init__(self, adapter: Optional[str] = None):
        self._adapter = adapter
        self._scanner: Any = None

    # -- presence --------------------------------------------------------------
    @staticmethod
    def available() -> bool:
        return _bleak_importable() and _bluez_tools()

    @staticmethod
    def peripheral_supported() -> bool:
        # GATT-server role needs raw D-Bus, bleak doesn't expose it.
        return _dbus_next_importable()

    @staticmethod
    def adapter_name(default: str = "hci0") -> str:
        return default

    # -- observer ---------------------------------------------------------------
    async def scan_start(self, on_adv) -> None:
        if not self.available():
            raise BackendUnavailable("bleak/BlueZ not available")
        from bleak import BleakScanner

        def _cb(device, adv_data) -> None:
            props = (device.details or {}).get("props", {}) \
                if isinstance(device.details, dict) else {}
            addr_type = props.get("AddressType") or "anonymous"
            adv = Advertisement(
                address=str(device.address),
                rssi=int(getattr(adv_data, "rssi", 0) or 0),
                ts=asyncio.get_event_loop().time(),
                tx_power=getattr(adv_data, "tx_power", None),
                addr_type=str(addr_type).lower()
                    if str(addr_type).lower() in ("public", "random")
                    else "anonymous",
                name=adv_data.local_name or device.name or None,
                service_uuids=list(adv_data.service_uuids or []),
            )
            on_adv(adv)

        kwargs: Dict[str, Any] = {"detection_callback": _cb}
        if self._adapter:
            kwargs["adapter"] = self._adapter
        self._scanner = BleakScanner(**kwargs)
        await self._scanner.start()

    async def scan_stop(self) -> None:
        if self._scanner is not None:
            try:
                await self._scanner.stop()
            except Exception:
                pass
            self._scanner = None

    # -- central ------------------------------------------------------------------
    async def connect(self, address: str) -> CentralHandle:
        """Central role — one BleakClient to a peripheral."""
        if not self.available():
            raise BackendUnavailable("bleak/BlueZ not available")
        from bleak import BleakClient
        kwargs: Dict[str, Any] = {}
        if self._adapter:
            kwargs["adapter"] = self._adapter
        client = BleakClient(address, **kwargs)
        await client.connect()
        return CentralHandle(client=client, address=address)

    async def subscribe(self, handle: CentralHandle, char_uuid: str,
                        on_data) -> None:
        await handle.client.start_notify(
            char_uuid,
            lambda _c, data: on_data(bytes(data)))

    async def write_char(self, handle: CentralHandle, char_uuid: str,
                         data: bytes) -> None:
        await handle.client.write_gatt_char(char_uuid, bytes(data))

    async def disconnect(self, handle: Optional[CentralHandle]) -> None:
        if handle is not None:
            try:
                await handle.client.disconnect()
            except Exception:
                pass


class NullBackend:
    """No BLE here — used on non-BlueZ hosts and in tests."""

    @staticmethod
    def available() -> bool:
        return False

    @staticmethod
    def peripheral_supported() -> bool:
        return False

    @staticmethod
    def adapter_name(default: str = "hci0") -> str:
        return default

    async def scan_start(self, on_adv) -> None:
        raise BackendUnavailable("no BLE backend")

    async def scan_stop(self) -> None:
        return None

    async def connect(self, address: str) -> CentralHandle:
        raise BackendUnavailable("no BLE backend")

    async def subscribe(self, handle, char_uuid, on_data) -> None:
        raise BackendUnavailable("no BLE backend")

    async def write_char(self, handle, char_uuid, data) -> None:
        raise BackendUnavailable("no BLE backend")

    async def disconnect(self, handle) -> None:
        return None


def default_backend(adapter: Optional[str] = None):
    """Pick the real backend when BLE is present, else the null one."""
    if BleakBackend.available():
        return BleakBackend(adapter)
    return NullBackend()


__all__ = [
    "Advertisement", "BackendUnavailable", "BleakBackend",
    "CentralHandle", "NullBackend", "default_backend",
]
