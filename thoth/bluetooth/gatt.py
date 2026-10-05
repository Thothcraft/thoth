"""GATT server for BLE commissioning — BlueZ peripheral role via D-Bus.

BlueZ's ``GattManager1`` expects a D-Bus object tree: a root
implementing ``ObjectManager``, ``GattService1`` objects, and
``GattCharacteristic1`` objects with ``ReadValue``/``WriteValue``/
``StartNotify``. Pairing it with ``LEAdvertisingManager1`` makes the
node discoverable while unprovisioned — the phone writes credentials,
the daemon answers notifications over a status characteristic.

Import-guarded: ``dbus_next`` is a hardware dep; the module loads
everywhere and raises :class:`BackendUnavailable` only on use.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from .backend import BackendUnavailable

logger = logging.getLogger(__name__)

BLUEZ = "org.bluez"
GATT_MANAGER = "org.bluez.GattManager1"
GATT_SERVICE = "org.bluez.GattService1"
GATT_CHAR = "org.bluez.GattCharacteristic1"
LE_ADV_MANAGER = "org.bluez.LEAdvertisingManager1"
LE_ADVERTISEMENT = "org.bluez.LEAdvertisement1"
OBJECT_MANAGER = "org.freedesktop.DBus.ObjectManager"

# Thoth commissioning service — 128-bit vendor space
SVC_UUID = "a0630100-01c4-4d1e-9b2f-0011778899aa"
CHAR_CONFIGURE = "a0630101-01c4-4d1e-9b2f-0011778899aa"  # write: {ssid,psk}
CHAR_STATUS = "a0630102-01c4-4d1e-9b2f-0011778899aa"    # read+notify: state
CHAR_SCAN = "a0630103-01c4-4d1e-9b2f-0011778899aa"      # read: network list
CHAR_DEVICE_ID = "a0630104-01c4-4d1e-9b2f-0011778899aa"  # read: device id


def _require_dbus_next():
    try:
        from dbus_next.aio import MessageBus              # noqa: F401
        from dbus_next import BusType, Variant            # noqa: F401
        from dbus_next.service import (ServiceInterface,  # noqa: F401
                                       dbus_property, method, signal)
        return True
    except Exception as exc:
        raise BackendUnavailable(f"dbus_next not importable: {exc}")


class _CharacteristicMixin:
    """Shared plumbing for GattCharacteristic1 objects."""

    def _dbus_value(self, payload: Any) -> Any:
        from dbus_next import Variant
        return Variant("ay", bytes(json.dumps(payload).encode()))


class GattApplication:
    """Object tree + registration. Constructed on the BLE asyncio loop
    (``await app.register()``); callbacks stay synchronous callables so
    the provisioning manager stays thread-agnostic."""

    def __init__(self, name: str, device_id: str,
                 on_configure: Callable[[Dict[str, Any]], None],
                 on_scan: Optional[Callable[[], List[Dict[str, Any]]]] = None,
                 get_status: Optional[Callable[[], Dict[str, Any]]] = None):
        _require_dbus_next()
        self.name = name
        self.device_id = device_id
        self.on_configure = on_configure
        self.on_scan = on_scan
        self.get_status = get_status or (lambda: {"state": "idle"})
        self._bus = None
        self._status_char = None
        self._adv_path = "/org/thoth/prov/adv"

    # -- D-Bus objects ----------------------------------------------------------
    def _build(self):
        from dbus_next.service import (ServiceInterface, dbus_property,
                                       method, signal)
        from dbus_next import Variant

        app = self

        class ObjectManagerImpl(ServiceInterface):
            def __init__(self):
                super().__init__(OBJECT_MANAGER)

            @method()
            def GetManagedObjects(self) -> "a{oa{sa{sv}}}":
                return app._managed_objects()

        class Service(ServiceInterface):
            def __init__(self, path, uuid, primary=True):
                super().__init__(GATT_SERVICE)
                self.path, self.uuid, self.primary = path, uuid, primary

            @dbus_property()
            def UUID(self) -> "s":
                return self.uuid

            @dbus_property()
            def Primary(self) -> "b":
                return self.primary

            @dbus_property()
            def Includes(self) -> "ao":
                return []

        class Char(ServiceInterface, _CharacteristicMixin):
            def __init__(self, path, uuid, service_path, flags,
                         read_fn=None, write_fn=None, notify=False):
                super().__init__(GATT_CHAR)
                self.path = path
                self.uuid = uuid
                self.service_path = service_path
                self._flags = flags
                self._read = read_fn
                self._write = write_fn
                self._notify = notify
                self._notifying = False

            @dbus_property()
            def UUID(self) -> "s":
                return self.uuid

            @dbus_property()
            def Service(self) -> "o":
                return self.service_path

            @dbus_property()
            def Flags(self) -> "as":
                return self._flags

            @dbus_property()
            def NotifyAcquired(self) -> "b":
                return self._notifying

            @method()
            def ReadValue(self, options: "a{sv}") -> "ay":
                if self._read is None:
                    return b""
                return bytes(self._read())

            @method()
            def WriteValue(self, value: "ay", options: "a{sv}"):
                if self._write is not None:
                    self._write(bytes(value))

            @method()
            def StartNotify(self):
                if self._notify:
                    self._notifying = True

            @method()
            def StopNotify(self):
                self._notifying = False

            @signal()
            def PropertiesChanged(self, iface: "s", changed: "a{sv}",
                                  invalidated: "as"):
                return [iface, changed, invalidated]

        class Advertisement(ServiceInterface):
            def __init__(self):
                super().__init__(LE_ADVERTISEMENT)

            @dbus_property()
            def Type(self) -> "s":
                return "peripheral"

            @dbus_property()
            def ServiceUUIDs(self) -> "as":
                return [SVC_UUID]

            @dbus_property()
            def LocalName(self) -> "s":
                return app.name

            @dbus_property()
            def Includes(self) -> "as":
                return ["tx-power"]

            @method()
            def Release(self):
                pass

        self._obj_manager_cls = ObjectManagerImpl
        self._service_cls = Service
        self._char_cls = Char
        self._adv_cls = Advertisement

    def _managed_objects(self) -> Dict[str, Any]:
        from dbus_next import Variant
        svc_path = "/org/thoth/prov/svc0"
        chars = {
            f"{svc_path}/char0": {
                "UUID": Variant("s", CHAR_CONFIGURE),
                "Service": Variant("o", svc_path),
                "Flags": Variant("as", ["write"]),
            },
            f"{svc_path}/char1": {
                "UUID": Variant("s", CHAR_STATUS),
                "Service": Variant("o", svc_path),
                "Flags": Variant("as", ["read", "notify"]),
            },
            f"{svc_path}/char2": {
                "UUID": Variant("s", CHAR_SCAN),
                "Service": Variant("o", svc_path),
                "Flags": Variant("as", ["read"]),
            },
            f"{svc_path}/char3": {
                "UUID": Variant("s", CHAR_DEVICE_ID),
                "Service": Variant("o", svc_path),
                "Flags": Variant("as", ["read"]),
            },
        }
        objs: Dict[str, Any] = {
            svc_path: {GATT_SERVICE: {
                "UUID": Variant("s", SVC_UUID),
                "Primary": Variant("b", True),
                "Includes": Variant("ao", []),
            }},
            self._adv_path: {LE_ADVERTISEMENT: {
                "Type": Variant("s", "peripheral"),
                "ServiceUUIDs": Variant("as", [SVC_UUID]),
                "LocalName": Variant("s", self.name),
                "Includes": Variant("as", ["tx-power"]),
            }},
        }
        for path, props in chars.items():
            objs[path] = {GATT_CHAR: props}
        return objs

    # -- lifecycle --------------------------------------------------------------
    async def register(self, adapter: str = "hci0") -> None:
        """Export objects, register the app + advertisement with BlueZ."""
        from dbus_next.aio import MessageBus
        from dbus_next import BusType, Message

        self._build()
        self._bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
        svc_path = "/org/thoth/prov/svc0"
        root = "/org/thoth/prov"
        bus = self._bus
        bus.export(root, self._obj_manager_cls())
        bus.export(svc_path, self._service_cls(svc_path, SVC_UUID))

        status_char = self._char_cls(
            f"{svc_path}/char1", CHAR_STATUS, svc_path,
            ["read", "notify"],
            read_fn=lambda: json.dumps(self.get_status()).encode(),
            notify=True)
        self._status_char = status_char
        bus.export(f"{svc_path}/char0", self._char_cls(
            f"{svc_path}/char0", CHAR_CONFIGURE, svc_path, ["write"],
            write_fn=self._on_configure_write))
        bus.export(f"{svc_path}/char1", status_char)
        bus.export(f"{svc_path}/char2", self._char_cls(
            f"{svc_path}/char2", CHAR_SCAN, svc_path, ["read"],
            read_fn=lambda: json.dumps(
                self.on_scan() if self.on_scan else []).encode()))
        bus.export(f"{svc_path}/char3", self._char_cls(
            f"{svc_path}/char3", CHAR_DEVICE_ID, svc_path, ["read"],
            read_fn=lambda: self.device_id.encode()))
        bus.export(self._adv_path, self._adv_cls())

        introspect = await bus.introspect(BLUEZ, f"/org/bluez/{adapter}")
        del introspect  # adapter must exist — failure surfaces below
        reply = await bus.call(
            __import__("dbus_next").Message(
                destination=BLUEZ,
                path=f"/org/bluez/{adapter}",
                interface=GATT_MANAGER,
                member="RegisterApplication",
                signature="oa{sv}",
                body=[root, {}]))
        if reply.message_type.name == "ERROR":
            raise BackendUnavailable(f"GattManager1: {reply.body}")
        reply = await bus.call(
            __import__("dbus_next").Message(
                destination=BLUEZ,
                path=f"/org/bluez/{adapter}",
                interface=LE_ADV_MANAGER,
                member="RegisterAdvertisement",
                signature="oa{sv}",
                body=[self._adv_path, {}]))
        if reply.message_type.name == "ERROR":
            logger.debug("RegisterAdvertisement failed: %s", reply.body)
        logger.info("ble peripheral: commissioning service advertising")

    def _on_configure_write(self, raw: bytes) -> None:
        try:
            payload = json.loads(raw.decode())
        except Exception:
            payload = {}
        try:
            self.on_configure(payload)
        except Exception as exc:
            logger.debug("on_configure failed: %s", exc)

    def notify_status(self) -> None:
        """Push a status-characteristic notification to centrals."""
        char = self._status_char
        if char is not None and char._notifying:
            try:
                from dbus_next import Variant
                char.PropertiesChanged(
                    GATT_CHAR,
                    {"Value": Variant("ay", json.dumps(
                        self.get_status()).encode())}, [])
            except Exception:
                pass

    async def unregister(self) -> None:
        if self._bus is not None:
            try:
                from dbus_next import Message
                await self._bus.call(Message(
                    destination=BLUEZ, path="/org/bluez/hci0",
                    interface=GATT_MANAGER, member="UnregisterApplication",
                    signature="o", body=["/org/thoth/prov"]))
            except Exception:
                pass
            try:
                self._bus.disconnect()
            except Exception:
                pass
            self._bus = None


__all__ = [
    "CHAR_CONFIGURE", "CHAR_DEVICE_ID", "CHAR_SCAN", "CHAR_STATUS",
    "GattApplication", "SVC_UUID",
]
