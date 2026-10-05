"""Bluetooth subsystem — BlueZ observer/central/peripheral roles.

See ``docs/observation-contract-v1.md`` §5 for the emitted schemas and
``docs/bluetooth-architecture.md`` for the role/duty model.
"""

from .backend import (Advertisement, BackendUnavailable, BleakBackend,
                      CentralHandle, NullBackend, default_backend)
from .known import KnownDeviceStore
from .subsystem import BluetoothSubsystem

__all__ = [
    "Advertisement", "BackendUnavailable", "BleakBackend",
    "BluetoothSubsystem", "CentralHandle", "KnownDeviceStore",
    "NullBackend", "default_backend",
]
