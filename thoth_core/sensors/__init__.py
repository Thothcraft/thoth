"""Thoth Core Sensors — base classes and sensor manager.

Platform-specific sensor implementations live in each platform package
(thoth_mac/sensors, thoth_win/sensors, thoth_rpi/sensors).
"""

from .base import (
    BaseSensor,
    SensorRegistry,
    SensorType,
    SensorStatus,
    SensorConfig,
    CollectionSession,
)
from .manager import SensorManager

__all__ = [
    "BaseSensor",
    "SensorRegistry",
    "SensorType",
    "SensorStatus",
    "SensorConfig",
    "CollectionSession",
    "SensorManager",
]
