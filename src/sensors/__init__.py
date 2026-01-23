"""Modular Sensor Module for Thoth Device.

This module provides a unified, extensible interface for sensor data collection
on the Thoth device. It supports multiple sensor types and integrates with
the data management module for automatic metadata generation.

Supported Sensors:
- IMU (Accelerometer, Gyroscope, Magnetometer)
- WiFi CSI (Channel State Information)
- Camera (Video/Image)
- Microphone (Audio)
- Radar (mmWave)
- GPS

Extensibility:
To add a new sensor, create a class that inherits from BaseSensor and
register it with the SensorRegistry.

Usage:
    from thoth.src.sensors import SensorManager, SensorType
    
    # Initialize sensor manager
    manager = SensorManager()
    
    # Discover available sensors
    sensors = manager.discover_sensors()
    
    # Start collection
    manager.start_collection("imu", duration=60, labels={"activity": "walking"})
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
from .imu import IMUSensor
from .csi import CSISensor

__all__ = [
    "BaseSensor",
    "SensorRegistry",
    "SensorType",
    "SensorStatus",
    "SensorConfig",
    "CollectionSession",
    "SensorManager",
    "IMUSensor",
    "CSISensor",
]
