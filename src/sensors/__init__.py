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

__all__ = [
    "BaseSensor",
    "SensorRegistry",
    "SensorType",
    "SensorStatus",
    "SensorConfig",
    "CollectionSession",
]

# Lazy imports to avoid circular dependency issues
def get_sensor_manager():
    from .manager import SensorManager
    return SensorManager

def get_imu_sensor():
    from .imu import IMUSensor
    return IMUSensor

def get_csi_sensor():
    from .csi import CSISensor
    return CSISensor
