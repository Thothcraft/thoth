"""Data models for Thoth sensor readings and system state."""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Optional, Any
import json

@dataclass
class IMUReading:
    """IMU sensor reading data structure."""
    pitch: float
    roll: float
    yaw: float

@dataclass
class AccelerometerReading:
    """Accelerometer reading data structure."""
    x: float
    y: float
    z: float

@dataclass
class GyroscopeReading:
    """Gyroscope reading data structure."""
    x: float
    y: float
    z: float

@dataclass
class MagnetometerReading:
    """Magnetometer reading data structure."""
    x: float
    y: float
    z: float

@dataclass
class SensorReading:
    """Complete sensor reading with all IMU data."""
    timestamp: str
    imu: IMUReading
    accel: AccelerometerReading
    gyro: GyroscopeReading
    mag: MagnetometerReading
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'imu': asdict(self.imu),
            'accel': asdict(self.accel),
            'gyro': asdict(self.gyro),
            'mag': asdict(self.mag)
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensorReading':
        """Create SensorReading from dictionary."""
        return cls(
            timestamp=data['timestamp'],
            imu=IMUReading(**data['imu']),
            accel=AccelerometerReading(**data['accel']),
            gyro=GyroscopeReading(**data['gyro']),
            mag=MagnetometerReading(**data['mag'])
        )

@dataclass
class SystemStatus:
    """System status information."""
    status: str
    battery_level: Optional[int] = None
    wifi_connected: bool = False
    ap_mode: bool = False
    collection_active: bool = False
    uptime: Optional[str] = None
    temperature: Optional[float] = None
    disk_usage: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

@dataclass
class ButtonConfig:
    """Button configuration for PiSugar."""
    single: str = "toggle_collection"
    double: str = "start_ap"
    long: str = "shutdown"
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'ButtonConfig':
        """Create ButtonConfig from dictionary."""
        return cls(**data)

@dataclass
class UploadResult:
    """Result of data upload operation."""
    success: bool
    uploaded_count: int
    error_message: Optional[str] = None
    upload_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
