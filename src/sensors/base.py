"""Base classes and registry for sensors.

This module defines the base sensor class and registry pattern for
extensible sensor support.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Type, Callable
import numpy as np

logger = logging.getLogger(__name__)


class SensorType(str, Enum):
    """Supported sensor types."""
    IMU = "imu"
    CSI = "csi"
    CAMERA = "camera"
    MICROPHONE = "mic"
    RADAR = "radar"
    GPS = "gps"
    CUSTOM = "custom"


class SensorStatus(str, Enum):
    """Sensor status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    COLLECTING = "collecting"
    ERROR = "error"
    INITIALIZING = "initializing"


@dataclass
class SensorConfig:
    """Configuration for a sensor."""
    sensor_type: SensorType
    sample_rate_hz: float = 100.0
    buffer_size: int = 1000
    auto_save: bool = True
    save_format: str = "npy"
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_type": self.sensor_type.value,
            "sample_rate_hz": self.sample_rate_hz,
            "buffer_size": self.buffer_size,
            "auto_save": self.auto_save,
            "save_format": self.save_format,
            "custom_params": self.custom_params,
        }


@dataclass
class CollectionSession:
    """Represents an active data collection session."""
    session_id: str
    sensor_type: SensorType
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    num_samples: int = 0
    labels: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    error: Optional[str] = None
    data_file: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "sensor_type": self.sensor_type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "num_samples": self.num_samples,
            "labels": self.labels,
            "status": self.status,
            "error": self.error,
            "data_file": self.data_file,
        }


class BaseSensor(ABC):
    """Base class for all sensors.
    
    To implement a new sensor:
    1. Inherit from BaseSensor
    2. Implement the abstract methods
    3. Register with SensorRegistry
    
    Example:
        @SensorRegistry.register
        class MySensor(BaseSensor):
            sensor_type = SensorType.CUSTOM
            sensor_name = "My Custom Sensor"
            
            def initialize(self) -> bool:
                # Initialize hardware
                return True
            
            def read_sample(self) -> Optional[np.ndarray]:
                # Read one sample
                return np.array([...])
            
            def cleanup(self):
                # Cleanup resources
                pass
    """
    
    # Class-level metadata (override in subclasses)
    sensor_type: SensorType = SensorType.CUSTOM
    sensor_name: str = "Base Sensor"
    sensor_description: str = "Base sensor class"
    supported_platforms: List[str] = ["all"]
    default_sample_rate: float = 100.0
    data_channels: int = 1
    data_dtype: np.dtype = np.float32
    
    def __init__(self, config: Optional[SensorConfig] = None):
        """Initialize the sensor.
        
        Args:
            config: Sensor configuration
        """
        self.config = config or SensorConfig(sensor_type=self.sensor_type)
        self.status = SensorStatus.INITIALIZING
        self._buffer: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._collecting = False
        self._collection_thread: Optional[threading.Thread] = None
        self._current_session: Optional[CollectionSession] = None
        self._callbacks: List[Callable[[np.ndarray], None]] = []
        self._error_message: Optional[str] = None
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the sensor hardware.
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def read_sample(self) -> Optional[np.ndarray]:
        """Read a single sample from the sensor.
        
        Returns:
            NumPy array with sensor data, or None if read failed
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup sensor resources."""
        pass
    
    def is_available(self) -> bool:
        """Check if sensor is available on this platform.
        
        Returns:
            True if sensor is available
        """
        try:
            return self.initialize()
        except Exception as e:
            logger.debug(f"Sensor {self.sensor_name} not available: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get sensor information.
        
        Returns:
            Dictionary with sensor info
        """
        return {
            "sensor_type": self.sensor_type.value,
            "name": self.sensor_name,
            "description": self.sensor_description,
            "status": self.status.value,
            "sample_rate_hz": self.config.sample_rate_hz,
            "data_channels": self.data_channels,
            "supported_platforms": self.supported_platforms,
            "error": self._error_message,
        }
    
    def start_collection(
        self,
        duration_seconds: Optional[float] = None,
        labels: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> CollectionSession:
        """Start data collection.
        
        Args:
            duration_seconds: Collection duration (None for continuous)
            labels: Labels for the collected data
            callback: Optional callback for each sample
        
        Returns:
            CollectionSession instance
        """
        if self._collecting:
            raise RuntimeError("Collection already in progress")
        
        import uuid
        session_id = str(uuid.uuid4())[:8]
        
        self._current_session = CollectionSession(
            session_id=session_id,
            sensor_type=self.sensor_type,
            start_time=datetime.utcnow(),
            labels=labels or {},
        )
        
        if callback:
            self._callbacks.append(callback)
        
        self._collecting = True
        self._buffer = []
        self.status = SensorStatus.COLLECTING
        
        # Start collection thread
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(duration_seconds,),
            daemon=True
        )
        self._collection_thread.start()
        
        logger.info(f"Started collection session {session_id} for {self.sensor_name}")
        return self._current_session
    
    def stop_collection(self) -> Optional[CollectionSession]:
        """Stop data collection.
        
        Returns:
            Completed CollectionSession
        """
        if not self._collecting:
            return None
        
        self._collecting = False
        
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        
        if self._current_session:
            self._current_session.end_time = datetime.utcnow()
            self._current_session.duration_seconds = (
                self._current_session.end_time - self._current_session.start_time
            ).total_seconds()
            self._current_session.status = "completed"
        
        self.status = SensorStatus.AVAILABLE
        self._callbacks = []
        
        session = self._current_session
        self._current_session = None
        
        logger.info(f"Stopped collection: {session.num_samples} samples collected")
        return session
    
    def get_buffer(self) -> np.ndarray:
        """Get collected data from buffer.
        
        Returns:
            NumPy array of collected samples
        """
        with self._lock:
            if not self._buffer:
                return np.array([])
            return np.stack(self._buffer, axis=0)
    
    def clear_buffer(self):
        """Clear the data buffer."""
        with self._lock:
            self._buffer = []
    
    def _collection_loop(self, duration_seconds: Optional[float]):
        """Main collection loop (runs in thread)."""
        start_time = time.time()
        sample_interval = 1.0 / self.config.sample_rate_hz
        
        try:
            while self._collecting:
                # Check duration
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    break
                
                # Read sample
                sample = self.read_sample()
                
                if sample is not None:
                    with self._lock:
                        self._buffer.append(sample)
                        if self._current_session:
                            self._current_session.num_samples += 1
                    
                    # Call callbacks
                    for callback in self._callbacks:
                        try:
                            callback(sample)
                        except Exception as e:
                            logger.warning(f"Callback error: {e}")
                
                # Sleep to maintain sample rate
                time.sleep(sample_interval)
        
        except Exception as e:
            logger.error(f"Collection error: {e}")
            self._error_message = str(e)
            self.status = SensorStatus.ERROR
            if self._current_session:
                self._current_session.status = "error"
                self._current_session.error = str(e)
        
        finally:
            self._collecting = False


class SensorRegistry:
    """Registry for sensor implementations."""
    
    _sensors: Dict[SensorType, Type[BaseSensor]] = {}
    
    @classmethod
    def register(cls, sensor_class: Type[BaseSensor]) -> Type[BaseSensor]:
        """Register a sensor class (decorator).
        
        Usage:
            @SensorRegistry.register
            class MySensor(BaseSensor):
                ...
        """
        sensor_type = sensor_class.sensor_type
        cls._sensors[sensor_type] = sensor_class
        logger.debug(f"Registered sensor: {sensor_class.sensor_name} ({sensor_type.value})")
        return sensor_class
    
    @classmethod
    def get(cls, sensor_type: SensorType) -> Optional[Type[BaseSensor]]:
        """Get a sensor class by type."""
        return cls._sensors.get(sensor_type)
    
    @classmethod
    def create(cls, sensor_type: SensorType, config: Optional[SensorConfig] = None) -> Optional[BaseSensor]:
        """Create a sensor instance."""
        sensor_class = cls.get(sensor_type)
        if sensor_class:
            return sensor_class(config)
        return None
    
    @classmethod
    def list_sensors(cls) -> List[Dict[str, Any]]:
        """List all registered sensors."""
        return [
            {
                "type": sensor_type.value,
                "name": sensor_class.sensor_name,
                "description": sensor_class.sensor_description,
            }
            for sensor_type, sensor_class in cls._sensors.items()
        ]
    
    @classmethod
    def discover_available(cls) -> List[BaseSensor]:
        """Discover available sensors on this platform."""
        available = []
        for sensor_type, sensor_class in cls._sensors.items():
            try:
                sensor = sensor_class()
                if sensor.is_available():
                    sensor.status = SensorStatus.AVAILABLE
                    available.append(sensor)
                else:
                    sensor.cleanup()
            except Exception as e:
                logger.debug(f"Sensor {sensor_class.sensor_name} not available: {e}")
        return available
