"""IMU Sensor Implementation.

This module provides IMU (Inertial Measurement Unit) sensor support for:
- Accelerometer (3-axis)
- Gyroscope (3-axis)
- Magnetometer (3-axis, if available)

Supports Raspberry Pi Sense HAT and other IMU hardware.
"""

import logging
import platform
from typing import Optional, Dict, Any
import numpy as np

from .base import BaseSensor, SensorRegistry, SensorType, SensorStatus, SensorConfig

logger = logging.getLogger(__name__)


@SensorRegistry.register
class IMUSensor(BaseSensor):
    """IMU sensor for accelerometer, gyroscope, and magnetometer data."""
    
    sensor_type = SensorType.IMU
    sensor_name = "IMU Sensor"
    sensor_description = "6-axis IMU (accelerometer + gyroscope)"
    supported_platforms = ["linux", "raspberry_pi"]
    default_sample_rate = 100.0
    data_channels = 6  # accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    data_dtype = np.float32
    
    def __init__(self, config: Optional[SensorConfig] = None):
        super().__init__(config)
        self._sense_hat = None
        self._mock_mode = False
    
    def initialize(self) -> bool:
        """Initialize IMU hardware."""
        # Try Sense HAT first (Raspberry Pi)
        try:
            from sense_hat import SenseHat
            self._sense_hat = SenseHat()
            self._sense_hat.clear()
            self.status = SensorStatus.AVAILABLE
            logger.info("IMU initialized via Sense HAT")
            return True
        except ImportError:
            logger.debug("Sense HAT not available")
        except Exception as e:
            logger.debug(f"Sense HAT initialization failed: {e}")
        
        # Try generic IMU libraries
        try:
            # Could add support for other IMU libraries here
            # e.g., mpu6050, bno055, etc.
            pass
        except Exception:
            pass
        
        # Fall back to mock mode for development
        if platform.system() != "Linux":
            logger.info("IMU running in mock mode (non-Linux platform)")
            self._mock_mode = True
            self.status = SensorStatus.AVAILABLE
            return True
        
        self.status = SensorStatus.UNAVAILABLE
        self._error_message = "No IMU hardware found"
        return False
    
    def read_sample(self) -> Optional[np.ndarray]:
        """Read a single IMU sample.
        
        Returns:
            Array of [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        """
        try:
            if self._sense_hat:
                # Read from Sense HAT
                accel = self._sense_hat.get_accelerometer_raw()
                gyro = self._sense_hat.get_gyroscope_raw()
                
                return np.array([
                    accel['x'], accel['y'], accel['z'],
                    gyro['x'], gyro['y'], gyro['z']
                ], dtype=self.data_dtype)
            
            elif self._mock_mode:
                # Generate mock data for development
                return self._generate_mock_sample()
            
            else:
                return None
                
        except Exception as e:
            logger.error(f"IMU read error: {e}")
            return None
    
    def _generate_mock_sample(self) -> np.ndarray:
        """Generate mock IMU data for development."""
        # Simulate realistic IMU noise
        accel = np.random.normal(0, 0.1, 3)
        accel[2] += 9.8  # Gravity on z-axis
        gyro = np.random.normal(0, 0.01, 3)
        
        return np.array([
            accel[0], accel[1], accel[2],
            gyro[0], gyro[1], gyro[2]
        ], dtype=self.data_dtype)
    
    def cleanup(self):
        """Cleanup IMU resources."""
        if self._sense_hat:
            try:
                self._sense_hat.clear()
            except Exception:
                pass
            self._sense_hat = None
        self._mock_mode = False
    
    def get_info(self) -> Dict[str, Any]:
        """Get IMU sensor information."""
        info = super().get_info()
        info.update({
            "has_sense_hat": self._sense_hat is not None,
            "mock_mode": self._mock_mode,
            "channels": ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
        })
        return info
    
    def calibrate(self, duration_seconds: float = 5.0) -> Dict[str, Any]:
        """Calibrate the IMU by collecting baseline samples.
        
        Args:
            duration_seconds: Duration to collect calibration data
        
        Returns:
            Calibration results with offsets
        """
        logger.info(f"Starting IMU calibration for {duration_seconds}s")
        
        samples = []
        import time
        start = time.time()
        
        while (time.time() - start) < duration_seconds:
            sample = self.read_sample()
            if sample is not None:
                samples.append(sample)
            time.sleep(1.0 / self.config.sample_rate_hz)
        
        if not samples:
            return {"success": False, "error": "No samples collected"}
        
        samples = np.array(samples)
        
        # Calculate offsets (mean should be [0, 0, 9.8, 0, 0, 0] at rest)
        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)
        
        # Expected values at rest
        expected = np.array([0, 0, 9.8, 0, 0, 0])
        offsets = mean - expected
        
        return {
            "success": True,
            "num_samples": len(samples),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "offsets": offsets.tolist(),
        }
