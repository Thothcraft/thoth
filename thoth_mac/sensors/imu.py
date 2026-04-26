"""macOS IMU Sensor Implementation.

macOS laptops have an accelerometer (Sudden Motion Sensor) but no
public API exposes it in recent versions.  This sensor runs in mock
mode for development and will be extended when real hardware is
connected (e.g., external IMU via USB/BLE).
"""

import logging
from typing import Optional, Dict, Any
import numpy as np

from thoth_core.sensors.base import BaseSensor, SensorRegistry, SensorType, SensorStatus, SensorConfig

logger = logging.getLogger(__name__)


@SensorRegistry.register
class MacIMUSensor(BaseSensor):
    """macOS IMU sensor (mock / dev mode)."""

    sensor_type = SensorType.IMU
    sensor_name = "macOS IMU"
    sensor_description = "6-axis IMU — mock mode on macOS (extend for external hardware)"
    supported_platforms = ["darwin"]
    default_sample_rate = 100.0
    data_channels = 6  # accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    data_dtype = np.float32

    def __init__(self, config: Optional[SensorConfig] = None):
        super().__init__(config)
        self._mock_mode = True

    def initialize(self) -> bool:
        # TODO: detect external IMU via USB/BLE
        self._mock_mode = True
        self.status = SensorStatus.AVAILABLE
        logger.info("macOS IMU initialized in mock mode")
        return True

    def read_sample(self) -> Optional[np.ndarray]:
        if self._mock_mode:
            accel = np.random.normal(0, 0.1, 3).astype(self.data_dtype)
            accel[2] += 9.8
            gyro = np.random.normal(0, 0.01, 3).astype(self.data_dtype)
            return np.concatenate([accel, gyro])
        return None

    def cleanup(self):
        self._mock_mode = False

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["mock_mode"] = self._mock_mode
        info["channels"] = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        return info
