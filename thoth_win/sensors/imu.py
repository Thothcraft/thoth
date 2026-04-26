"""Windows IMU Sensor Implementation.

Runs in mock mode.  Extend for external IMU via USB/BLE.
"""

import logging
from typing import Optional, Dict, Any
import numpy as np

from thoth_core.sensors.base import BaseSensor, SensorRegistry, SensorType, SensorStatus, SensorConfig

logger = logging.getLogger(__name__)


@SensorRegistry.register
class WinIMUSensor(BaseSensor):
    """Windows IMU sensor (mock / dev mode)."""

    sensor_type = SensorType.IMU
    sensor_name = "Windows IMU"
    sensor_description = "6-axis IMU — mock mode (extend for external hardware)"
    supported_platforms = ["win32", "windows"]
    default_sample_rate = 100.0
    data_channels = 6
    data_dtype = np.float32

    def __init__(self, config: Optional[SensorConfig] = None):
        super().__init__(config)
        self._mock_mode = True

    def initialize(self) -> bool:
        self._mock_mode = True
        self.status = SensorStatus.AVAILABLE
        logger.info("Windows IMU initialized in mock mode")
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
        return info
