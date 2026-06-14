"""macOS IMU Sensor Implementation."""

import logging
from typing import Optional, Dict, Any
import numpy as np

from thoth_core.sensors.base import BaseSensor, SensorRegistry, SensorType, SensorStatus, SensorConfig

logger = logging.getLogger(__name__)


@SensorRegistry.register
class MacIMUSensor(BaseSensor):
    """macOS IMU sensor stub."""

    sensor_type = SensorType.IMU
    sensor_name = "macOS IMU"
    sensor_description = "6-axis IMU stub for macOS (extend for external hardware)"
    supported_platforms = ["darwin"]
    default_sample_rate = 100.0
    data_channels = 6  # accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    data_dtype = np.float32

    def __init__(self, config: Optional[SensorConfig] = None):
        super().__init__(config)

    def initialize(self) -> bool:
        logger.error("No macOS IMU source configured")
        self.status = SensorStatus.UNAVAILABLE
        return False

    def read_sample(self) -> Optional[np.ndarray]:
        return None

    def cleanup(self):
        return None

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["channels"] = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        return info
