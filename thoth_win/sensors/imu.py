"""Windows IMU Sensor Implementation."""

import logging
from typing import Optional, Dict, Any
import numpy as np

from thoth_core.sensors.base import BaseSensor, SensorRegistry, SensorType, SensorStatus, SensorConfig

logger = logging.getLogger(__name__)


@SensorRegistry.register
class WinIMUSensor(BaseSensor):
    """Windows IMU sensor stub."""

    sensor_type = SensorType.IMU
    sensor_name = "Windows IMU"
    sensor_description = "6-axis IMU stub for Windows (extend for external hardware)"
    supported_platforms = ["win32", "windows"]
    default_sample_rate = 100.0
    data_channels = 6
    data_dtype = np.float32

    def __init__(self, config: Optional[SensorConfig] = None):
        super().__init__(config)

    def initialize(self) -> bool:
        logger.error("No Windows IMU source configured")
        self.status = SensorStatus.UNAVAILABLE
        return False

    def read_sample(self) -> Optional[np.ndarray]:
        return None

    def cleanup(self):
        return None

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        return info
