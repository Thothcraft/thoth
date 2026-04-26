"""Windows Camera Sensor Implementation.

Uses OpenCV (DirectShow backend) to capture from built-in or USB cameras.
"""

import logging
from typing import Optional, Dict, Any
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from thoth_core.sensors.base import BaseSensor, SensorRegistry, SensorType, SensorStatus, SensorConfig

logger = logging.getLogger(__name__)


@SensorRegistry.register
class WinCameraSensor(BaseSensor):
    """Windows camera sensor using OpenCV DirectShow."""

    sensor_type = SensorType.CAMERA
    sensor_name = "Windows Camera"
    sensor_description = "Built-in or USB camera (DirectShow)"
    supported_platforms = ["win32", "windows"]
    default_sample_rate = 30.0
    data_channels = 3
    data_dtype = np.uint8

    def __init__(self, config: Optional[SensorConfig] = None):
        super().__init__(config)
        self._cap = None
        self._camera_index = 0
        if config and config.custom_params:
            self._camera_index = config.custom_params.get("camera_index", 0)

    def initialize(self) -> bool:
        if cv2 is None:
            self._error_message = "opencv-python not installed"
            self.status = SensorStatus.UNAVAILABLE
            return False
        self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        if self._cap.isOpened():
            self.status = SensorStatus.AVAILABLE
            logger.info("Windows camera initialized (index %d)", self._camera_index)
            return True
        self.status = SensorStatus.UNAVAILABLE
        self._error_message = "Camera not detected"
        return False

    def read_sample(self) -> Optional[np.ndarray]:
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if ret and frame is not None:
            return frame.flatten().astype(self.data_dtype)
        return None

    def cleanup(self):
        if self._cap:
            self._cap.release()
            self._cap = None
