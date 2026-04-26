"""macOS Camera Sensor Implementation.

Uses OpenCV to capture from built-in FaceTime camera or any connected
camera device on macOS.
"""

import logging
import platform
from typing import Optional, Dict, Any
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from thoth_core.sensors.base import BaseSensor, SensorRegistry, SensorType, SensorStatus, SensorConfig

logger = logging.getLogger(__name__)


@SensorRegistry.register
class MacCameraSensor(BaseSensor):
    """macOS camera sensor using OpenCV."""

    sensor_type = SensorType.CAMERA
    sensor_name = "macOS Camera"
    sensor_description = "Built-in or connected camera (FaceTime HD, Continuity, etc.)"
    supported_platforms = ["darwin"]
    default_sample_rate = 30.0
    data_channels = 3  # BGR
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
        self._cap = cv2.VideoCapture(self._camera_index)
        if self._cap.isOpened():
            self.status = SensorStatus.AVAILABLE
            logger.info("macOS camera initialized (index %d)", self._camera_index)
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

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["camera_index"] = self._camera_index
        if self._cap and self._cap.isOpened():
            info["resolution"] = (
                f"{int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
            )
        return info
