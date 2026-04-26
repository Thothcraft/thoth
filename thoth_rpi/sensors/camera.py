"""Raspberry Pi Camera Sensor Implementation.

Uses OpenCV or picamera2 for RPi camera module.
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
class RPiCameraSensor(BaseSensor):
    """Raspberry Pi camera sensor."""

    sensor_type = SensorType.CAMERA
    sensor_name = "RPi Camera"
    sensor_description = "Raspberry Pi camera module or USB camera"
    supported_platforms = ["linux", "raspberry_pi"]
    default_sample_rate = 30.0
    data_channels = 3
    data_dtype = np.uint8

    def __init__(self, config: Optional[SensorConfig] = None):
        super().__init__(config)
        self._cap = None
        self._picamera = None

    def initialize(self) -> bool:
        # Try picamera2 first (RPi native)
        try:
            from picamera2 import Picamera2
            self._picamera = Picamera2()
            self._picamera.configure(self._picamera.create_still_configuration())
            self._picamera.start()
            self.status = SensorStatus.AVAILABLE
            logger.info("RPi camera initialized via picamera2")
            return True
        except (ImportError, Exception) as e:
            logger.debug(f"picamera2 not available: {e}")

        # Fallback to OpenCV
        if cv2 is not None:
            self._cap = cv2.VideoCapture(0)
            if self._cap.isOpened():
                self.status = SensorStatus.AVAILABLE
                logger.info("RPi camera initialized via OpenCV")
                return True

        self.status = SensorStatus.UNAVAILABLE
        self._error_message = "No camera detected"
        return False

    def read_sample(self) -> Optional[np.ndarray]:
        if self._picamera:
            try:
                frame = self._picamera.capture_array()
                return frame.flatten().astype(self.data_dtype)
            except Exception as e:
                logger.error(f"picamera2 read error: {e}")
                return None
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret and frame is not None:
                return frame.flatten().astype(self.data_dtype)
        return None

    def cleanup(self):
        if self._picamera:
            try:
                self._picamera.stop()
            except Exception:
                pass
            self._picamera = None
        if self._cap:
            self._cap.release()
            self._cap = None
