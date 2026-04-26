"""Windows Microphone Sensor Implementation.

Uses PyAudio to capture audio from the default Windows recording device.
"""

import logging
from typing import Optional
import numpy as np

from thoth_core.sensors.base import BaseSensor, SensorRegistry, SensorType, SensorStatus, SensorConfig

logger = logging.getLogger(__name__)


@SensorRegistry.register
class WinMicrophoneSensor(BaseSensor):
    """Windows microphone sensor via PyAudio."""

    sensor_type = SensorType.MICROPHONE
    sensor_name = "Windows Microphone"
    sensor_description = "Built-in or USB microphone"
    supported_platforms = ["win32", "windows"]
    default_sample_rate = 44100.0
    data_channels = 1
    data_dtype = np.float32

    def __init__(self, config: Optional[SensorConfig] = None):
        super().__init__(config)
        self._pyaudio = None
        self._stream = None
        self._chunk_size = 1024

    def initialize(self) -> bool:
        try:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()
            if self._pyaudio.get_device_count() > 0:
                self.status = SensorStatus.AVAILABLE
                logger.info("Windows microphone initialized")
                return True
        except ImportError:
            logger.debug("PyAudio not available")
        except Exception as e:
            logger.debug(f"PyAudio init failed: {e}")
        self.status = SensorStatus.UNAVAILABLE
        self._error_message = "No audio backend available"
        return False

    def read_sample(self) -> Optional[np.ndarray]:
        if self._pyaudio and self._stream:
            try:
                data = self._stream.read(self._chunk_size, exception_on_overflow=False)
                return np.frombuffer(data, dtype=np.int16).astype(self.data_dtype) / 32768.0
            except Exception as e:
                logger.error(f"Mic read error: {e}")
        return None

    def cleanup(self):
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pyaudio:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None
