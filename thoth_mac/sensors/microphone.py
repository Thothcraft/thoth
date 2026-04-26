"""macOS Microphone Sensor Implementation.

Uses PyAudio (or falls back to subprocess sox/ffmpeg) to capture audio
from the default macOS input device.
"""

import logging
import subprocess
import tempfile
import os
from typing import Optional, Dict, Any
import numpy as np

from thoth_core.sensors.base import BaseSensor, SensorRegistry, SensorType, SensorStatus, SensorConfig

logger = logging.getLogger(__name__)


@SensorRegistry.register
class MacMicrophoneSensor(BaseSensor):
    """macOS microphone sensor."""

    sensor_type = SensorType.MICROPHONE
    sensor_name = "macOS Microphone"
    sensor_description = "Built-in or connected microphone"
    supported_platforms = ["darwin"]
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
                logger.info("macOS microphone initialized via PyAudio")
                return True
        except ImportError:
            logger.debug("PyAudio not available, will use sox/ffmpeg fallback")
        except Exception as e:
            logger.debug(f"PyAudio init failed: {e}")

        # Fallback: check if sox or ffmpeg is available
        for cmd in ("rec", "ffmpeg"):
            try:
                subprocess.run([cmd, "--version"], capture_output=True, timeout=3)
                self.status = SensorStatus.AVAILABLE
                logger.info(f"macOS microphone initialized via {cmd}")
                return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        self.status = SensorStatus.UNAVAILABLE
        self._error_message = "No audio backend (pyaudio/sox/ffmpeg)"
        return False

    def read_sample(self) -> Optional[np.ndarray]:
        if self._pyaudio and self._stream:
            try:
                data = self._stream.read(self._chunk_size, exception_on_overflow=False)
                return np.frombuffer(data, dtype=np.int16).astype(self.data_dtype) / 32768.0
            except Exception as e:
                logger.error(f"Mic read error: {e}")
                return None
        # In non-stream mode, return None (recording is handled via subprocess)
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
