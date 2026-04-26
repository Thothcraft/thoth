"""Raspberry Pi WiFi CSI Sensor Implementation.

Reads CSI data from a USB-connected ESP32 receiver on Linux serial ports
(/dev/ttyUSB0, /dev/ttyACM0, etc.).
"""

import glob
import logging
import os
import threading
from typing import Optional, Dict, Any, List
import numpy as np

from thoth_core.sensors.base import BaseSensor, SensorRegistry, SensorType, SensorStatus, SensorConfig

logger = logging.getLogger(__name__)


@SensorRegistry.register
class RPiCSISensor(BaseSensor):
    """WiFi CSI sensor via USB-connected ESP32 on Raspberry Pi."""

    sensor_type = SensorType.CSI
    sensor_name = "WiFi CSI Sensor (RPi)"
    sensor_description = "WiFi Channel State Information via ESP32"
    supported_platforms = ["linux", "raspberry_pi"]
    default_sample_rate = 100.0
    data_channels = 64
    data_dtype = np.float32

    def __init__(self, config: Optional[SensorConfig] = None):
        super().__init__(config)
        self._serial_port: Optional[str] = None
        self._serial_connection = None
        self._mock_mode = False
        self._num_subcarriers = config.custom_params.get("num_subcarriers", 64) if config else 64
        self._include_phase = config.custom_params.get("include_phase", True) if config else True

    @staticmethod
    def find_esp32_ports() -> List[str]:
        """Find ESP32 serial ports on Linux."""
        candidates = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0", "/dev/ttyACM1"]
        return [p for p in candidates if os.path.exists(p)]

    def initialize(self) -> bool:
        ports = self.find_esp32_ports()
        if ports:
            try:
                import serial
                self._serial_connection = serial.Serial(ports[0], baudrate=921600, timeout=1.0)
                self._serial_port = ports[0]
                self.status = SensorStatus.AVAILABLE
                logger.info("CSI initialized on %s", self._serial_port)
                return True
            except ImportError:
                logger.warning("pyserial not installed")
            except Exception as e:
                logger.error("Failed to open serial port: %s", e)
        logger.info("CSI running in mock mode (no ESP32 detected)")
        self._mock_mode = True
        self.status = SensorStatus.AVAILABLE
        return True

    def read_sample(self) -> Optional[np.ndarray]:
        if self._serial_connection:
            return self._read_from_serial()
        if self._mock_mode:
            return self._generate_mock_sample()
        return None

    def _read_from_serial(self) -> Optional[np.ndarray]:
        if not self._serial_connection:
            return None
        try:
            line = self._serial_connection.readline().decode("utf-8").strip()
            if not line or not line.startswith("CSI_DATA"):
                return None
            parts = line.split(",")
            if len(parts) < 4:
                return None
            csi_values = [float(x) for x in parts[3:]]
            if self._include_phase:
                real = np.array(csi_values[0::2])
                imag = np.array(csi_values[1::2])
                amplitude = np.sqrt(real ** 2 + imag ** 2)
                phase = np.arctan2(imag, real)
                return np.concatenate([amplitude, phase]).astype(self.data_dtype)
            return np.array(csi_values[: self._num_subcarriers], dtype=self.data_dtype)
        except Exception as e:
            logger.debug("Serial read error: %s", e)
            return None

    def _generate_mock_sample(self) -> np.ndarray:
        base = np.random.uniform(20, 40, self._num_subcarriers)
        noise = np.random.normal(0, 2, self._num_subcarriers)
        amplitude = base + noise
        if self._include_phase:
            phase = np.random.uniform(-np.pi, np.pi, self._num_subcarriers)
            return np.concatenate([amplitude, phase]).astype(self.data_dtype)
        return amplitude.astype(self.data_dtype)

    def cleanup(self):
        if self._serial_connection:
            try:
                self._serial_connection.close()
            except Exception:
                pass
            self._serial_connection = None
        self._mock_mode = False
