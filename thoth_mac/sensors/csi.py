"""macOS WiFi CSI Sensor — scaffolded for future ESP32 USB extension.

This sensor detects ESP32 devices connected via USB on macOS
(/dev/tty.usbserial-*, /dev/tty.SLAB*) and reads streamed
Channel State Information (CSI) from the receiver ESP32.

Currently scaffolded; will be fully functional once ESP32 CSI
firmware and USB detection are integrated.
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
class MacCSISensor(BaseSensor):
    """WiFi CSI sensor via USB-connected ESP32 on macOS."""

    sensor_type = SensorType.CSI
    sensor_name = "WiFi CSI Sensor (macOS)"
    sensor_description = "WiFi Channel State Information via USB-connected ESP32"
    supported_platforms = ["darwin"]
    default_sample_rate = 100.0
    data_channels = 64  # subcarriers
    data_dtype = np.float32

    def __init__(self, config: Optional[SensorConfig] = None):
        super().__init__(config)
        self._serial_port: Optional[str] = None
        self._serial_connection = None
        self._mock_mode = False
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        self._num_subcarriers = config.custom_params.get("num_subcarriers", 64) if config else 64
        self._include_phase = config.custom_params.get("include_phase", True) if config else True

    # ------------------------------------------------------------------
    # ESP32 detection
    # ------------------------------------------------------------------

    @staticmethod
    def find_esp32_ports() -> List[str]:
        """Find ESP32 serial ports on macOS."""
        patterns = [
            "/dev/tty.usbserial*",
            "/dev/tty.SLAB*",
            "/dev/tty.wchusbserial*",
            "/dev/cu.usbserial*",
            "/dev/cu.SLAB*",
        ]
        ports: List[str] = []
        for pat in patterns:
            ports.extend(glob.glob(pat))
        return ports

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

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
                logger.warning("pyserial not installed — CSI collection unavailable")
            except Exception as e:
                logger.error("Failed to open serial port: %s", e)

        # Fall back to mock mode
        logger.info("CSI running in mock mode (no ESP32 detected on macOS)")
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
        self._running = False
        if self._reader_thread:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        if self._serial_connection:
            try:
                self._serial_connection.close()
            except Exception:
                pass
            self._serial_connection = None
        self._mock_mode = False

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "serial_port": self._serial_port,
            "mock_mode": self._mock_mode,
            "num_subcarriers": self._num_subcarriers,
            "include_phase": self._include_phase,
            "detected_esp32_ports": self.find_esp32_ports(),
        })
        return info
