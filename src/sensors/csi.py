"""WiFi CSI Sensor Implementation.

This module provides WiFi Channel State Information (CSI) sensor support.
CSI data captures the fine-grained channel characteristics of WiFi signals,
useful for activity recognition, gesture detection, and localization.

Requires ESP32 CSI collection setup or compatible hardware.
"""

import logging
import os
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

from .base import BaseSensor, SensorRegistry, SensorType, SensorStatus, SensorConfig

logger = logging.getLogger(__name__)


@SensorRegistry.register
class CSISensor(BaseSensor):
    """WiFi CSI sensor for channel state information collection."""
    
    sensor_type = SensorType.CSI
    sensor_name = "WiFi CSI Sensor"
    sensor_description = "WiFi Channel State Information for sensing"
    supported_platforms = ["linux", "raspberry_pi"]
    default_sample_rate = 100.0
    data_channels = 64  # Number of subcarriers (typical for ESP32)
    data_dtype = np.float32
    
    def __init__(self, config: Optional[SensorConfig] = None):
        super().__init__(config)
        self._serial_port = None
        self._serial_connection = None
        self._mock_mode = False
        self._csi_buffer: List[np.ndarray] = []
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        
        # CSI-specific config
        self._num_subcarriers = config.custom_params.get("num_subcarriers", 64) if config else 64
        self._include_phase = config.custom_params.get("include_phase", True) if config else True
    
    def initialize(self) -> bool:
        """Initialize CSI collection hardware."""
        # Try to find ESP32 serial port
        serial_port = self._find_esp32_port()
        
        if serial_port:
            try:
                import serial
                self._serial_connection = serial.Serial(
                    serial_port,
                    baudrate=921600,
                    timeout=1.0
                )
                self._serial_port = serial_port
                self.status = SensorStatus.AVAILABLE
                logger.info(f"CSI initialized on {serial_port}")
                return True
            except ImportError:
                logger.warning("pyserial not installed, CSI collection unavailable")
            except Exception as e:
                logger.error(f"Failed to open serial port: {e}")
        
        # Check for CSI data files (offline mode)
        csi_data_dir = Path(__file__).parent.parent.parent / "data"
        csi_files = list(csi_data_dir.glob("csi_*.csv")) + list(csi_data_dir.glob("csi_*.npy"))
        
        if csi_files:
            logger.info(f"CSI running in file mode ({len(csi_files)} files available)")
            self._mock_mode = True
            self.status = SensorStatus.AVAILABLE
            return True
        
        # Fall back to mock mode for development
        logger.info("CSI running in mock mode (no hardware/files)")
        self._mock_mode = True
        self.status = SensorStatus.AVAILABLE
        return True
    
    def _find_esp32_port(self) -> Optional[str]:
        """Find ESP32 serial port."""
        import platform
        
        if platform.system() == "Linux":
            # Common ESP32 ports on Linux/Raspberry Pi
            ports = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0"]
            for port in ports:
                if os.path.exists(port):
                    return port
        
        elif platform.system() == "Darwin":  # macOS
            import glob
            ports = glob.glob("/dev/tty.usbserial*") + glob.glob("/dev/tty.SLAB*")
            if ports:
                return ports[0]
        
        elif platform.system() == "Windows":
            # Would need to enumerate COM ports
            pass
        
        return None
    
    def read_sample(self) -> Optional[np.ndarray]:
        """Read a single CSI sample.
        
        Returns:
            Array of CSI amplitudes (and optionally phases)
        """
        try:
            if self._serial_connection:
                return self._read_from_serial()
            elif self._mock_mode:
                return self._generate_mock_sample()
            else:
                return None
                
        except Exception as e:
            logger.error(f"CSI read error: {e}")
            return None
    
    def _read_from_serial(self) -> Optional[np.ndarray]:
        """Read CSI data from serial port."""
        if not self._serial_connection:
            return None
        
        try:
            line = self._serial_connection.readline().decode('utf-8').strip()
            if not line or not line.startswith("CSI_DATA"):
                return None
            
            # Parse CSI data (format depends on ESP32 firmware)
            # Expected format: CSI_DATA,<timestamp>,<rssi>,<data...>
            parts = line.split(",")
            if len(parts) < 4:
                return None
            
            # Extract CSI values
            csi_values = [float(x) for x in parts[3:]]
            
            # Reshape to amplitude (and phase if included)
            if self._include_phase:
                # Interleaved real/imag -> amplitude/phase
                num_values = len(csi_values) // 2
                real = np.array(csi_values[0::2])
                imag = np.array(csi_values[1::2])
                amplitude = np.sqrt(real**2 + imag**2)
                phase = np.arctan2(imag, real)
                return np.concatenate([amplitude, phase]).astype(self.data_dtype)
            else:
                return np.array(csi_values[:self._num_subcarriers], dtype=self.data_dtype)
                
        except Exception as e:
            logger.debug(f"Serial read error: {e}")
            return None
    
    def _generate_mock_sample(self) -> np.ndarray:
        """Generate mock CSI data for development."""
        # Simulate realistic CSI with noise
        base_amplitude = np.random.uniform(20, 40, self._num_subcarriers)
        noise = np.random.normal(0, 2, self._num_subcarriers)
        amplitude = base_amplitude + noise
        
        if self._include_phase:
            phase = np.random.uniform(-np.pi, np.pi, self._num_subcarriers)
            return np.concatenate([amplitude, phase]).astype(self.data_dtype)
        else:
            return amplitude.astype(self.data_dtype)
    
    def cleanup(self):
        """Cleanup CSI resources."""
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
        """Get CSI sensor information."""
        info = super().get_info()
        info.update({
            "serial_port": self._serial_port,
            "mock_mode": self._mock_mode,
            "num_subcarriers": self._num_subcarriers,
            "include_phase": self._include_phase,
            "channels": self._num_subcarriers * (2 if self._include_phase else 1),
        })
        return info
    
    def set_subcarrier_range(self, start: int, end: int):
        """Set the subcarrier range to use.
        
        Args:
            start: Start subcarrier index
            end: End subcarrier index
        """
        if 0 <= start < end <= 64:
            self._num_subcarriers = end - start
            self.config.custom_params["subcarrier_start"] = start
            self.config.custom_params["subcarrier_end"] = end
            logger.info(f"CSI subcarrier range set to [{start}, {end})")
