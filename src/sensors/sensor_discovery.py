"""Sensor Discovery Module.

This module provides comprehensive device type detection and sensor discovery
for various platforms including Raspberry Pi, laptops, desktops, and mobile devices.

Data is saved to: thoth/data/ directory with prefixes:
- imu_*.json - IMU/accelerometer data
- cam_*.mp4 or cam_*.jpg - Camera data
- mic_*.wav - Microphone audio data
- csi_*.csv - WiFi CSI data
"""

import os
import sys
import platform
import subprocess
import json
import logging
import time
import threading
import wave
import struct
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Enumeration of supported device types."""
    RASPBERRY_PI = "raspberry_pi"
    LAPTOP = "laptop"
    DESKTOP = "desktop"
    MOBILE = "mobile"
    UNKNOWN = "unknown"


class SensorType(Enum):
    """Enumeration of supported sensor types."""
    CAMERA = "camera"
    MICROPHONE = "microphone"
    IMU = "imu"  # Accelerometer, Gyroscope, Magnetometer
    SENSE_HAT = "sense_hat"  # Raspberry Pi Sense HAT
    GPS = "gps"
    WIFI_CSI = "wifi_csi"  # WiFi Channel State Information
    RADAR = "radar"  # mmWave radar (MMW-HAT)


@dataclass
class SensorInfo:
    """Information about a detected sensor."""
    sensor_type: str
    name: str
    available: bool
    device_path: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_type": self.sensor_type,
            "name": self.name,
            "available": self.available,
            "device_path": self.device_path,
            "capabilities": self.capabilities or {},
            "error": self.error
        }


@dataclass
class DeviceInfo:
    """Comprehensive device information."""
    device_type: str
    platform_system: str
    platform_release: str
    platform_machine: str
    processor: str
    hostname: str
    python_version: str
    cpu_count: int
    memory_total_gb: float
    is_raspberry_pi: bool
    raspberry_pi_model: Optional[str] = None
    sensors: List[SensorInfo] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "device_type": self.device_type,
            "platform_system": self.platform_system,
            "platform_release": self.platform_release,
            "platform_machine": self.platform_machine,
            "processor": self.processor,
            "hostname": self.hostname,
            "python_version": self.python_version,
            "cpu_count": self.cpu_count,
            "memory_total_gb": self.memory_total_gb,
            "is_raspberry_pi": self.is_raspberry_pi,
            "raspberry_pi_model": self.raspberry_pi_model,
            "sensors": [s.to_dict() for s in (self.sensors or [])]
        }
        return result


class SensorDiscovery:
    """Discovers device type and available sensors."""
    
    def __init__(self, data_dir: str = None):
        """Initialize sensor discovery.
        
        Args:
            data_dir: Directory for saving collected data. Defaults to thoth/data/
        """
        self.data_dir = data_dir or self._get_default_data_dir()
        os.makedirs(self.data_dir, exist_ok=True)
        self._device_info: Optional[DeviceInfo] = None
        self._collection_threads: Dict[str, threading.Thread] = {}
        self._collection_stop_events: Dict[str, threading.Event] = {}
        
    def _get_default_data_dir(self) -> str:
        """Get the default data directory."""
        # Try to find thoth/data relative to this file
        current_dir = Path(__file__).parent.parent.parent  # Go up to thoth/
        data_dir = current_dir / "data"
        if data_dir.exists():
            return str(data_dir)
        # Fallback to current working directory
        return os.path.join(os.getcwd(), "data")
    
    def detect_device_type(self) -> DeviceType:
        """Detect the type of device we're running on."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Check for Raspberry Pi
        if self._is_raspberry_pi():
            return DeviceType.RASPBERRY_PI
        
        # Check for mobile (Android/iOS - typically won't run Python directly)
        if 'android' in system or 'ios' in system:
            return DeviceType.MOBILE
        
        # Check for laptop vs desktop on Windows/Mac/Linux
        if system in ['windows', 'darwin', 'linux']:
            if self._has_battery():
                return DeviceType.LAPTOP
            else:
                return DeviceType.DESKTOP
        
        return DeviceType.UNKNOWN
    
    def _is_raspberry_pi(self) -> bool:
        """Check if running on a Raspberry Pi."""
        try:
            # Check /proc/cpuinfo for Raspberry Pi
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read().lower()
                    if 'raspberry' in cpuinfo or 'bcm' in cpuinfo:
                        return True
            
            # Check /proc/device-tree/model
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().lower()
                    if 'raspberry' in model:
                        return True
            
            # Check for Raspberry Pi OS
            if os.path.exists('/etc/os-release'):
                with open('/etc/os-release', 'r') as f:
                    os_release = f.read().lower()
                    if 'raspbian' in os_release or 'raspberry' in os_release:
                        return True
                        
        except Exception as e:
            logger.debug(f"Error checking for Raspberry Pi: {e}")
        
        return False
    
    def _get_raspberry_pi_model(self) -> Optional[str]:
        """Get the Raspberry Pi model if running on one."""
        try:
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    return f.read().strip().replace('\x00', '')
        except Exception:
            pass
        return None
    
    def _has_battery(self) -> bool:
        """Check if the device has a battery (laptop indicator)."""
        try:
            import psutil
            battery = psutil.sensors_battery()
            return battery is not None
        except Exception:
            pass
        
        # Linux-specific check
        if platform.system() == 'Linux':
            battery_paths = [
                '/sys/class/power_supply/BAT0',
                '/sys/class/power_supply/BAT1',
                '/sys/class/power_supply/battery'
            ]
            for path in battery_paths:
                if os.path.exists(path):
                    return True
        
        # Windows-specific check
        if platform.system() == 'Windows':
            try:
                result = subprocess.run(
                    ['powershell', '-Command', 
                     '(Get-WmiObject Win32_Battery).BatteryStatus'],
                    capture_output=True, text=True, timeout=5
                )
                if result.stdout.strip():
                    return True
            except Exception:
                pass
        
        # macOS check
        if platform.system() == 'Darwin':
            try:
                result = subprocess.run(
                    ['pmset', '-g', 'batt'],
                    capture_output=True, text=True, timeout=5
                )
                if 'Battery' in result.stdout or 'InternalBattery' in result.stdout:
                    return True
            except Exception:
                pass
        
        return False
    
    def discover_sensors(self) -> List[SensorInfo]:
        """Discover all available sensors on the device."""
        sensors = []
        device_type = self.detect_device_type()
        
        # Always check for camera
        sensors.append(self._detect_camera())
        
        # Always check for microphone
        sensors.append(self._detect_microphone())
        
        # Check for IMU (accelerometer/gyroscope)
        sensors.append(self._detect_imu())
        
        # Raspberry Pi specific sensors
        if device_type == DeviceType.RASPBERRY_PI:
            sensors.append(self._detect_sense_hat())
            sensors.append(self._detect_mmw_radar())
            sensors.append(self._detect_wifi_csi())
        
        # Filter out None values
        return [s for s in sensors if s is not None]
    
    def _detect_camera(self) -> SensorInfo:
        """Detect camera availability."""
        try:
            # Try OpenCV first
            try:
                import cv2
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, _ = cap.read()
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    if ret:
                        return SensorInfo(
                            sensor_type=SensorType.CAMERA.value,
                            name="Camera",
                            available=True,
                            device_path="/dev/video0" if platform.system() == 'Linux' else "default",
                            capabilities={
                                "width": width,
                                "height": height,
                                "fps": fps,
                                "backend": "opencv"
                            }
                        )
                cap.release()
            except ImportError:
                pass
            
            # Linux: Check for video devices
            if platform.system() == 'Linux':
                video_devices = list(Path('/dev').glob('video*'))
                if video_devices:
                    return SensorInfo(
                        sensor_type=SensorType.CAMERA.value,
                        name="Camera",
                        available=True,
                        device_path=str(video_devices[0]),
                        capabilities={"devices": [str(d) for d in video_devices]}
                    )
            
            # Windows: Check for cameras via DirectShow
            if platform.system() == 'Windows':
                try:
                    result = subprocess.run(
                        ['powershell', '-Command',
                         'Get-PnpDevice -Class Camera | Select-Object -Property FriendlyName'],
                        capture_output=True, text=True, timeout=10
                    )
                    if 'Camera' in result.stdout or 'Webcam' in result.stdout:
                        return SensorInfo(
                            sensor_type=SensorType.CAMERA.value,
                            name="Camera",
                            available=True,
                            capabilities={"backend": "windows_pnp"}
                        )
                except Exception:
                    pass
            
            return SensorInfo(
                sensor_type=SensorType.CAMERA.value,
                name="Camera",
                available=False,
                error="No camera detected"
            )
            
        except Exception as e:
            return SensorInfo(
                sensor_type=SensorType.CAMERA.value,
                name="Camera",
                available=False,
                error=str(e)
            )
    
    def _detect_microphone(self) -> SensorInfo:
        """Detect microphone availability."""
        try:
            # Try PyAudio
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                device_count = p.get_device_count()
                input_devices = []
                
                for i in range(device_count):
                    info = p.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0:
                        input_devices.append({
                            'index': i,
                            'name': info['name'],
                            'channels': info['maxInputChannels'],
                            'sample_rate': int(info['defaultSampleRate'])
                        })
                
                p.terminate()
                
                if input_devices:
                    return SensorInfo(
                        sensor_type=SensorType.MICROPHONE.value,
                        name="Microphone",
                        available=True,
                        capabilities={
                            "devices": input_devices,
                            "default_device": input_devices[0],
                            "backend": "pyaudio"
                        }
                    )
            except ImportError:
                pass
            
            # Linux: Check for audio input devices
            if platform.system() == 'Linux':
                try:
                    result = subprocess.run(
                        ['arecord', '-l'],
                        capture_output=True, text=True, timeout=5
                    )
                    if 'card' in result.stdout.lower():
                        return SensorInfo(
                            sensor_type=SensorType.MICROPHONE.value,
                            name="Microphone",
                            available=True,
                            capabilities={"backend": "alsa"}
                        )
                except Exception:
                    pass
            
            # Windows: Check for audio input
            if platform.system() == 'Windows':
                try:
                    result = subprocess.run(
                        ['powershell', '-Command',
                         'Get-PnpDevice -Class AudioEndpoint | Where-Object {$_.FriendlyName -like "*Microphone*"}'],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.stdout.strip():
                        return SensorInfo(
                            sensor_type=SensorType.MICROPHONE.value,
                            name="Microphone",
                            available=True,
                            capabilities={"backend": "windows_pnp"}
                        )
                except Exception:
                    pass
            
            return SensorInfo(
                sensor_type=SensorType.MICROPHONE.value,
                name="Microphone",
                available=False,
                error="No microphone detected"
            )
            
        except Exception as e:
            return SensorInfo(
                sensor_type=SensorType.MICROPHONE.value,
                name="Microphone",
                available=False,
                error=str(e)
            )
    
    def _detect_imu(self) -> SensorInfo:
        """Detect IMU (accelerometer/gyroscope) availability."""
        try:
            # Check for Windows Motion Sensors
            if platform.system() == 'Windows':
                try:
                    # Try Windows Sensor API via ctypes
                    import ctypes
                    sensorsapi = ctypes.windll.LoadLibrary("SensorsApi.dll")
                    return SensorInfo(
                        sensor_type=SensorType.IMU.value,
                        name="Motion Sensors",
                        available=True,
                        capabilities={"backend": "windows_sensor_api"}
                    )
                except Exception:
                    pass
            
            # Linux: Check for IIO devices (common for laptop accelerometers)
            if platform.system() == 'Linux':
                iio_devices = list(Path('/sys/bus/iio/devices').glob('iio:device*')) if Path('/sys/bus/iio/devices').exists() else []
                for device in iio_devices:
                    name_file = device / 'name'
                    if name_file.exists():
                        with open(name_file, 'r') as f:
                            name = f.read().strip()
                            if 'accel' in name.lower() or 'gyro' in name.lower():
                                return SensorInfo(
                                    sensor_type=SensorType.IMU.value,
                                    name=f"IMU ({name})",
                                    available=True,
                                    device_path=str(device),
                                    capabilities={"backend": "linux_iio"}
                                )
            
            # macOS: Check for motion sensors
            if platform.system() == 'Darwin':
                try:
                    # macOS laptops have Sudden Motion Sensor
                    result = subprocess.run(
                        ['ioreg', '-c', 'SMCMotionSensor'],
                        capture_output=True, text=True, timeout=5
                    )
                    if 'SMCMotionSensor' in result.stdout:
                        return SensorInfo(
                            sensor_type=SensorType.IMU.value,
                            name="Sudden Motion Sensor",
                            available=True,
                            capabilities={"backend": "macos_smc"}
                        )
                except Exception:
                    pass
            
            return SensorInfo(
                sensor_type=SensorType.IMU.value,
                name="IMU",
                available=False,
                error="No IMU/motion sensors detected"
            )
            
        except Exception as e:
            return SensorInfo(
                sensor_type=SensorType.IMU.value,
                name="IMU",
                available=False,
                error=str(e)
            )
    
    def _detect_sense_hat(self) -> SensorInfo:
        """Detect Raspberry Pi Sense HAT."""
        try:
            from sense_hat import SenseHat
            sense = SenseHat()
            # Try to read a value to confirm it's working
            temp = sense.get_temperature()
            return SensorInfo(
                sensor_type=SensorType.SENSE_HAT.value,
                name="Sense HAT",
                available=True,
                capabilities={
                    "temperature": True,
                    "humidity": True,
                    "pressure": True,
                    "imu": True,
                    "led_matrix": True,
                    "joystick": True
                }
            )
        except ImportError:
            return SensorInfo(
                sensor_type=SensorType.SENSE_HAT.value,
                name="Sense HAT",
                available=False,
                error="sense_hat library not installed"
            )
        except Exception as e:
            return SensorInfo(
                sensor_type=SensorType.SENSE_HAT.value,
                name="Sense HAT",
                available=False,
                error=f"Sense HAT not detected: {str(e)}"
            )
    
    def _detect_mmw_radar(self) -> SensorInfo:
        """Detect mmWave radar (MMW-HAT)."""
        try:
            # Check if MMW-HAT directory exists
            mmw_paths = [
                Path(__file__).parent.parent.parent / 'WS' / 'MMW-HAT',
                Path('/home/pi/MMW-HAT'),
                Path.home() / 'MMW-HAT'
            ]
            
            for path in mmw_paths:
                if path.exists():
                    return SensorInfo(
                        sensor_type=SensorType.RADAR.value,
                        name="MMW-HAT Radar",
                        available=True,
                        device_path=str(path),
                        capabilities={
                            "type": "BGT60TR13C",
                            "frequency": "60GHz"
                        }
                    )
            
            return SensorInfo(
                sensor_type=SensorType.RADAR.value,
                name="MMW-HAT Radar",
                available=False,
                error="MMW-HAT not found"
            )
            
        except Exception as e:
            return SensorInfo(
                sensor_type=SensorType.RADAR.value,
                name="MMW-HAT Radar",
                available=False,
                error=str(e)
            )
    
    def _detect_wifi_csi(self) -> SensorInfo:
        """Detect WiFi CSI capability."""
        try:
            # Check for ESP32 CSI receiver
            csi_paths = [
                Path(__file__).parent.parent.parent / 'WS' / 'csi_recv',
                Path('/home/pi/csi_recv')
            ]
            
            for path in csi_paths:
                if path.exists():
                    return SensorInfo(
                        sensor_type=SensorType.WIFI_CSI.value,
                        name="WiFi CSI",
                        available=True,
                        device_path=str(path),
                        capabilities={
                            "type": "ESP32",
                            "subcarriers": 52
                        }
                    )
            
            return SensorInfo(
                sensor_type=SensorType.WIFI_CSI.value,
                name="WiFi CSI",
                available=False,
                error="CSI receiver not found"
            )
            
        except Exception as e:
            return SensorInfo(
                sensor_type=SensorType.WIFI_CSI.value,
                name="WiFi CSI",
                available=False,
                error=str(e)
            )
    
    def get_device_info(self) -> DeviceInfo:
        """Get comprehensive device information including sensors."""
        if self._device_info is not None:
            return self._device_info
        
        import psutil
        
        device_type = self.detect_device_type()
        sensors = self.discover_sensors()
        
        memory = psutil.virtual_memory()
        
        self._device_info = DeviceInfo(
            device_type=device_type.value,
            platform_system=platform.system(),
            platform_release=platform.release(),
            platform_machine=platform.machine(),
            processor=platform.processor(),
            hostname=platform.node(),
            python_version=platform.python_version(),
            cpu_count=os.cpu_count() or 1,
            memory_total_gb=round(memory.total / (1024**3), 2),
            is_raspberry_pi=self._is_raspberry_pi(),
            raspberry_pi_model=self._get_raspberry_pi_model(),
            sensors=sensors
        )
        
        return self._device_info
    
    def start_collection(self, sensor_type: str, duration_minutes: int = 1) -> Tuple[bool, str]:
        """Start data collection for a specific sensor.
        
        Args:
            sensor_type: Type of sensor to collect from
            duration_minutes: Duration in minutes (1, 5, or 10)
            
        Returns:
            Tuple of (success, message or filename)
        """
        if sensor_type in self._collection_threads and self._collection_threads[sensor_type].is_alive():
            return False, f"Collection already in progress for {sensor_type}"
        
        duration_seconds = duration_minutes * 60
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        stop_event = threading.Event()
        self._collection_stop_events[sensor_type] = stop_event
        
        if sensor_type == SensorType.CAMERA.value:
            filename = f"cam_{timestamp}.mp4"
            thread = threading.Thread(
                target=self._collect_camera,
                args=(filename, duration_seconds, stop_event)
            )
        elif sensor_type == SensorType.MICROPHONE.value:
            filename = f"mic_{timestamp}.wav"
            thread = threading.Thread(
                target=self._collect_microphone,
                args=(filename, duration_seconds, stop_event)
            )
        elif sensor_type == SensorType.IMU.value or sensor_type == SensorType.SENSE_HAT.value:
            filename = f"imu_{timestamp}.json"
            thread = threading.Thread(
                target=self._collect_imu,
                args=(filename, duration_seconds, stop_event)
            )
        else:
            return False, f"Unsupported sensor type: {sensor_type}"
        
        thread.daemon = True
        thread.start()
        self._collection_threads[sensor_type] = thread
        
        return True, filename
    
    def stop_collection(self, sensor_type: str) -> bool:
        """Stop data collection for a specific sensor."""
        if sensor_type in self._collection_stop_events:
            self._collection_stop_events[sensor_type].set()
            return True
        return False
    
    def is_collecting(self, sensor_type: str) -> bool:
        """Check if collection is in progress for a sensor."""
        if sensor_type in self._collection_threads:
            return self._collection_threads[sensor_type].is_alive()
        return False
    
    def _collect_camera(self, filename: str, duration: int, stop_event: threading.Event):
        """Collect camera data."""
        try:
            import cv2
            filepath = os.path.join(self.data_dir, filename)
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("Failed to open camera")
                return
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            
            start_time = time.time()
            while time.time() - start_time < duration and not stop_event.is_set():
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                else:
                    break
            
            cap.release()
            out.release()
            logger.info(f"Camera data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error collecting camera data: {e}")
    
    def _collect_microphone(self, filename: str, duration: int, stop_event: threading.Event):
        """Collect microphone data."""
        try:
            import pyaudio
            filepath = os.path.join(self.data_dir, filename)
            
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 44100
            
            p = pyaudio.PyAudio()
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            frames = []
            start_time = time.time()
            
            while time.time() - start_time < duration and not stop_event.is_set():
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Save as WAV file
            wf = wave.open(filepath, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            logger.info(f"Microphone data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error collecting microphone data: {e}")
    
    def _collect_imu(self, filename: str, duration: int, stop_event: threading.Event):
        """Collect IMU/Sense HAT data."""
        try:
            filepath = os.path.join(self.data_dir, filename)
            readings = []
            
            # Try Sense HAT first
            sense = None
            try:
                from sense_hat import SenseHat
                sense = SenseHat()
            except (ImportError, Exception):
                pass
            
            start_time = time.time()
            sample_rate = 50  # 50 Hz
            
            while time.time() - start_time < duration and not stop_event.is_set():
                reading = {
                    "timestamp": datetime.now().isoformat()
                }
                
                if sense:
                    orientation = sense.get_orientation()
                    accel = sense.get_accelerometer_raw()
                    gyro = sense.get_gyroscope_raw()
                    mag = sense.get_compass_raw()
                    
                    reading.update({
                        "orientation": orientation,
                        "accelerometer": accel,
                        "gyroscope": gyro,
                        "magnetometer": mag,
                        "temperature": sense.get_temperature(),
                        "humidity": sense.get_humidity(),
                        "pressure": sense.get_pressure()
                    })
                else:
                    # Fallback: try to get IMU data from other sources
                    reading["error"] = "No IMU source available"
                
                readings.append(reading)
                time.sleep(1.0 / sample_rate)
            
            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(readings, f, indent=2)
            
            logger.info(f"IMU data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error collecting IMU data: {e}")


# Global instance for easy access
_discovery_instance: Optional[SensorDiscovery] = None


def get_sensor_discovery(data_dir: str = None) -> SensorDiscovery:
    """Get or create the global SensorDiscovery instance."""
    global _discovery_instance
    if _discovery_instance is None:
        _discovery_instance = SensorDiscovery(data_dir)
    return _discovery_instance


def get_device_info() -> Dict[str, Any]:
    """Get device information as a dictionary."""
    return get_sensor_discovery().get_device_info().to_dict()


def get_available_sensors() -> List[Dict[str, Any]]:
    """Get list of available sensors."""
    info = get_sensor_discovery().get_device_info()
    return [s.to_dict() for s in info.sensors if s.available]


if __name__ == "__main__":
    # Test the sensor discovery
    logging.basicConfig(level=logging.INFO)
    
    discovery = SensorDiscovery()
    info = discovery.get_device_info()
    
    print("\n=== Device Information ===")
    print(json.dumps(info.to_dict(), indent=2))
    
    print("\n=== Available Sensors ===")
    for sensor in info.sensors:
        status = "✓" if sensor.available else "✗"
        print(f"  {status} {sensor.name}: {sensor.available}")
        if sensor.error:
            print(f"      Error: {sensor.error}")
