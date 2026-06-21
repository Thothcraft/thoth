"""Attached sensor detection for Thoth data collection."""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from .config import Config


THOTH_ROOT = Path(__file__).resolve().parents[2]
MMW_RELEASE = THOTH_ROOT / "WS" / "MMW-HAT" / "MMW-HAT-Release"
RADAR_CONFIG_DIR = MMW_RELEASE / "radar_config" / "config_3rx_3m"


def serial_candidates() -> List[str]:
    candidates: List[str] = []
    try:
        import serial.tools.list_ports as list_ports

        candidates.extend(port.device for port in list_ports.comports())
    except Exception:
        pass

    for pattern in ("/dev/serial/by-id/*", "/dev/ttyACM*", "/dev/ttyUSB*"):
        for port in glob.glob(pattern):
            resolved = str(Path(port).resolve()) if port.startswith("/dev/serial/by-id/") else port
            if resolved not in candidates:
                candidates.append(resolved)
    return sorted(candidates)


def detect_usb_camera() -> Dict[str, Any]:
    requested = Config.CAPTURE_CAMERA_DEVICE
    devices = sorted(glob.glob("/dev/video*"))
    online = os.path.exists(requested) or bool(devices)
    source = requested if os.path.exists(requested) else (devices[0] if devices else requested)
    return {
        "name": "USB Camera",
        "key": "usb_camera",
        "online": online,
        "source": source,
        "stream": "/captures/live/video",
        "files": "mp4 video",
        "devices": devices,
    }


def detect_sense_hat() -> Dict[str, Any]:
    online = False
    error = None
    try:
        from sense_hat import SenseHat

        SenseHat()
        online = True
    except Exception as exc:
        error = str(exc)
    return {
        "name": "Sense HAT",
        "key": "sense_hat",
        "online": online,
        "source": "GPIO / I2C",
        "stream": None,
        "files": "imu/environment jsonl",
        "error": error,
    }


def detect_esp32_csi() -> Dict[str, Any]:
    candidates = serial_candidates()
    return {
        "name": "ESP32 CSI",
        "key": "esp32_csi",
        "online": bool(candidates),
        "source": candidates[0] if candidates else "USB serial",
        "stream": "/captures/live/csi",
        "files": "csv/jsonl",
        "devices": candidates,
    }


def detect_dreamhat_radar() -> Dict[str, Any]:
    service_active = False
    try:
        service_active = subprocess.run(
            ["systemctl", "is-active", "thoth-collector"],
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip() == "active"
    except Exception:
        service_active = False

    driver_available = MMW_RELEASE.exists() and RADAR_CONFIG_DIR.exists()
    chip_online = False
    error = None
    if driver_available:
        try:
            if str(MMW_RELEASE) not in sys.path:
                sys.path.insert(0, str(MMW_RELEASE))
            from utility.BGT60TR13C import BGT60TR13C, RET_VAL_OK

            radar = BGT60TR13C(spi_speed=50_000_000)
            chip_online = radar.check_chip_id() == RET_VAL_OK
        except Exception as exc:
            error = str(exc)

    return {
        "name": "DreamHAT+ Radar",
        "key": "dreamhat_radar",
        "online": bool(service_active or chip_online),
        "source": "BGT60TR13C",
        "stream": "/captures/live/radar",
        "files": "radar binary",
        "driver_available": driver_available,
        "service_active": service_active,
        "error": error,
    }


def detect_sensor_inventory() -> List[Dict[str, Any]]:
    return [
        detect_dreamhat_radar(),
        detect_usb_camera(),
        detect_esp32_csi(),
        detect_sense_hat(),
    ]
