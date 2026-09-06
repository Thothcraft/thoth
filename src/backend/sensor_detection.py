"""Attached sensor detection for Thoth data collection."""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List

from .config import Config


THOTH_ROOT = Path(__file__).resolve().parents[2]
MMW_RELEASE = THOTH_ROOT / "WS" / "MMW-HAT" / "MMW-HAT-Release"
RADAR_CONFIG_DIR = MMW_RELEASE / "radar_config" / "config_3rx_3m"
ESP32_USB_VIDS = {0x303A, 0x10C4, 0x1A86, 0x0403}
ESP32_SERIAL_HINTS = (
    "esp32", "espressif", "cp210", "silicon labs", "ch340", "ch341", "wch", "ftdi",
)


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


def likely_csi_serial_candidates() -> List[str]:
    """Return serial ports likely to be ESP32 CSI receivers.

    Prefer Espressif USB VID/metadata so unrelated USB serial adapters are not
    counted as CSI receivers.  The generic list remains a compatibility
    fallback for boards whose USB descriptors are unavailable.
    """
    esp32_ports: List[str] = []
    try:
        import serial.tools.list_ports as list_ports

        for port in list_ports.comports():
            metadata = " ".join((
                str(port.manufacturer or ""), str(port.description or ""),
                str(getattr(port, "product", "") or ""), str(getattr(port, "hwid", "") or ""),
            )).lower()
            if port.vid in ESP32_USB_VIDS or any(hint in metadata for hint in ESP32_SERIAL_HINTS):
                if port.device not in esp32_ports:
                    esp32_ports.append(port.device)
    except Exception:
        pass
    # Descriptor strings vary across ESP32 dev boards. USB serial device names
    # are still a strong signal, while onboard UARTs such as ttyAMA are not.
    for device in serial_candidates():
        if Path(device).name.startswith(("ttyACM", "ttyUSB")) and device not in esp32_ports:
            esp32_ports.append(device)
    return sorted(esp32_ports)


def csi_serial_candidates() -> List[str]:
    """Return only ports whose USB identity is consistent with an ESP32."""
    return likely_csi_serial_candidates()


def _usb_parent_for_video(device: str) -> Path | None:
    sysfs_device = Path("/sys/class/video4linux") / Path(device).name / "device"
    try:
        resolved = sysfs_device.resolve(strict=True)
    except OSError:
        return None
    return next((parent for parent in (resolved, *resolved.parents) if (parent / "idVendor").exists()), None)


def usable_usb_camera_devices() -> List[str]:
    """Return verified USB UVC capture nodes, excluding metadata/loopback nodes."""
    verified: List[str] = []
    v4l2_ctl = shutil.which("v4l2-ctl")
    for device in sorted(glob.glob("/dev/video*")):
        if _usb_parent_for_video(device) is None:
            continue
        if not v4l2_ctl:
            # Do not claim online when query capabilities cannot be verified.
            continue
        try:
            probe = subprocess.run(
                [v4l2_ctl, "--device", device, "--all"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        text = (probe.stdout or "").lower()
        is_uvc = "uvcvideo" in text or "bus info" in text and "usb-" in text
        is_capture = "video capture" in text and "video output" not in text
        if probe.returncode == 0 and is_uvc and is_capture:
            verified.append(device)
    return verified


def detect_usb_camera() -> Dict[str, Any]:
    requested = Config.CAPTURE_CAMERA_DEVICE
    nodes = sorted(glob.glob("/dev/video*"))
    devices = usable_usb_camera_devices()
    source = requested if requested in devices else (devices[0] if devices else None)
    return {
        "name": "USB Camera",
        "key": "usb_camera",
        "online": bool(source),
        "available": bool(source),
        "source": source,
        "stream": None,
        "files": "JPEG frames in capture.npz",
        "devices": devices,
        "detected_nodes": nodes,
        "error": None if source else ("video nodes are not usable USB capture devices" if nodes else "no USB camera detected"),
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
        "files": "IMU/environment samples in capture.npz",
        "error": error,
    }


def detect_esp32_csi() -> Dict[str, Any]:
    candidates = csi_serial_candidates()
    count = len(candidates)
    return {
        "name": f"csix{count}" if count > 1 else "CSI",
        "key": "esp32_csi",
        "online": bool(candidates),
        "source": candidates[0] if candidates else "USB serial",
        "stream": None,
        "files": "receiver samples in capture.npz",
        "devices": candidates,
        "receiver_count": count,
        "display_key": f"csix{count}" if count > 1 else "csi",
    }


def detect_dreamhat_radar() -> Dict[str, Any]:
    devices = sorted(glob.glob("/dev/spidev*"))
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
    spi_available = bool(devices)
    # The continuous collector is the sole owner of the radar's SPI and GPIO
    # lines. Probing the chip from the web process races the collector and can
    # leave GPIO12/GPIO25 busy, so inventory detection is intentionally passive.
    chip_online = False
    try:
        data_root = Path(Config.CAPTURE_DATA_DIR).expanduser()
        manifests = sorted(data_root.rglob("manifest.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        if manifests and time.time() - manifests[0].stat().st_mtime < 180:
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            radar = (manifest.get("outputs") or {}).get("radar") or {}
            chip_online = int(radar.get("sample_count") or 0) > 0
    except Exception:
        chip_online = False
    error = None if chip_online else "no recent radar samples"

    online = chip_online
    source = "BGT60TR13C"
    if chip_online:
        source = "BGT60TR13C chip detected"
    elif service_active:
        source = "collector running; radar unverified"
    elif spi_available:
        source = devices[0]

    return {
        "name": "DreamHAT+ Radar",
        "key": "dreamhat_radar",
        "online": online,
        "available": online,
        "source": source,
        "stream": None,
        "files": "radar samples in capture.npz",
        "devices": devices,
        "driver_available": driver_available,
        "spi_available": spi_available,
        "service_active": service_active,
        "chip_online": chip_online,
        "error": error,
    }


def detect_sensor_inventory() -> List[Dict[str, Any]]:
    return [
        detect_dreamhat_radar(),
        detect_usb_camera(),
        detect_esp32_csi(),
        detect_sense_hat(),
    ]
