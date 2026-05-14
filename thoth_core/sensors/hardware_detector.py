"""Hardware sensor detection helpers for Thoth.

The probes in this module avoid long-running reads and keep hardware access
best-effort so the dashboard can boot even when a sensor is missing or busy.
"""

import glob
import os
import sys
import time
from typing import Any, Dict, List, Optional


def _base_result(sensor_type: str, name: str, available: bool, **extra: Any) -> Dict[str, Any]:
    result = {
        "sensor_type": sensor_type,
        "name": name,
        "available": available,
        "error": None,
    }
    result.update(extra)
    return result


def ensure_system_dist_packages() -> None:
    """Expose Debian/Raspberry Pi Python packages inside a local venv."""
    for path in ("/usr/lib/python3/dist-packages", "/usr/lib/python3.11/dist-packages"):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)


def detect_sense_hat() -> Dict[str, Any]:
    """Detect a Sense HAT and verify that IMU data can be read."""
    try:
        ensure_system_dist_packages()
        from sense_hat import SenseHat

        sense = SenseHat()
        orientation = sense.get_orientation_degrees()
        acceleration = sense.get_accelerometer_raw()
        compass = sense.get_compass_raw()
        return _base_result(
            "sensehat_imu",
            "Sense HAT IMU",
            True,
            sample={
                "orientation": orientation,
                "acceleration": acceleration,
                "compass": compass,
            },
        )
    except Exception as exc:
        result = _base_result("sensehat_imu", "Sense HAT IMU", False)
        result["error"] = str(exc)
        return result


def _serial_candidates() -> List[str]:
    ports: List[str] = []
    try:
        import serial.tools.list_ports as list_ports

        ports.extend(p.device for p in list_ports.comports())
    except Exception:
        pass

    ports.extend(glob.glob("/dev/ttyUSB*"))
    ports.extend(glob.glob("/dev/ttyACM*"))
    ports.extend(glob.glob("/dev/cu.usbserial*"))
    ports.extend(glob.glob("/dev/cu.usbmodem*"))

    seen = set()
    unique_ports = []
    for port in ports:
        if port not in seen and os.path.exists(port):
            seen.add(port)
            unique_ports.append(port)
    return unique_ports


def detect_esp32_csi(baud: int = 115200, timeout: float = 0.25) -> Dict[str, Any]:
    """Detect an ESP32 CSI receiver and prefer ports that emit CSI_DATA."""
    ports = _serial_candidates()
    if not ports:
        return _base_result(
            "wifi_csi",
            "ESP32 CSI Receiver",
            False,
            ports=[],
            error="No serial ports found",
        )

    try:
        import serial
    except Exception as exc:
        return _base_result(
            "wifi_csi",
            "ESP32 CSI Receiver",
            False,
            ports=ports,
            error=f"pyserial unavailable: {exc}",
        )

    first_openable: Optional[str] = None
    errors: Dict[str, str] = {}
    for port in ports:
        try:
            with serial.Serial(port, baudrate=baud, timeout=timeout) as ser:
                if first_openable is None:
                    first_openable = port
                deadline = time.time() + timeout
                while time.time() < deadline:
                    line = ser.readline()
                    if not line:
                        continue
                    text = line.decode("utf-8", errors="ignore").strip()
                    if text.startswith("CSI_DATA,"):
                        return _base_result(
                            "wifi_csi",
                            "ESP32 CSI Receiver",
                            True,
                            port=port,
                            ports=ports,
                            verified=True,
                        )
        except Exception as exc:
            errors[port] = str(exc)

    if first_openable:
        return _base_result(
            "wifi_csi",
            "ESP32 CSI Receiver",
            True,
            port=first_openable,
            ports=ports,
            verified=False,
            error="Serial port detected; CSI stream not observed during quick probe",
        )

    return _base_result(
        "wifi_csi",
        "ESP32 CSI Receiver",
        False,
        ports=ports,
        errors=errors,
        error="No ESP32 CSI serial receiver could be opened",
    )


def detect_usb_camera(max_indexes: int = 6) -> Dict[str, Any]:
    """Detect a USB/OpenCV-compatible camera."""
    device_nodes = sorted(glob.glob("/dev/video*"))
    try:
        import cv2
    except Exception as exc:
        return _base_result(
            "camera",
            "USB Camera",
            bool(device_nodes),
            devices=device_nodes,
            error=f"OpenCV unavailable: {exc}",
        )

    detected = []
    for index in range(max_indexes):
        cap = cv2.VideoCapture(index)
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                info = {"index": index}
                if ok and frame is not None:
                    info["width"] = int(frame.shape[1])
                    info["height"] = int(frame.shape[0])
                detected.append(info)
        finally:
            cap.release()

    return _base_result(
        "camera",
        "USB Camera",
        bool(detected or device_nodes),
        devices=device_nodes,
        cameras=detected,
        error=None if detected or device_nodes else "Camera not detected",
    )


def detect_dreamhat_placeholder() -> Dict[str, Any]:
    """Placeholder probe for future DreamHAT support."""
    return _base_result(
        "dreamhat",
        "DreamHAT",
        False,
        error="DreamHAT detector not implemented yet",
    )


def detect_sensors() -> List[Dict[str, Any]]:
    """Return all known smart sensor detections."""
    return [
        detect_sense_hat(),
        detect_esp32_csi(),
        detect_usb_camera(),
        detect_dreamhat_placeholder(),
    ]
