"""Thoth Flask Backend Application.

This module provides the main Flask application for the Thoth device,
including REST API endpoints and WebSocket support for real-time data streaming.
"""

import os
import sys
import json
import io
import math
import html
import subprocess
import threading
import time
import logging
import uuid
import socket
import psutil
import platform
import netifaces
import requests
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache
from apscheduler.schedulers.background import BackgroundScheduler
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from flask import (
    Flask, jsonify, request, render_template,
    redirect, url_for, flash, session, send_from_directory, abort, send_file, Response, after_this_request
)
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import requests
from werkzeug.security import generate_password_hash, check_password_hash

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import Sense HAT; do not substitute mock sensor data.
try:
    from sense_hat import SenseHat
    sense = SenseHat()
    SENSE_HAT_AVAILABLE = True
    print("Using real Sense HAT")
except (ImportError, OSError):
    print("Sense HAT not found")
    sense = None
    SENSE_HAT_AVAILABLE = False

from backend.config import Config, BUTTON_ACTIONS, SENSOR_CONFIG
from backend.models import SensorReading, SystemStatus, ButtonConfig, UploadResult
from backend.device_manager import DeviceManager
from backend.auth_manager import AuthManager
from backend.capture_manager import (
    list_minutes,
    get_minute,
    capture_files,
    current_minute,
    minute_summary,
    cleanup_old_minutes,
    zip_minute_folder,
    preview_text,
)

THOTH_ROOT = Path(__file__).resolve().parents[2]
MMW_RELEASE = THOTH_ROOT / 'WS' / 'MMW-HAT' / 'MMW-HAT-Release'
if str(MMW_RELEASE) not in sys.path:
    sys.path.append(str(MMW_RELEASE))

try:
    import numpy as np
except Exception:  # pragma: no cover - import may fail on minimal installs
    np = None

try:
    from utility.mmw_cube_proc_v0 import CubeProcessor
except Exception:  # pragma: no cover - import may fail on minimal installs
    CubeProcessor = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

RADAR_PLOTS = ('range-doppler', 'azimuth-range', 'azimuth-doppler')
RADAR_PLOT_AXES = {
    'range-doppler': ('Range', 'Doppler'),
    'azimuth-range': ('Azimuth', 'Range'),
    'azimuth-doppler': ('Azimuth', 'Doppler'),
}
RADAR_CONFIG_DIR = MMW_RELEASE / 'radar_config' / 'config_3rx_3m'

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config.from_object(Config)

# Register blueprints
from backend.routes import files as files_bp
app.register_blueprint(files_bp.bp)

# Initialize file manager
from backend.file_manager import file_manager

# Add request logging
@app.before_request
def log_request():
    logger.info(f"Request: {request.method} {request.path} - {request.remote_addr}")

@app.after_request
def log_response(response):
    logger.info(f"Response: {request.method} {request.path} - {response.status_code}")
    return response

# Add current date to all templates
@app.context_processor
def inject_now():
    return {'now': datetime.utcnow()}

# Initialize scheduler
device_scheduler = BackgroundScheduler()
app.secret_key = Config.SECRET_KEY
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Ensure required directories exist
os.makedirs(Config.CONFIG_DIR, exist_ok=True)
os.makedirs(Config.LOGS_DIR, exist_ok=True)
os.makedirs(Config.CAPTURE_DATA_DIR, exist_ok=True)

# Initialize managers
auth_manager = AuthManager(Config)
device_manager = DeviceManager(Config)

# Global state
collection_active = False
wifi_manager = None

# Mock user for local authentication (in production, use a proper user database)
USERS = {
    'admin': {
        'password': generate_password_hash('admin123'),
        'role': 'admin'
    },
    'user': {
        'password': generate_password_hash('password123'),
        'role': 'user'
    }
}

def scan_wifi_networks():
    """Scan for available WiFi networks."""
    networks = []

    try:
        if platform.system() == 'Windows':
            # Use netsh on Windows to scan for networks
            result = subprocess.run(
                ['netsh', 'wlan', 'show', 'networks', 'mode=bssid'],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                current_ssid = None
                current_secure = False
                current_signal = 0

                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if line.startswith('SSID') and ':' in line and 'BSSID' not in line:
                        # Save previous network if exists
                        if current_ssid:
                            networks.append({
                                'ssid': current_ssid,
                                'secure': current_secure,
                                'signal': current_signal
                            })
                        # Parse new SSID
                        current_ssid = line.split(':', 1)[1].strip()
                        current_secure = False
                        current_signal = 0
                    elif 'Authentication' in line and ':' in line:
                        auth = line.split(':', 1)[1].strip()
                        current_secure = auth.lower() != 'open'
                    elif 'Signal' in line and ':' in line:
                        try:
                            signal_str = line.split(':', 1)[1].strip().replace('%', '')
                            current_signal = int(signal_str)
                        except ValueError:
                            current_signal = 50

                # Add last network
                if current_ssid:
                    networks.append({
                        'ssid': current_ssid,
                        'secure': current_secure,
                        'signal': current_signal
                    })
        else:
            # Linux - use iwlist or nmcli
            result = subprocess.run(
                ['iwlist', 'wlan0', 'scan'],
                capture_output=True, text=True, timeout=15
            )
            # Parse Linux output (simplified)
            for line in result.stdout.split('\n'):
                if 'ESSID:' in line:
                    ssid = line.split('ESSID:')[1].strip().strip('"')
                    if ssid:
                        networks.append({'ssid': ssid, 'secure': True, 'signal': 50})
    except Exception as e:
        logger.error(f"Error scanning WiFi networks: {e}")

    # Remove duplicates and empty SSIDs
    seen = set()
    unique_networks = []
    for net in networks:
        if net['ssid'] and net['ssid'] not in seen:
            seen.add(net['ssid'])
            unique_networks.append(net)

    return unique_networks if unique_networks else []

def get_system_uptime() -> str:
    """Get system uptime in a human-readable format."""
    try:
        if platform.system() == 'Windows':
            # Use psutil for Windows
            uptime_seconds = time.time() - psutil.boot_time()
            return str(timedelta(seconds=int(uptime_seconds)))
        else:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.readline().split()[0])
                return str(timedelta(seconds=uptime_seconds)).split('.')[0]
    except Exception:
        return "unknown"


def detect_sensor_inventory() -> List[Dict[str, Any]]:
    """Detect available sensors and recording sources."""
    minutes = list_minutes()
    latest = minutes[0] if minutes else None
    current_dir = current_minute()
    current_files = capture_files(current_dir) if current_dir else {}

    def _has_serial_device() -> bool:
        candidates = []
        for pattern in ("/dev/ttyACM", "/dev/ttyUSB"):
            for idx in range(10):
                path = f"{pattern}{idx}"
                if os.path.exists(path):
                    candidates.append(path)
        serial_dir = Path("/dev/serial/by-id")
        if serial_dir.exists():
            for item in serial_dir.iterdir():
                candidates.append(str(item))
        return bool(candidates)

    return [
        {
            "name": "DreamHat Radar",
            "key": "dreamhat_radar",
            "online": bool(current_files.get("radar") and current_files["radar"].exists()) or subprocess.run(
                ["systemctl", "is-active", "thoth-collector"],
                capture_output=True,
                text=True,
            ).stdout.strip() == "active",
            "source": "BGT60TR13C",
            "stream": "/captures/live/radar",
            "files": "radar binary",
        },
        {
            "name": "USB Camera",
            "key": "usb_camera",
            "online": os.path.exists(Config.CAPTURE_CAMERA_DEVICE),
            "source": Config.CAPTURE_CAMERA_DEVICE,
            "stream": "/captures/live/video",
            "files": "mp4 video",
        },
        {
            "name": "ESP32 CSI",
            "key": "esp32_csi",
            "online": _has_serial_device() or bool(
                current_files.get("csi_timestamped")
                and current_files["csi_timestamped"].exists()
            ),
            "source": "USB serial",
            "stream": "/captures/live/csi",
            "files": "csv/jsonl",
        },
        {
            "name": "Sense HAT",
            "key": "sense_hat",
            "online": SENSE_HAT_AVAILABLE,
            "source": "GPIO / I2C",
            "stream": None,
            "files": "imu json",
        },
    ]


@lru_cache(maxsize=1)
def _radar_setting_path() -> Optional[Path]:
    if not RADAR_CONFIG_DIR.exists():
        return None
    matches = sorted(RADAR_CONFIG_DIR.glob('BGT60TR13C_settings_*.json'))
    return matches[0] if matches else None


@lru_cache(maxsize=1)
def _radar_setting() -> Optional[Dict[str, Any]]:
    setting_path = _radar_setting_path()
    if not setting_path:
        return None
    try:
        with open(setting_path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except Exception as exc:
        logger.error(f"Failed to load radar setting {setting_path}: {exc}")
        return None


def _iter_radar_frames(path: Path):
    with open(path, 'rb') as handle:
        while True:
            version_bytes = handle.read(4)
            if not version_bytes or len(version_bytes) < 4:
                break
            version = int.from_bytes(version_bytes, byteorder='little', signed=False)
            if version != 0:
                break
            seq = int.from_bytes(handle.read(4), byteorder='little', signed=False)
            data_len = int.from_bytes(handle.read(4), byteorder='little', signed=False)
            raw_data = handle.read(data_len)
            if len(raw_data) != data_len:
                break
            yield seq, raw_data


def _split_csv_line(line: str) -> List[str]:
    cells: List[str] = []
    current: List[str] = []
    in_quotes = False
    idx = 0
    while idx < len(line):
        char = line[idx]
        if char == '"':
            if in_quotes and idx + 1 < len(line) and line[idx + 1] == '"':
                current.append('"')
                idx += 2
                continue
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            cells.append(''.join(current))
            current = []
        else:
            current.append(char)
        idx += 1
    cells.append(''.join(current))
    return cells


def _parse_csi_average_series(path: Path, limit: int = 180) -> List[float]:
    if not path.exists():
        return []

    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except Exception:
        return []

    if not lines:
        return []

    def _mean_from_payload(payload: str) -> Optional[float]:
        payload = payload.strip().strip('[]')
        if not payload:
            return None
        try:
            values = [float(value) for value in payload.split(',') if value.strip()]
        except ValueError:
            return None
        if len(values) < 2:
            return None
        mags: List[float] = []
        for idx in range(0, len(values) - 1, 2):
            imag = values[idx]
            real = values[idx + 1]
            mags.append(math.sqrt((real * real) + (imag * imag)))
        if not mags:
            return None
        return sum(mags) / len(mags)

    series: List[float] = []
    first = lines[0]
    if first.startswith('{'):
        for line in lines:
            try:
                row = json.loads(line)
            except Exception:
                continue
            raw = str(row.get('line') or row.get('raw') or row.get('raw_csi_line') or '').strip()
            if not raw.startswith('CSI_DATA,'):
                continue
            cells = _split_csv_line(raw)
            mean = _mean_from_payload(cells[-1] if cells else '')
            if mean is not None:
                series.append(mean)
    else:
        header = _split_csv_line(first)
        data_index = header.index('data') if 'data' in header else -1
        if data_index < 0:
            return []
        for line in lines[1:]:
            if not line.startswith('CSI_DATA,'):
                continue
            cells = _split_csv_line(line)
            if len(cells) <= data_index:
                continue
            mean = _mean_from_payload(cells[data_index])
            if mean is not None:
                series.append(mean)

    return series[-limit:]


def _build_csi_svg(points: List[float]) -> str:
    width = 960
    height = 260
    pad = 18
    if not points:
        return (
            f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
            f'<rect width="{width}" height="{height}" rx="16" fill="#020617"/>'
            '<text x="20" y="40" fill="#94a3b8" font-size="14">Waiting for CSI samples...</text>'
            '</svg>'
        )

    min_v = min(points)
    max_v = max(points)
    span = max_v - min_v or 1.0
    step = (width - pad * 2) / max(1, len(points) - 1)
    polyline = ' '.join(
        f'{pad + idx * step:.2f},{height - pad - ((value - min_v) / span) * (height - pad * 2):.2f}'
        for idx, value in enumerate(points)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{width}" height="{height}" rx="16" fill="#020617"/>'
        f'<polyline fill="none" stroke="#60a5fa" stroke-width="2.5" points="{html.escape(polyline)}"/>'
        '<text x="20" y="28" fill="#cbd5e1" font-size="14">Average CSI amplitude</text>'
        f'<text x="20" y="48" fill="#94a3b8" font-size="11">Packets: {len(points)}</text>'
        '</svg>'
    )


def _current_live_video_path() -> Optional[Path]:
    minute_dir = current_minute()
    if not minute_dir:
        return None
    files = capture_files(minute_dir)
    video_path = files.get('video')
    if video_path and video_path.exists():
        return video_path
    return None


def _csi_plot_payload(path: Path, limit: int = 180) -> Dict[str, Any]:
    points = _parse_csi_average_series(path, limit=limit)
    return {
        'points': points,
        'sample_count': len(points),
        'title': 'Average CSI amplitude',
        'x_label': 'Packet',
        'y_label': 'Average magnitude',
        'updated': datetime.utcnow().isoformat(),
    }


def _radar_plot_payload(radar_path: Path, plot: str) -> Dict[str, Any]:
    if np is None or CubeProcessor is None:
        raise RuntimeError('Radar plotting dependencies are unavailable')

    setting = _radar_setting()
    if not setting:
        raise RuntimeError('Radar settings not found')

    axis_names = RADAR_PLOT_AXES.get(plot)
    if not axis_names:
        raise RuntimeError(f'Unsupported radar plot kind: {plot}')

    mmw_proc = CubeProcessor(setting, num_azimuth_bin=16, num_elevation_bin=16)
    for _seq, raw_data in _iter_radar_frames(radar_path):
        mmw_proc.process_raw_data(raw_data)

    if mmw_proc.data_cube_fft is None:
        raise RuntimeError('No radar frames available')

    img = mmw_proc.vis_2d(axis_names[0], axis_names[1])
    img = np.log10(np.maximum(img, 1e-9))

    x_name = axis_names[1].lower()
    y_name = axis_names[0].lower()
    x_values = np.asarray(mmw_proc.proc_param.get(f'{x_name}_bin', []), dtype=float).tolist()
    y_values = np.asarray(mmw_proc.proc_param.get(f'{y_name}_bin', []), dtype=float).tolist()

    return {
        'plot': plot,
        'title': f'{axis_names[0]} vs {axis_names[1]}',
        'x_label': axis_names[1],
        'y_label': axis_names[0],
        'x': x_values,
        'y': y_values,
        'z': img.tolist(),
        'updated': datetime.utcnow().isoformat(),
    }


def _render_live_radar_png(radar_path: Path, plot: str) -> bytes:
    payload = _radar_plot_payload(radar_path, plot)
    img = np.asarray(payload['z'], dtype=float)

    try:
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - handled at runtime
        raise RuntimeError(f'matplotlib unavailable: {exc}') from exc

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    ax.imshow(img, aspect='auto', cmap='viridis')
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return buf.getvalue()


def _cached_radar_plot_path(radar_path: Path, plot: str, cache_name: str) -> Path:
    cache_dir = Path(tempfile.gettempdir()) / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f'{radar_path.stem}-{plot}.png'
    radar_mtime = radar_path.stat().st_mtime
    if not cache_path.exists() or cache_path.stat().st_mtime < radar_mtime:
        png_bytes = _render_live_radar_png(radar_path, plot)
        with open(cache_path, 'wb') as handle:
            handle.write(png_bytes)
    return cache_path


def get_capture_overview() -> Dict[str, Any]:
    """Summarize the capture directory."""
    minutes = list_minutes()
    latest = minutes[0] if minutes else None
    return {
        "capture_dir": Config.CAPTURE_DATA_DIR,
        "keep_minutes": Config.CAPTURE_KEEP_MINUTES,
        "minute_count": len(minutes),
        "latest_minute": latest,
    }

# Global state
collection_active = False
wifi_manager = None

def get_system_status(update_remote: bool = True) -> SystemStatus:
    """Get current system status and optionally update the Brain server.

    Args:
        update_remote: If True, update the status on the Brain server

    Returns:
        SystemStatus: Current system status
    """
    try:
        is_windows = platform.system() == 'Windows'

        # Check WiFi connection (platform-aware)
        try:
            if is_windows:
                wifi_connected = subprocess.run(
                    ["ping", "-n", "1", "8.8.8.8"],
                    capture_output=True, timeout=5
                ).returncode == 0
            else:
                wifi_connected = subprocess.run(
                    ["ping", "-c1", "8.8.8.8"],
                    capture_output=True, timeout=5
                ).returncode == 0
        except Exception:
            wifi_connected = False

        # Check collection status on Linux.
        collection_status = False
        if not is_windows:
            try:
                collection_status = subprocess.run(
                    ["systemctl", "is-active", "thoth-collector"],
                    capture_output=True, text=True
                ).stdout.strip() == "active"
            except Exception:
                collection_status = False

        # Get battery level.
        battery_level = None
        try:
            battery = psutil.sensors_battery()
            if battery:
                battery_level = int(battery.percent)
        except Exception:
            battery_level = None

        # Get CPU temperature
        cpu_temp = None
        if not is_windows:
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    cpu_temp = float(f.read().strip()) / 1000.0
            except Exception:
                pass  # Temperature not available on this system

        # Get IP address
        ip_address = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
            s.close()
        except Exception as e:
            logger.debug(f"Could not get IP address: {e}")

        # Get disk usage
        disk_usage = None
        try:
            disk = psutil.disk_usage('/')
            disk_usage = {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent,
                'total_gb': disk.total / (1024 ** 3),
                'used_gb': disk.used / (1024 ** 3),
                'free_gb': disk.free / (1024 ** 3),
                'percent_used': disk.percent
            }
        except Exception:
            pass

        # Get uptime
        uptime_output = get_system_uptime()

        # Create status object
        status = SystemStatus(
            status="ok",
            battery_level=battery_level,
            wifi_connected=wifi_connected,
            ap_mode=not wifi_connected,
            collection_active=collection_status,
            uptime=uptime_output,
            temperature=cpu_temp,
            disk_usage=disk_usage,
            ip_address=ip_address
        )

        # Update device manager status
        if update_remote:
            try:
                device_manager.update_status({
                    'battery_level': battery_level,
                    'wifi_connected': wifi_connected,
                    'collection_active': collection_status,
                    'online': True,
                    'ip_address': ip_address,
                    'temperature': cpu_temp,
                    'disk_usage': disk_usage
                })
            except Exception as e:
                logger.error(f"Error updating device status: {e}")

        return status

    except Exception as e:
        logger.error(f"Error getting system status: {e}", exc_info=True)
        return SystemStatus(status="error", error=str(e))

def tail_sensor_data():
    """Background task to tail sensor data and emit via WebSocket."""
    sensor_file = Config.SENSOR_DATA_FILE

    while True:
        try:
            if os.path.exists(sensor_file):
                with open(sensor_file, 'r') as f:
                    # Go to end of file
                    f.seek(0, 2)
                    while True:
                        line = f.readline()
                        if line:
                            try:
                                data = json.loads(line.strip())
                                socketio.emit('imu_data', data)
                            except json.JSONDecodeError:
                                continue
                        else:
                            time.sleep(0.1)
            else:
                time.sleep(1)
        except Exception as e:
            print(f"Error in sensor data tail: {e}")
            time.sleep(5)

def get_device_info() -> Dict[str, Any]:
    """Get detailed information about the device."""
    try:
        # Get network interfaces and MAC addresses
        interfaces = {}
        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            if netifaces.AF_LINK in addrs and addrs[netifaces.AF_LINK]:
                mac = addrs[netifaces.AF_LINK][0].get('addr')
                if mac and mac != '00:00:00:00:00:00':
                    interfaces[iface] = mac

        # Get system information
        system_info = {
            'system': platform.system(),
            'node': platform.node(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'cpu_count': os.cpu_count(),
            'memory': psutil.virtual_memory()._asdict(),
            'disk_usage': psutil.disk_usage('/')._asdict(),
            'network_interfaces': interfaces,
            'hostname': socket.gethostname(),
            'ip_address': socket.gethostbyname(socket.gethostname()),
            'python_version': platform.python_version(),
            'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }
        return system_info
    except Exception as e:
        logger.error(f"Error getting device info: {e}")
        return {}

def register_device_periodically():
    """Register the current Thoth device with Brain."""
    try:
        if not getattr(Config, 'BRAIN_SERVER_URL', None):
            logger.warning("Brain server URL not configured, skipping device registration")
            return False

        auth_token = getattr(Config, 'USER_AUTH_TOKEN', None)
        if not auth_token:
            logger.debug("No authenticated user token available, skipping device registration")
            return False

        success, message = device_manager.register_device(auth_token.strip())
        if success:
            logger.info(message)
        else:
            logger.warning(message)
        return success
    except Exception as e:
        logger.error(f"Unexpected error in device registration: {str(e)}", exc_info=True)
        return False

# Start background tasks
socketio.start_background_task(tail_sensor_data)

# Start device registration scheduler (10 second interval for responsive uploads)
device_scheduler.add_job(
    register_device_periodically,
    'interval',
    seconds=10,
    id='device_registration',
    replace_existing=True
)
device_scheduler.add_job(
    lambda: cleanup_old_minutes(Config.CAPTURE_KEEP_MINUTES),
    'interval',
    minutes=10,
    id='capture_cleanup',
    replace_existing=True
)
device_scheduler.start()

# Load registration info if available
device_manager.load_registration_info()
cleanup_old_minutes(Config.CAPTURE_KEEP_MINUTES)

# WiFi credentials storage file
WIFI_CREDENTIALS_FILE = os.path.join(Config.CONFIG_DIR, 'wifi_credentials.json')
WIFI_CONFIGURED_FILE = os.path.join(Config.CONFIG_DIR, 'wifi_configured.flag')

def load_wifi_credentials():
    """Load saved WiFi credentials."""
    try:
        if os.path.exists(WIFI_CREDENTIALS_FILE):
            with open(WIFI_CREDENTIALS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading WiFi credentials: {e}")
    return {}

def save_wifi_credentials(ssid, password):
    """Save WiFi credentials for future use."""
    try:
        os.makedirs(Config.CONFIG_DIR, exist_ok=True)
        creds = load_wifi_credentials()
        creds[ssid] = password
        creds['_active_ssid'] = ssid  # Track the active network
        with open(WIFI_CREDENTIALS_FILE, 'w') as f:
            json.dump(creds, f)
        # Mark WiFi as configured
        with open(WIFI_CONFIGURED_FILE, 'w') as f:
            f.write(ssid)
        logger.info(f"WiFi credentials saved for {ssid}")
    except Exception as e:
        logger.error(f"Error saving WiFi credentials: {e}")

def is_wifi_configured():
    """Check if WiFi has been explicitly configured by user."""
    return os.path.exists(WIFI_CONFIGURED_FILE)

def get_configured_ssid():
    """Get the SSID of the configured WiFi network."""
    try:
        if os.path.exists(WIFI_CONFIGURED_FILE):
            with open(WIFI_CONFIGURED_FILE, 'r') as f:
                return f.read().strip()
    except Exception:
        pass
    return None

def clear_wifi_configuration():
    """Clear WiFi configuration (disconnect from network)."""
    try:
        if os.path.exists(WIFI_CONFIGURED_FILE):
            os.remove(WIFI_CONFIGURED_FILE)
            logger.info("WiFi configuration cleared")
    except Exception as e:
        logger.error(f"Error clearing WiFi configuration: {e}")

def check_wifi_connected():
    """Check if WiFi is connected AND configured."""
    # First check if WiFi has been explicitly configured
    if not is_wifi_configured():
        return False

    try:
        # Try to connect to Google DNS to check internet connectivity
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2)
        s.connect(("8.8.8.8", 80))
        s.close()
        return True
    except Exception:
        return False

# Routes
@app.route('/')
def index():
    """Serve the appropriate page based on authentication and registration status."""
    # If already authenticated, go to status
    if 'username' in session:
        return redirect(url_for('status'))

    # Render the setup page directly so the root path shows content.
    return setup()

@app.route('/setup')
def setup():
    """Show the setup page for WiFi and login."""
    # If already authenticated, go to status
    if 'username' in session:
        return redirect(url_for('status'))

    # Check if user wants to change WiFi
    change_wifi = request.args.get('change_wifi') == '1'
    if change_wifi:
        clear_wifi_configuration()

    wifi_connected = check_wifi_connected()
    available_networks = scan_wifi_networks()

    # Get current SSID from saved configuration
    current_ssid = get_configured_ssid()

    return render_template('setup.html',
                         wifi_connected=wifi_connected,
                         available_networks=available_networks,
                         current_ssid=current_ssid,
                         version=Config.VERSION)

@app.route('/api/wifi/scan', methods=['GET'])
def api_wifi_scan():
    """Scan for available WiFi networks."""
    try:
        networks = scan_wifi_networks()
        return jsonify({'status': 'success', 'networks': networks})
    except Exception as e:
        logger.error(f"Error scanning WiFi: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e), 'networks': []}), 500

def connect_wifi_raspberry_pi(ssid: str, password: str) -> tuple:
    """Connect to WiFi on Raspberry Pi using the connect-wifi.sh script."""
    try:
        script_path = os.path.join(os.path.dirname(__file__), '..', '..', 'setup', 'connect-wifi.sh')
        if os.path.exists(script_path):
            result = subprocess.run(
                ['sudo', script_path, ssid, password],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                # Extract IP from output
                ip_address = None
                for line in result.stdout.split('\n'):
                    if 'IP Address:' in line:
                        ip_address = line.split(':')[1].strip()
                return True, ip_address
            else:
                return False, result.stderr
        else:
            return False, "connect-wifi.sh not found"
    except Exception as e:
        return False, str(e)

@app.route('/api/wifi/connect', methods=['POST'])
def api_wifi_connect():
    """Connect to a WiFi network and save credentials."""
    try:
        data = request.get_json() or request.form
        ssid = data.get('ssid')
        password = data.get('password', '')

        if not ssid:
            return jsonify({'status': 'error', 'error': 'SSID is required'}), 400

        logger.info(f"Attempting to connect to WiFi: {ssid}")

        # Save credentials for future use
        save_wifi_credentials(ssid, password)

        # Store in session
        session['wifi_ssid'] = ssid

        # On Raspberry Pi, use the connect script
        if platform.system() == 'Linux' and os.path.exists('/etc/hostapd'):
            success, result = connect_wifi_raspberry_pi(ssid, password)
            if success:
                ip_address = result
                return jsonify({
                    'status': 'success',
                    'message': f'Connected to {ssid}',
                    'wifi_connected': True,
                    'ip_address': ip_address
                })
            else:
                return jsonify({
                    'status': 'error',
                    'error': f'Failed to connect: {result}'
                }), 500

        # On Windows/Mac dev, simulate success
        return jsonify({
            'status': 'success',
            'message': f'WiFi credentials saved for {ssid}',
            'wifi_connected': True,
            'ip_address': '127.0.0.1'  # Dev mode
        })

    except Exception as e:
        logger.error(f"Error connecting to WiFi: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/setup/login', methods=['POST'])
def api_setup_login():
    """Handle login from setup page."""
    try:
        data = request.get_json() or request.form
        username = data.get('username')
        password = data.get('user_password')

        if not username or not password:
            return jsonify({'status': 'error', 'error': 'Username and password are required'}), 400

        # Authenticate with the Brain server
        result = auth_manager.login(username, password)

        if result.get('success'):
            # Store user session
            session['user_id'] = result['user'].get('user_id')
            session['username'] = username
            session['token'] = result['token']

            # Store token globally for device registration
            Config.USER_AUTH_TOKEN = result['token']

            # Register the device
            success, message = device_manager.register_device(result['token'])

            if success:
                device_manager.start_heartbeat(Config.HEARTBEAT_INTERVAL)
                logger.info(f"Login successful for user: {username}, device registered")
            else:
                logger.warning(f"Device registration failed: {message}")

            return jsonify({
                'status': 'success',
                'message': 'Login successful',
                'redirect': url_for('status')
            })
        else:
            return jsonify({'status': 'error', 'error': 'Invalid username or password'}), 401

    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login and device registration."""
    # Redirect to status if already logged in
    if 'username' in session:
        return redirect(url_for('status'))

    if request.method == 'GET':
        return render_template('login.html', next=request.args.get('next', ''))

    # Handle POST request
    try:
        username = request.form.get('username')
        password = request.form.get('password')
        next_page = request.form.get('next', '')

        if not username or not password:
            flash('Username and password are required', 'error')
            return redirect(url_for('login', next=next_page))

        # Authenticate with the Brain server
        try:
            result = auth_manager.login(username, password)

            if result.get('success'):
                # Store user session
                session['user_id'] = result['user'].get('user_id')
                session['username'] = username
                session['token'] = result['token']

                # Store token globally for device registration
                Config.USER_AUTH_TOKEN = result['token']

                logger.info(f"Login successful for user: {username}")
                logger.info("Device registration will now use this user's token")

                # Show the token on success page
                return render_template('login_success.html',
                                     username=username,
                                     access_token=result['token'],
                                     user_info=result['user'])
            else:
                flash('Invalid username or password', 'error')

        except Exception as e:
            logger.error(f"Login error: {str(e)}", exc_info=True)
            flash(f'Login failed: {str(e)}', 'error')

    except Exception as e:
        logger.error(f"Login error: {str(e)}", exc_info=True)
        flash('An error occurred. Please try again.', 'error')

    return redirect(url_for('login'))

@app.route('/status')
def status():
    """Display the device status page."""
    # Check if user is logged in
    if 'username' not in session:
        return redirect(url_for('login', next=url_for('status')))

    try:
        # Get system status
        system_status = get_system_status()

        # Get device information
        device_info = device_manager.get_device_info()

        # Get disk usage
        disk_usage = psutil.disk_usage('/')

        # Get CPU temperature (Linux only)
        cpu_temp = None
        if platform.system() != 'Windows':
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    cpu_temp = float(f.read().strip()) / 1000.0  # Convert millidegrees to degrees
            except Exception:
                pass

        # Get IP address
        ip_address = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip_address = s.getsockname()[0]
            s.close()
        except Exception as e:
            logger.error(f"Error getting IP address: {e}")

        # Get list of available WiFi networks
        available_networks = scan_wifi_networks()
        capture_overview = get_capture_overview()
        sensors = detect_sensor_inventory()

        return render_template('status.html',
                            system_status=system_status,
                            device_info=device_info,
                            disk_usage=disk_usage,
                            cpu_temp=cpu_temp,
                            ip_address=ip_address,
                            username=session.get('username'),
                            available_networks=available_networks,
                            capture_overview=capture_overview,
                            sensors=sensors)

    except Exception as e:
        logger.error(f"Error in status route: {str(e)}", exc_info=True)
        flash('An error occurred while loading the status page.', 'error')
        return redirect(url_for('index'))

@app.route('/captures')
def captures():
    """Show the minute capture browser."""
    if 'username' not in session:
        return redirect(url_for('login', next=url_for('captures')))

    minutes = list_minutes()
    active = current_minute()
    return render_template(
        'captures.html',
        minutes=minutes,
        active_minute=active.name if active else None,
        username=session.get('username'),
    )


@app.route('/captures/<minute>')
def capture_detail(minute):
    """Show a single minute capture."""
    if 'username' not in session:
        return redirect(url_for('login', next=url_for('capture_detail', minute=minute)))

    minute_dir = get_minute(minute)
    if not minute_dir:
        abort(404, description='Minute folder not found')

    files = capture_files(minute_dir)
    detail = minute_summary(minute_dir)
    video_preview = f"/api/captures/{minute}/file/video" if files.get("video") else None
    radar_preview = preview_text(files.get("radar"), 12000)
    csi_preview = preview_text(files.get("csi_timestamped") or files.get("csi_csv"), 12000)
    serial_preview = preview_text(files.get("csi_serial"), 12000)

    return render_template(
        'capture_detail.html',
        minute=minute_dir.name,
        capture=minute_dir,
        minute_info=detail,
        files=files,
        video_url=video_preview,
        radar_data_urls=[url_for('api_capture_radar_data', minute=minute_dir.name, plot=plot) for plot in RADAR_PLOTS],
        csi_data_url=url_for('api_capture_csi_data', minute=minute_dir.name),
        radar_preview=radar_preview,
        csi_preview=csi_preview,
        serial_preview=serial_preview,
        username=session.get('username'),
        active_minute=current_minute().name if current_minute() else None,
    )


@app.route('/captures/<minute>/download')
def download_capture_minute(minute):
    """Download all files for a minute as a zip archive."""
    if 'username' not in session:
        return redirect(url_for('login', next=url_for('captures')))

    minute_dir = get_minute(minute)
    if not minute_dir:
        abort(404, description='Minute folder not found')

    zip_path = zip_minute_folder(minute_dir)
    @after_this_request
    def cleanup(response):
        try:
            if zip_path.exists():
                zip_path.unlink()
        except Exception:
            pass
        return response

    return send_file(
        zip_path,
        as_attachment=True,
        download_name=f"{minute_dir.name}.zip",
        mimetype='application/zip',
        conditional=True,
    )


@app.route('/api/captures', methods=['GET'])
def api_list_captures():
    """Return the capture minute index."""
    return jsonify({
        'status': 'success',
        'minutes': list_minutes(),
        'active_minute': current_minute().name if current_minute() else None,
        'capture_dir': Config.CAPTURE_DATA_DIR,
        'keep_minutes': Config.CAPTURE_KEEP_MINUTES,
    })


@app.route('/api/captures/<minute>', methods=['GET'])
def api_capture_detail(minute):
    """Return the metadata for a single capture minute."""
    minute_dir = get_minute(minute)
    if not minute_dir:
        return jsonify({'status': 'error', 'message': 'Minute folder not found'}), 404

    files = capture_files(minute_dir)
    detail = minute_summary(minute_dir)
    detail['files_on_disk'] = {key: str(path) if path else None for key, path in files.items()}
    return jsonify({'status': 'success', 'capture': detail})


@app.route('/api/captures/<minute>/file/<kind>', methods=['GET'])
def api_capture_file(minute, kind):
    """Serve an individual capture file."""
    minute_dir = get_minute(minute)
    if not minute_dir:
        return jsonify({'status': 'error', 'message': 'Minute folder not found'}), 404

    files = capture_files(minute_dir)
    mapping = {
        'video': files.get('video'),
        'radar': files.get('radar'),
        'csi_csv': files.get('csi_csv'),
        'csi_timestamped': files.get('csi_timestamped'),
        'csi_serial': files.get('csi_serial'),
        'manifest': files.get('manifest'),
        'log': files.get('video_log'),
    }
    file_path = mapping.get(kind)
    if not file_path or not file_path.exists():
        return jsonify({'status': 'error', 'message': f'File {kind} not found'}), 404

    mime = 'application/octet-stream'
    if file_path.suffix == '.mp4':
        mime = 'video/mp4'
    elif file_path.suffix == '.csv':
        mime = 'text/csv'
    elif file_path.suffix == '.json':
        mime = 'application/json'
    elif file_path.suffix == '.jsonl':
        mime = 'application/x-ndjson'
    elif file_path.suffix == '.log':
        mime = 'text/plain'

    return send_file(file_path, as_attachment=False, mimetype=mime, conditional=True)


@app.route('/api/captures/<minute>/csi/plot')
def api_capture_csi_plot(minute):
    """Render a CSI amplitude plot for a saved minute."""
    if request.method == 'HEAD':
        return Response(status=200)

    minute_dir = get_minute(minute)
    if not minute_dir:
        abort(404, description='Minute folder not found')

    files = capture_files(minute_dir)
    path = files.get('csi_csv') or files.get('csi_timestamped') or files.get('csi_serial')
    if not path or not path.exists():
        abort(404, description='No CSI data available')

    points = _parse_csi_average_series(path)
    svg = _build_csi_svg(points)
    return Response(svg, mimetype='image/svg+xml', headers={'Cache-Control': 'no-cache'})


@app.route('/api/captures/<minute>/csi/data')
def api_capture_csi_data(minute):
    """Return interactive CSI data for a saved minute."""
    if 'username' not in session:
        return redirect(url_for('login', next=url_for('captures')))

    minute_dir = get_minute(minute)
    if not minute_dir:
        abort(404, description='Minute folder not found')

    files = capture_files(minute_dir)
    path = files.get('csi_csv') or files.get('csi_timestamped') or files.get('csi_serial')
    if not path or not path.exists():
        abort(404, description='No CSI data available')

    return jsonify({'status': 'success', 'minute': minute, 'data': _csi_plot_payload(path)})


@app.route('/api/captures/<minute>/radar/plot/<plot>')
def api_capture_radar_plot(minute, plot):
    """Render a radar plot for a saved minute."""
    if request.method == 'HEAD':
        return Response(status=200)

    minute_dir = get_minute(minute)
    if not minute_dir:
        abort(404, description='Minute folder not found')

    plot = plot.lower()
    if plot not in RADAR_PLOTS:
        abort(404, description=f'Unsupported radar plot kind: {plot}')

    files = capture_files(minute_dir)
    radar_path = files.get('radar')
    if not radar_path or not radar_path.exists():
        abort(404, description='No radar data available')

    try:
        out_path = _cached_radar_plot_path(radar_path, plot, 'thoth-capture-radar')
    except Exception as exc:
        logger.exception(f"Failed to render saved radar plot {plot}: {exc}")
        abort(500, description=str(exc))

    return send_file(out_path, mimetype='image/png', conditional=False, max_age=0)


@app.route('/api/captures/<minute>/radar/data/<plot>')
def api_capture_radar_data(minute, plot):
    """Return interactive radar data for a saved minute."""
    if 'username' not in session:
        return redirect(url_for('login', next=url_for('captures')))

    minute_dir = get_minute(minute)
    if not minute_dir:
        abort(404, description='Minute folder not found')

    plot = plot.lower()
    if plot not in RADAR_PLOTS:
        abort(404, description=f'Unsupported radar plot kind: {plot}')

    files = capture_files(minute_dir)
    radar_path = files.get('radar')
    if not radar_path or not radar_path.exists():
        abort(404, description='No radar data available')

    try:
        payload = _radar_plot_payload(radar_path, plot)
    except Exception as exc:
        logger.exception(f"Failed to build radar data {plot}: {exc}")
        abort(500, description=str(exc))

    return jsonify({'status': 'success', 'minute': minute, 'data': payload})


@app.route('/api/captures/<minute>/upload', methods=['POST'])
def upload_capture_minute(minute):
    """Upload all files in a capture minute to Brain."""
    auth_token = getattr(Config, 'USER_AUTH_TOKEN', None)
    if not auth_token:
        return jsonify({'status': 'error', 'message': 'Not authenticated'}), 401

    minute_dir = get_minute(minute)
    if not minute_dir:
        return jsonify({'status': 'error', 'message': 'Minute folder not found'}), 404

    files = capture_files(minute_dir)
    uploaded = []
    errors = []
    import base64

    for key, path in files.items():
        if not path or not path.exists():
            continue
        try:
            with open(path, 'rb') as handle:
                content = base64.b64encode(handle.read()).decode('utf-8')
            suffix = path.suffix.lower()
            if suffix == '.mp4':
                content_type = 'video/mp4'
            elif suffix == '.csv':
                content_type = 'text/csv'
            elif suffix == '.json':
                content_type = 'application/json'
            elif suffix == '.jsonl':
                content_type = 'application/x-ndjson'
            else:
                content_type = 'application/octet-stream'
            headers = {
                'Authorization': f'Bearer {auth_token}',
                'Content-Type': 'application/json',
            }
            payload = {
                'filename': f'{minute_dir.name}_{path.name}',
                'content': content,
                'is_base64': True,
                'device_id': getattr(Config, 'DEVICE_ID', None),
                'content_type': content_type,
            }
            response = requests.post(
                f"{Config.BRAIN_SERVER_URL}/file/upload",
                json=payload,
                headers=headers,
                timeout=120,
            )
            if response.status_code in (200, 201):
                uploaded.append(path.name)
            else:
                errors.append(f"{path.name}: {response.status_code}")
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    return jsonify({
        'status': 'success' if not errors else 'partial',
        'minute': minute_dir.name,
        'uploaded': uploaded,
        'errors': errors,
    })


@app.route('/captures/live/<kind>')
def live_capture_stream(kind):
    """Render the live sensor view for the selected modality."""
    if 'username' not in session:
        return redirect(url_for('login', next=url_for('captures')))

    kind = kind.lower()
    if kind not in {'video', 'csi', 'radar'}:
        abort(404, description=f'Unsupported live stream kind: {kind}')

    minute_dir = current_minute()
    files = capture_files(minute_dir) if minute_dir else {}

    return render_template(
        'live_stream.html',
        kind=kind,
        active_minute=minute_dir.name if minute_dir else None,
        username=session.get('username'),
        video_url=url_for('api_live_capture_video'),
        csi_data_url=url_for('api_live_capture_csi_data'),
        radar_data_urls=[url_for('api_live_capture_radar_data', plot=plot) for plot in RADAR_PLOTS],
        has_video=bool(files.get('video') and files['video'].exists()) if minute_dir else False,
        has_csi=bool(files.get('csi_csv') or files.get('csi_timestamped') or files.get('csi_serial')),
        has_radar=bool(files.get('radar')),
    )


@app.route('/api/captures/live/video')
def api_live_capture_video():
    """Serve the current minute video file."""
    if 'username' not in session:
        return redirect(url_for('login', next=url_for('captures')))

    video_path = _current_live_video_path()
    if not video_path:
        abort(404, description='No live video available')

    return send_file(video_path, mimetype='video/mp4', conditional=True, max_age=0)


@app.route('/api/captures/live/csi')
def api_live_capture_csi():
    """Redirect to the live CSI plot view."""
    if 'username' not in session:
        return redirect(url_for('login', next=url_for('captures')))
    return redirect(url_for('live_capture_stream', kind='csi'))


@app.route('/api/captures/live/radar')
def api_live_capture_radar():
    """Redirect to the live radar plot view."""
    if 'username' not in session:
        return redirect(url_for('login', next=url_for('captures')))
    return redirect(url_for('live_capture_stream', kind='radar'))


@app.route('/api/captures/live/csi/data')
def api_live_capture_csi_data():
    """Return interactive live CSI data."""
    if 'username' not in session:
        return redirect(url_for('login', next=url_for('captures')))

    minute_dir = current_minute()
    if not minute_dir:
        abort(404, description='No live capture minute available')

    files = capture_files(minute_dir)
    path = files.get('csi_csv') or files.get('csi_timestamped') or files.get('csi_serial')
    if not path or not path.exists():
        abort(404, description='No CSI data available')

    return jsonify({'status': 'success', 'minute': minute_dir.name, 'data': _csi_plot_payload(path)})


@app.route('/api/captures/live/csi/plot')
def api_live_capture_csi_plot():
    """Render a live CSI amplitude SVG from the active minute."""
    if request.method == 'HEAD':
        return Response(status=200)

    if 'username' not in session:
        return redirect(url_for('login', next=url_for('captures')))

    minute_dir = current_minute()
    if not minute_dir:
        abort(404, description='No live capture minute available')

    files = capture_files(minute_dir)
    path = files.get('csi_csv') or files.get('csi_timestamped') or files.get('csi_serial')
    if not path or not path.exists():
        abort(404, description='No CSI data available')

    points = _parse_csi_average_series(path)
    svg = _build_csi_svg(points)
    return Response(svg, mimetype='image/svg+xml', headers={'Cache-Control': 'no-cache'})


@app.route('/api/captures/live/radar/data/<plot>')
def api_live_capture_radar_data(plot):
    """Return interactive live radar data."""
    if 'username' not in session:
        return redirect(url_for('login', next=url_for('captures')))

    plot = plot.lower()
    if plot not in RADAR_PLOTS:
        abort(404, description=f'Unsupported radar plot kind: {plot}')

    minute_dir = current_minute()
    if not minute_dir:
        abort(404, description='No live capture minute available')

    files = capture_files(minute_dir)
    radar_path = files.get('radar')
    if not radar_path or not radar_path.exists():
        abort(404, description='No radar data available')

    try:
        payload = _radar_plot_payload(radar_path, plot)
    except Exception as exc:
        logger.exception(f"Failed to build live radar data {plot}: {exc}")
        abort(500, description=str(exc))

    return jsonify({'status': 'success', 'minute': minute_dir.name, 'data': payload})


@app.route('/api/captures/live/radar/plot/<plot>')
def api_live_capture_plot(plot):
    """Render the current minute radar plot as PNG."""
    if request.method == 'HEAD':
        return Response(status=200)

    if 'username' not in session:
        return redirect(url_for('login', next=url_for('captures')))

    plot = plot.lower()
    if plot not in RADAR_PLOTS:
        abort(404, description=f'Unsupported radar plot kind: {plot}')

    minute_dir = current_minute()
    if not minute_dir:
        abort(404, description='No live capture minute available')

    files = capture_files(minute_dir)
    radar_path = files.get('radar')
    if not radar_path or not radar_path.exists():
        abort(404, description='No radar data available')

    try:
        out_path = _cached_radar_plot_path(radar_path, plot, 'thoth-live-radar')
    except Exception as exc:
        logger.exception(f"Failed to render live radar plot {plot}: {exc}")
        abort(500, description=str(exc))

    return send_file(out_path, mimetype='image/png', conditional=False, max_age=0)

@app.route('/logout')
def logout():
    """Log out the current user."""
    # Update device status
    try:
        device_manager.update_status({
            'online': False,
            'last_seen': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Error updating device status on logout: {e}")

    # Clear the user's auth token so device registration stops
    Config.USER_AUTH_TOKEN = None

    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/wifi/config', methods=['POST'])
def wifi_config():
    """Handle WiFi configuration from captive portal."""
    try:
        ssid = request.form.get('ssid')
        password = request.form.get('password')
        username = request.form.get('username')
        user_password = request.form.get('user_password')

        if not ssid:
            return jsonify({"error": "SSID required"}), 400

        # Configure the WiFi network
        logger.info(f"Configuring WiFi network: {ssid}")

        # In a real implementation, you would configure the WiFi here
        # For example:
        # configure_wifi(ssid, password)

        # If user credentials were provided, log in and register the device
        response_data = {'status': 'success', 'message': f'Successfully connected to {ssid}'}

        if username and user_password:
            try:
                result = auth_manager.login(username, user_password)

                if result.get('success'):
                    # Register the device with the Brain server
                    success, message = device_manager.register_device(result['token'])

                    if success:
                        # Start the heartbeat to send periodic status updates
                        device_manager.start_heartbeat(Config.HEARTBEAT_INTERVAL)
                        logger.info("Device registered and heartbeat started")

                        # Store user session
                        session['user_id'] = result['user'].get('user_id')
                        session['username'] = username
                        session['token'] = result['token']

                        # Update device status with initial information
                        device_manager.update_status({
                            'online': True,
                            'wifi_connected': True,
                            'ip_address': request.remote_addr,
                            'last_seen': datetime.utcnow().isoformat()
                        })

                        response_data['redirect'] = url_for('status')
                    else:
                        response_data['error'] = f'Device registration failed: {message}'
                else:
                    response_data['error'] = 'Invalid username or password'

            except Exception as e:
                logger.error(f"Login/registration error: {str(e)}", exc_info=True)
                response_data['error'] = 'An error occurred during login. Please try again.'

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error configuring WiFi: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/collection/<action>', methods=['POST'])
def collection_action(action):
    """Start or stop data collection."""
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401

    global collection_active

    if action == 'start':
        collection_active = True
        logger.info('Data collection started')
        return jsonify({'status': 'success', 'message': 'Collection started'})
    elif action == 'stop':
        collection_active = False
        logger.info('Data collection stopped')
        return jsonify({'status': 'success', 'message': 'Collection stopped'})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid action'}), 400

@app.route('/api/system/restart-service', methods=['POST'])
def restart_service():
    """Restart a system service."""
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401

    data = request.get_json()
    service = data.get('service')

    if not service:
        return jsonify({'status': 'error', 'message': 'Service name is required'}), 400

    try:
        logger.info(f'Restarting service: {service}')
        # In a real implementation, you would use systemd or similar to restart the service
        # For now, we'll just log it
        return jsonify({
            'status': 'success',
            'message': f'Service {service} restart initiated'
        })
    except Exception as e:
        logger.error(f'Error restarting service {service}: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/system/shutdown', methods=['POST'])
def system_shutdown():
    """Shut down the device."""
    if 'username' not in session or session.get('role') != 'admin':
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401

    try:
        logger.info('Initiating system shutdown')
        # In a real implementation, you would call the system shutdown command
        # For safety, we'll just log it for now
        return jsonify({
            'status': 'success',
            'message': 'Shutdown initiated. The system will power off shortly.'
        })
    except Exception as e:
        logger.error(f'Error shutting down: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/files/sync', methods=['POST'])
def sync_files_to_cloud():
    """Sync local data files to the Brain server cloud storage."""
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401

    try:
        uploaded, skipped, errors = device_manager.sync_files_to_cloud()
        return jsonify({
            'status': 'success',
            'uploaded': uploaded,
            'skipped': skipped,
            'errors': errors,
            'message': f'Synced {uploaded} files to cloud ({skipped} already synced)'
        })
    except Exception as e:
        logger.error(f'Error syncing files: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Start the scheduler
    if not device_scheduler.running:
        device_scheduler.start()

    # Run the application with threading mode (more compatible on Windows)
    socketio.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        use_reloader=False,
        allow_unsafe_werkzeug=True
    )
