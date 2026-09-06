"""Thoth Flask Backend Application.

This module provides the main Flask application for the Thoth device,
including REST API endpoints and WebSocket support for real-time data streaming.
"""

import os
import sys
import json
import io
import math
import re
import html
import subprocess
import shutil
import threading
import time
import logging
import uuid
import socket
import psutil
import platform
try:
    import netifaces
except Exception:  # pragma: no cover - optional on minimal installs
    netifaces = None
import requests
import tempfile
import zipfile
import gzip
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import deque
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

from backend.config import Config, BUTTON_ACTIONS, SENSOR_CONFIG
from backend.models import SensorReading, SystemStatus, ButtonConfig, UploadResult
from backend.device_manager import DeviceManager
from backend.auth_manager import AuthManager
from backend.radar_analysis import create_example2_processor, serialize_example2_plot
from backend.terminal_manager import SSHTerminalManager
from backend.sensor_detection import detect_sensor_inventory
from backend.home_assistant import get_home_assistant_publisher, load_home_assistant_config, save_home_assistant_config, test_home_assistant_connection
from backend.capture_manager import (
    list_minutes,
    list_minute_folders,
    get_minute,
    capture_files,
    current_minute,
    minute_summary,
    minute_metrics,
    cleanup_old_minutes,
    zip_minute_folder,
    stream_minute_folders,
    update_minute_labels,
    preview_text,
)
from backend.capture_container import (
    csi_average_series as container_csi_average_series,
    first_camera_frame,
    radar_bytes as container_radar_bytes,
    read_camera_frame,
    read_capture_metadata,
)

THOTH_ROOT = Path(__file__).resolve().parents[2]
MMW_RELEASE = THOTH_ROOT / 'WS' / 'MMW-HAT' / 'MMW-HAT-Release'
RADAR_OCCUPANCY_STATE = THOTH_ROOT / 'config' / 'radar_occupancy.json'
RADAR_ROOM_CONFIG = MMW_RELEASE / 'example_2_advanced' / 'config' / 'room_config.json'
RADAR_LIVE_STALE_SECONDS = max(
    5.0, float(os.getenv('THOTH_RADAR_LIVE_STALE_SECONDS', '12.0'))
)
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

TRACK_EXAMPLE_DIR = MMW_RELEASE / 'example_2_track'
if str(TRACK_EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(TRACK_EXAMPLE_DIR))

try:
    from utility.helper import parse_radar_cfg, read_uint12, split_samples
except Exception:  # pragma: no cover - import may fail on minimal installs
    parse_radar_cfg = None
    read_uint12 = None
    split_samples = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

RADAR_PLOTS = ('xy-tracking',)
RADAR_PLOT_AXES = {
    'xy-tracking': ('X', 'Y'),
}
RADAR_CONFIG_DIR = MMW_RELEASE / 'radar_config' / 'config_3rx_3m'
RADAR_TRACKING_CONFIG = TRACK_EXAMPLE_DIR / 'config' / 'processing_config.json'
CSI_NUMBER_RE = re.compile(r'[-+]?\d+(?:\.\d+)?')
RADAR_CACHE_VERSION = 5
_radar_cache_lock = threading.RLock()
_home_assistant_manifest_lock = threading.Lock()

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
    if session.get('username') and not auth_manager.is_authenticated():
        session.clear()
    if request.path != '/api/radar/occupancy':
        logger.info(f"Request: {request.method} {request.path} - {request.remote_addr}")

@app.after_request
def log_response(response):
    if request.path != '/api/radar/occupancy':
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
terminal_manager = SSHTerminalManager(socketio, Config)

# Global state
collection_active = False
collection_process: Optional[subprocess.Popen] = None
COLLECTOR_PAUSE_PATH = THOTH_ROOT / 'config' / 'collector.pause'
_capture_timeline_cache: Dict[str, Any] = {'signature': None, 'items': []}
_capture_manifest_cache: Dict[str, Any] = {}
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

def get_active_wifi_state() -> Dict[str, Any]:
    """Return the live WiFi state from NetworkManager."""
    state = {
        'connected': False,
        'ssid': None,
        'connection': None,
        'interface': 'wlan0',
        'ip_address': None,
    }

    if platform.system() != 'Linux':
        return state

    try:
        status = subprocess.run(
            ['nmcli', '-t', '-f', 'DEVICE,TYPE,STATE,CONNECTION', 'device', 'status'],
            capture_output=True, text=True, timeout=5
        )
        if status.returncode == 0:
            for line in status.stdout.splitlines():
                parts = line.split(':', 3)
                if len(parts) < 4:
                    continue
                device, dev_type, dev_state, connection = parts
                if device == 'wlan0' and dev_type == 'wifi':
                    state['connected'] = dev_state == 'connected'
                    if connection and connection != '--':
                        state['connection'] = connection
                        state['ssid'] = connection
                    break

        detail = subprocess.run(
            ['nmcli', '-t', '-f', 'GENERAL.CONNECTION,IP4.ADDRESS[1]', 'device', 'show', 'wlan0'],
            capture_output=True, text=True, timeout=5
        )
        if detail.returncode == 0:
            for line in detail.stdout.splitlines():
                if line.startswith('GENERAL.CONNECTION:') and not state['ssid']:
                    connection = line.split(':', 1)[1].strip()
                    if connection and connection != '--':
                        state['ssid'] = connection
                        state['connection'] = connection
                elif line.startswith('IP4.ADDRESS[1]:'):
                    ip_value = line.split(':', 1)[1].strip()
                    if ip_value:
                        state['ip_address'] = ip_value.split('/', 1)[0]

        if not state['ip_address']:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.connect(('1.1.1.1', 80))
                state['ip_address'] = sock.getsockname()[0]
                state['connected'] = True
            finally:
                sock.close()
    except Exception as e:
        logger.debug(f"Unable to determine active WiFi state: {e}")

    return state

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
    if path.suffix.lower() == '.npz':
        handle_context = io.BytesIO(container_radar_bytes(path))
    else:
        handle_context = open(path, 'rb')
    with handle_context as handle:
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


def _extract_csi_payload(raw: str) -> str:
    payloads = re.findall(r'\[([^\]]*)\]', raw or '')
    return payloads[-1] if payloads else (raw or '')


def _parse_csi_payload(raw: str) -> List[float]:
    payload = _extract_csi_payload(raw)
    values: List[float] = []
    for token in CSI_NUMBER_RE.findall(payload):
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _parse_csi_average_series(path: Path, limit: int = 2400) -> List[float]:
    if not path.exists():
        return []
    if path.suffix.lower() == '.npz':
        try:
            return container_csi_average_series(path, limit=limit)
        except Exception:
            return []

    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            first = ''
            while True:
                first = handle.readline()
                if not first:
                    return []
                first = first.strip()
                if first:
                    break

            recent_lines = deque(maxlen=limit + 1)
            for line in handle:
                line = line.strip()
                if line:
                    recent_lines.append(line)
    except Exception:
        return []

    def _mean_from_payload(payload: str) -> Optional[float]:
        values = _parse_csi_payload(payload)
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
    if first.startswith('{'):
        for line in [first, *recent_lines]:
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
        data_index = -1
        for candidate in ('data', 'raw_csi_line'):
            if candidate in header:
                data_index = header.index(candidate)
                break
        if data_index < 0:
            data_index = len(header) - 1
        if data_index < 0:
            return []
        for line in recent_lines:
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


def _best_live_minute_for_kind(kind: str) -> tuple[Optional[Path], Dict[str, Optional[Path]]]:
    """Return the best available minute folder and file map for a live kind."""
    def _has_kind(files: Dict[str, Optional[Path]]) -> bool:
        container = files.get('container')
        container_info: Dict[str, Any] = {}
        if container and container.exists():
            try:
                seconds = read_capture_metadata(container).get('seconds') or []
                container_info = {
                    'camera': sum(int(item.get('camera_frames') or 0) for item in seconds if isinstance(item, dict)),
                    'csi': sum(int(item.get('csi_samples') or 0) for item in seconds if isinstance(item, dict)),
                    'radar': sum(int(item.get('radar_samples') or 0) for item in seconds if isinstance(item, dict)),
                }
            except Exception:
                container_info = {}
        if kind == 'video':
            return bool((files.get('video') and files['video'].exists()) or files.get('camera_images') or container_info.get('camera'))
        if kind == 'csi':
            return bool(
                (files.get('csi_csv') and files['csi_csv'].exists())
                or (files.get('csi_timestamped') and files['csi_timestamped'].exists())
                or (files.get('csi_serial') and files['csi_serial'].exists())
                or container_info.get('csi')
            )
        if kind == 'radar':
            return bool((files.get('radar') and files['radar'].exists()) or container_info.get('radar'))
        return False

    current = current_minute()
    if current:
        current_files = capture_files(current)
        if _has_kind(current_files):
            return current, current_files

    for minute in list_minutes():
        minute_dir = get_minute(minute.get('minute', ''))
        if not minute_dir:
            continue
        files = capture_files(minute_dir)
        if _has_kind(files):
            return minute_dir, files

    return current, capture_files(current) if current else {}


def _render_video_frame(video_path: Path) -> bytes:
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('ffmpeg was not found in PATH.')

    attempts = [
        ['-sseof', '-1'],
        ['-sseof', '-0.5'],
        [],
    ]
    last_error = None
    for seek_args in attempts:
        cmd = [
            ffmpeg,
            '-hide_banner',
            '-loglevel',
            'error',
            *seek_args,
            '-i',
            str(video_path),
            '-frames:v',
            '1',
            '-f',
            'image2pipe',
            '-vcodec',
            'mjpeg',
            'pipe:1',
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=12)
            if result.returncode == 0 and result.stdout:
                return result.stdout
            last_error = result.stderr.decode('utf-8', errors='replace').strip()
        except Exception as exc:
            last_error = str(exc)
    raise RuntimeError(last_error or 'Unable to render video frame')


def _csi_plot_payload(path: Path, limit: int = 2400) -> Dict[str, Any]:
    points = _parse_csi_average_series(path, limit=limit)
    frames = _sample_series_frames(points)
    return {
        'points': points,
        'frames': frames,
        'sample_count': len(points),
        'frame_count': len(frames),
        'frame_interval_ms': 120,
        'title': 'Average CSI amplitude',
        'x_label': 'Packet',
        'y_label': 'Average magnitude',
        'updated': datetime.utcnow().isoformat(),
    }


def _sample_series_frames(points: List[float], max_frames: int = 80) -> List[Dict[str, Any]]:
    if not points:
        return []
    if len(points) <= 1:
        return [{'index': len(points), 'points': points[:]}]

    step = max(1, math.ceil(len(points) / max_frames))
    frames: List[Dict[str, Any]] = []
    for end in range(step, len(points) + 1, step):
        frames.append({'index': end, 'points': points[:end]})

    if frames[-1]['index'] != len(points):
        frames.append({'index': len(points), 'points': points[:]})
    return frames


def _count_radar_frames(path: Path) -> int:
    count = 0
    with open(path, 'rb') as handle:
        while True:
            version_bytes = handle.read(4)
            if not version_bytes or len(version_bytes) < 4:
                break
            version = int.from_bytes(version_bytes, byteorder='little', signed=False)
            if version != 0:
                break
            handle.read(4)
            data_len_bytes = handle.read(4)
            if len(data_len_bytes) < 4:
                break
            data_len = int.from_bytes(data_len_bytes, byteorder='little', signed=False)
            handle.seek(data_len, os.SEEK_CUR)
            count += 1
    return count


@lru_cache(maxsize=32)
def _radar_animation_bundle(
    path_str: str,
    mtime_ns: int,
    size: int,
    detection_threshold_db: float,
) -> Dict[str, Dict[str, Any]]:
    del mtime_ns, size
    radar_path = Path(path_str)
    if np is None or CubeProcessor is None:
        raise RuntimeError('Radar plotting dependencies are unavailable')

    cache_path = radar_path.parent / (
        '.radar-playback-v%d-%.1f.pkl.gz' % (RADAR_CACHE_VERSION, detection_threshold_db)
    )
    with _radar_cache_lock:
        if cache_path.exists() and cache_path.stat().st_mtime_ns >= radar_path.stat().st_mtime_ns:
            try:
                with gzip.open(cache_path, 'rb') as handle:
                    return pickle.load(handle)
            except Exception:
                cache_path.unlink(missing_ok=True)

    setting = _radar_setting()
    if not setting:
        raise RuntimeError('Radar settings not found')

    total_frames = _count_radar_frames(radar_path)
    if total_frames <= 0:
        raise RuntimeError('No radar frames available')

    # These three heatmaps intentionally use the same CubeProcessor parameters,
    # dimension pairs, reduction, and log scaling as MMW-HAT Example 1.
    mmw_proc = CubeProcessor(setting, num_azimuth_bin=16, num_elevation_bin=16)
    tracking_proc = None
    if all((parse_radar_cfg, read_uint12, split_samples)) and RADAR_TRACKING_CONFIG.exists():
        tracking_proc = create_example2_processor(parse_radar_cfg(setting))
        tracking_proc.detection_threshold_db = detection_threshold_db
    max_frames = min(60, total_frames)
    sample_indices = {1, total_frames}
    if max_frames > 2:
        for slot in range(1, max_frames - 1):
            target = round(1 + (total_frames - 1) * (slot / (max_frames - 1)))
            sample_indices.add(max(1, min(total_frames, target)))
    sample_indices = set(sorted(sample_indices))
    frames_by_plot: Dict[str, List[Dict[str, Any]]] = {plot: [] for plot in RADAR_PLOTS}
    heatmap_plots = RADAR_PLOTS[:3]
    detected_frames = 0
    evaluated_frames = 0

    for index, (seq, raw_data) in enumerate(_iter_radar_frames(radar_path), start=1):
        tracking_result = None
        if tracking_proc is not None:
            try:
                radar_param = tracking_proc.radar_config
                adc_data = read_uint12(raw_data)
                split = split_samples(
                    adc_data,
                    1,
                    radar_param['num_chirps_per_frame'],
                    radar_param['num_samples_per_chirp'],
                    radar_param['num_antennas'],
                )
                tracking_frame = np.transpose(split[0], (2, 0, 1))
                location, score, gui_plot = tracking_proc.update(tracking_frame)
                tracking_result = (location, score, gui_plot, dict(tracking_proc.last_detection))
                evaluated_frames += 1
                if tracking_proc.last_detection.get('detected'):
                    detected_frames += 1
            except Exception as exc:
                logger.warning('Radar X-Y tracking failed for frame %s: %s', index, exc)
                tracking_proc = None

        if index not in sample_indices:
            continue
        mmw_proc.process_raw_data(raw_data)
        if mmw_proc.data_cube_fft is None:
            continue

        for plot in heatmap_plots:
            axis_names = RADAR_PLOT_AXES[plot]
            img = mmw_proc.vis_2d(axis_names[0], axis_names[1])
            img = np.log10(np.maximum(img, 1e-9))

            x_name = axis_names[1].lower()
            y_name = axis_names[0].lower()
            x_values = np.asarray(mmw_proc.proc_param.get(f'{x_name}_bin', []), dtype=float).tolist()
            y_values = np.asarray(mmw_proc.proc_param.get(f'{y_name}_bin', []), dtype=float).tolist()
            frames_by_plot[plot].append({
                'seq': seq,
                'index': index,
                'x': x_values,
                'y': y_values,
                'z': img.tolist(),
            })

        if tracking_result is not None:
            location, score, gui_plot, detection = tracking_result
            native_plot = serialize_example2_plot(tracking_proc, gui_plot, location, score, detection)
            playback = native_plot.get('playback') if isinstance(native_plot, dict) else None
            if not isinstance(playback, dict):
                continue
            frames_by_plot['xy-tracking'].append({
                'seq': seq,
                'index': index,
                **playback,
                'location': native_plot.get('location'),
                'score': native_plot.get('score'),
                'detected': bool(detection.get('detected')),
                'snr_db': detection.get('snr_db'),
                'threshold_db': detection.get('threshold_db'),
                'peak_power_db': detection.get('peak_power_db'),
                'noise_floor_db': detection.get('noise_floor_db'),
            })

    bundle: Dict[str, Dict[str, Any]] = {}
    occupancy_ratio = detected_frames / evaluated_frames if evaluated_frames else 0.0
    occupancy = {
        'label': 'occupied' if detected_frames * 2 >= evaluated_frames else 'empty',
        'detected_frames': detected_frames,
        'evaluated_frames': evaluated_frames,
        'ratio': occupancy_ratio,
        'rule': 'occupied when the detected-frame percentage meets the configured threshold',
    }
    for plot in RADAR_PLOTS:
        frames = frames_by_plot[plot]
        if not frames:
            continue
        latest = frames[-1]
        axis_names = RADAR_PLOT_AXES[plot]
        latest_z = latest.get('z')
        if latest_z is None and latest.get('z_shape'):
            rows, columns = latest['z_shape']
            latest_z = [[0.0] * columns for _ in range(rows)]
            for row, column, value in latest.get('z_sparse', []):
                latest_z[row][column] = value
        bundle[plot] = {
            'plot': plot,
            'title': f'{axis_names[0]} vs {axis_names[1]}',
            'x_label': axis_names[1],
            'y_label': axis_names[0],
            'x': latest['x'],
            'y': latest['y'],
            'z': latest_z,
            'frames': frames,
            'frame_count': total_frames,
            'sample_count': len(frames),
            'frame_interval_ms': max(50, int(round(1000.0 / max(1.0, float(setting.get('frame_rate') or 10.0))))),
            'updated': datetime.utcnow().isoformat(),
            'occupancy': occupancy,
        }
        if plot == 'xy-tracking':
            bundle[plot]['title'] = 'X-Y Tracking'
            bundle[plot]['x_label'] = 'Y / lateral (m)'
            bundle[plot]['y_label'] = 'X / forward (m)'
            bundle[plot]['location'] = latest.get('location')
            bundle[plot]['score'] = latest.get('score')
            bundle[plot]['detected'] = latest.get('detected', False)
            bundle[plot]['snr_db'] = latest.get('snr_db')
            bundle[plot]['threshold_db'] = latest.get('threshold_db')
            bundle[plot]['peak_power_db'] = latest.get('peak_power_db')
            bundle[plot]['noise_floor_db'] = latest.get('noise_floor_db')
            bundle[plot]['coordinate_space'] = 'example2_sensor_local'
            bundle[plot]['native_pipeline'] = True
    with _radar_cache_lock:
        try:
            temporary = cache_path.with_suffix(cache_path.suffix + '.tmp')
            with gzip.open(temporary, 'wb', compresslevel=3) as handle:
                pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(cache_path)
        except Exception as exc:
            logger.warning('Unable to persist radar playback cache: %s', exc)
    return bundle


def _radar_plot_payload(radar_path: Path, plot: str) -> Dict[str, Any]:
    axis_names = RADAR_PLOT_AXES.get(plot)
    if not axis_names:
        raise RuntimeError(f'Unsupported radar plot kind: {plot}')

    stat = radar_path.stat()
    settings = device_manager.get_device_settings()
    try:
        detection_threshold_db = min(30.0, max(0.0, float(settings.get('radar_detection_threshold_db', 8.0))))
    except (TypeError, ValueError):
        detection_threshold_db = 8.0
    # Four plot requests arrive together. Serialize the cache miss so only one
    # request performs the expensive minute-wide FFT/tracking pass.
    with _radar_cache_lock:
        bundle = _radar_animation_bundle(
            str(radar_path),
            stat.st_mtime_ns,
            stat.st_size,
            detection_threshold_db,
        )
    payload = bundle.get(plot)
    if not payload:
        raise RuntimeError('No radar frames available')
    occupancy = payload.get('occupancy') or {}
    detected_frames = max(0, int(occupancy.get('detected_frames') or 0))
    evaluated_frames = max(0, int(occupancy.get('evaluated_frames') or 0))
    occupancy['detected_frames'] = detected_frames
    occupancy['evaluated_frames'] = evaluated_frames
    occupancy['ratio'] = detected_frames / evaluated_frames if evaluated_frames else 0.0
    occupancy['threshold_db'] = detection_threshold_db
    occupancy['classification'] = 'green' if detected_frames > 0 else 'red'
    occupancy['label'] = 'occupied' if occupancy['classification'] == 'green' else 'empty'
    payload['occupancy'] = occupancy
    # Live collection owns persistence, automatic labeling, and Home Assistant
    # publication. Plot generation is deliberately read-only.
    return payload


def _prewarm_latest_radar_playback() -> None:
    """Process the newest completed minute before a user opens its plots."""
    if collection_active:
        return
    try:
        settings = device_manager.get_device_settings()
        threshold = min(30.0, max(0.0, float(settings.get('radar_detection_threshold_db', 8.0))))
        for summary in list_minutes()[:3]:
            if not summary.get('capture_finished'):
                continue
            minute_dir = get_minute(str(summary.get('minute', '')))
            if not minute_dir:
                continue
            radar_path = capture_files(minute_dir).get('radar')
            if not radar_path or not radar_path.exists():
                continue
            cache_path = radar_path.parent / (
                '.radar-playback-v%d-%.1f.pkl.gz' % (RADAR_CACHE_VERSION, threshold)
            )
            if cache_path.exists() and cache_path.stat().st_mtime_ns >= radar_path.stat().st_mtime_ns:
                continue
            _radar_plot_payload(radar_path, 'xy-tracking')
            break
    except Exception as exc:
        logger.warning('Radar playback prewarm failed: %s', exc)


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


def _blank_xy_payload(title: str = 'X-Y localization') -> Dict[str, Any]:
    return {
        'plot': 'xy-tracking',
        'title': title,
        'x_label': 'X (m)',
        'y_label': 'Y (m)',
        'x': [0.0, 1.0],
        'y': [0.0, 1.0],
        'z': [[0.0, 0.0], [0.0, 0.0]],
        'frames': [],
        'frame_count': 0,
        'sample_count': 0,
        'frame_interval_ms': 1000,
        'updated': datetime.utcnow().isoformat(),
        'occupancy': {
            'label': 'empty',
            'detected_frames': 0,
            'evaluated_frames': 0,
            'ratio': 0.0,
            'threshold_db': 8.0,
        },
        'location': [0.0, 0.0],
        'score': 0.0,
        'detected': False,
        'snr_db': 0.0,
        'threshold_db': 0.0,
        'peak_power_db': 0.0,
        'noise_floor_db': 0.0,
        'targets': [],
        'motion_points': 0,
        'coordinate_space': 'example2_sensor_local',
        'native_pipeline': True,
    }


def _load_xy_tracking_payload(minute_dir: Path, files: Dict[str, Optional[Path]]) -> Dict[str, Any]:
    xy_payload_path = minute_dir / 'xy-tracking.json'
    if xy_payload_path.exists():
        try:
            payload = json.loads(xy_payload_path.read_text(encoding='utf-8'))
            if isinstance(payload, dict):
                return payload
        except Exception as exc:
            logger.warning('Unable to read xy tracking payload for %s: %s', minute_dir.name, exc)

    chunk_payloads: list[Dict[str, Any]] = []
    for radar_path in files.get('radar_bins') or []:
        if not radar_path or not radar_path.exists():
            continue
        try:
            payload = _radar_plot_payload(radar_path, 'xy-tracking')
            if isinstance(payload, dict):
                chunk_payloads.append(payload)
        except Exception as exc:
            logger.debug('Skipping incomplete radar chunk %s: %s', radar_path.name, exc)
            continue
    if chunk_payloads:
        try:
            from backend.radar_analysis import compile_minute_xy_payload
            combined = compile_minute_xy_payload(chunk_payloads)
            if isinstance(combined, dict):
                try:
                    _write_json_file(xy_payload_path, combined)
                except OSError as exc:
                    logger.warning('Unable to cache xy tracking payload for %s: %s', minute_dir.name, exc)
                return combined
        except Exception as exc:
            logger.warning('Unable to combine xy tracking chunks for %s: %s', minute_dir.name, exc)

    return _blank_xy_payload()


def _live_xy_window(payload: Dict[str, Any], frame_limit: int = 10) -> Dict[str, Any]:
    """Return only the newest frames needed by a realtime client."""
    bounded = dict(payload)
    frames = payload.get('frames') if isinstance(payload.get('frames'), list) else []
    bounded['frames'] = frames[-max(1, frame_limit):]
    bounded['sample_count'] = len(bounded['frames'])
    bounded['stream_window_frames'] = max(1, frame_limit)
    if bounded['frames']:
        # Every playback frame carries its sparse map, so the 200x400 dense
        # fallback matrix is redundant in the live response.
        bounded['z'] = []
    return bounded


def get_capture_overview() -> Dict[str, Any]:
    """Summarize the capture directory."""
    minutes = list_minutes()
    latest = minutes[0] if minutes else None
    return {
        "capture_dir": Config.CAPTURE_DATA_DIR,
        "max_disk_percent": Config.CAPTURE_MAX_DISK_PERCENT,
        "minute_count": len(minutes),
        "latest_minute": latest,
    }

def get_system_status(update_remote: bool = True) -> SystemStatus:
    """Get current system status and optionally update the Brain server.

    Args:
        update_remote: If True, update the status on the Brain server

    Returns:
        SystemStatus: Current system status
    """
    try:
        is_windows = platform.system() == 'Windows'

        wifi_state = get_active_wifi_state()
        wifi_connected = bool(wifi_state.get('connected'))

        # Check collection status.
        collection_status = collection_active
        global collection_process
        if collection_process is not None and collection_process.poll() is not None:
            collection_process = None
            collection_status = False
        if not is_windows:
            try:
                service_collection_status = subprocess.run(
                    ["systemctl", "is-active", "thoth-collector"],
                    capture_output=True, text=True
                ).stdout.strip() == "active"
                collection_status = (collection_status or service_collection_status) and not COLLECTOR_PAUSE_PATH.exists()
            except Exception:
                pass

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

        ip_address = wifi_state.get('ip_address')
        if not ip_address:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("1.1.1.1", 80))
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
                    'disk_usage': disk_usage,
                    'hardware_info': device_manager._build_hardware_info(),
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
        if netifaces is not None:
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
            'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            'device_type': 'thoth',
            'is_raspberry_pi': platform.system() == 'Linux' and (
                'arm' in platform.machine().lower() or os.path.exists('/proc/device-tree/model')
            ),
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

        auth_token = getattr(Config, 'USER_AUTH_TOKEN', None) or getattr(Config, 'BRAIN_AUTH_TOKEN', None)
        if not auth_token:
            logger.debug("No authenticated user token available, skipping device registration")
            return False
        if device_manager.pairing_required:
            logger.debug("Device registration paused until thothHUB pairing is completed")
            return False
        # Registration establishes identity and ownership. A single heartbeat
        # loop maintains presence, inventory and settings after that; posting
        # both every ten seconds doubled traffic and repeatedly retried identity.
        if device_manager.registered and device_manager.auth_token:
            if not (device_manager.heartbeat_thread and device_manager.heartbeat_thread.is_alive()):
                device_manager.start_heartbeat(Config.HEARTBEAT_INTERVAL)
            return True

        success, message = device_manager.register_device(auth_token.strip())
        if success:
            logger.info(message)
            # Ensure heartbeat is running so the brain server sees the device as online
            if not (device_manager.heartbeat_thread and device_manager.heartbeat_thread.is_alive()):
                device_manager.start_heartbeat(Config.HEARTBEAT_INTERVAL)
        else:
            logger.warning(message)
        return success
    except Exception as e:
        logger.error(f"Unexpected error in device registration: {str(e)}", exc_info=True)
        return False


def _provision_terminal_login(username: str) -> None:
    """Create or update the local SSH identity for a freshly authenticated user."""
    try:
        ssh_user = terminal_manager.ensure_user(username)
        session['ssh_user'] = ssh_user
        logger.info("Provisioned SSH access for %s", ssh_user)
    except Exception as exc:
        logger.error("Failed to provision SSH access for %s: %s", username, exc, exc_info=True)
        session['ssh_user'] = username


def _activate_device_session(username: str, token: str) -> None:
    """Persist the authenticated session and bring the device online."""
    session['username'] = username
    session['token'] = token
    Config.USER_AUTH_TOKEN = token
    _provision_terminal_login(username)

    success, message = device_manager.register_device(token)
    if success:
        device_manager.start_heartbeat(Config.HEARTBEAT_INTERVAL)
        logger.info("Login successful for user: %s, device registered", username)
    else:
        logger.warning("Device registration failed: %s", message)

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
    lambda: cleanup_old_minutes(max_disk_percent=Config.CAPTURE_MAX_DISK_PERCENT),
    'interval',
    minutes=10,
    id='capture_cleanup',
    replace_existing=True
)
device_scheduler.start()

# Load registration info if available
device_manager.load_registration_info()
cleanup_old_minutes(max_disk_percent=Config.CAPTURE_MAX_DISK_PERCENT)
if not auth_manager.is_authenticated() and not getattr(Config, 'BRAIN_AUTH_TOKEN', None):
    try:
        device_manager.mark_device_offline()
    except Exception as e:
        logger.error(f"Error forcing offline state at startup: {e}")

# Routes
@app.route('/')
def index():
    """Show the local device dashboard."""
    return redirect(url_for('status'))


@app.route('/connect')
def connect():
    """Open the SSH-backed terminal for the logged-in user."""
    if 'username' not in session:
        return redirect(url_for('login', next=url_for('connect')))

    ssh_available = platform.system() == 'Linux'
    return render_template(
        'connect.html',
        username=session.get('username'),
        ssh_user=session.get('ssh_user') or session.get('username'),
        ssh_available=ssh_available,
        terminal_host=socket.gethostname(),
        terminal_port=Config.PORT,
    )

@app.route('/setup')
def setup():
    """Show the setup page and current network state."""
    # If already authenticated, go to status
    if 'username' in session:
        return redirect(url_for('status'))
    return render_template(
        'setup.html',
        wifi_state=get_active_wifi_state(),
        version=Config.VERSION,
        boot_flow=[
            "Set WiFi in Raspberry Pi Imager before first boot.",
            "Boot the Pi and let it join the saved WiFi automatically.",
            "Open http://thoth.local:5000 from any device on the LAN.",
            "Log in once to activate device registration and heartbeat.",
        ],
    )

@app.route('/api/wifi/scan', methods=['GET'])
def api_wifi_scan():
    """Legacy WiFi setup endpoint removed."""
    return jsonify({'status': 'error', 'error': 'WiFi setup is handled through Raspberry Pi Imager.'}), 410

@app.route('/api/wifi/connect', methods=['POST'])
def api_wifi_connect():
    """Legacy WiFi setup endpoint removed."""
    return jsonify({'status': 'error', 'error': 'WiFi setup is handled through Raspberry Pi Imager.'}), 410

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
            session['user_id'] = result['user'].get('user_id')
            _activate_device_session(username, result['token'])

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
                session['user_id'] = result['user'].get('user_id')
                _activate_device_session(username, result['token'])

                logger.info(f"Login successful for user: {username}")
                logger.info("Device registration will now use this user's token")

                return render_template(
                    'login_success.html',
                    username=username,
                    access_token=result['token'],
                    user_info=result['user'],
                )

            flash('Invalid username or password', 'error')

        except Exception as e:
            logger.error(f"Login error: {str(e)}", exc_info=True)
            flash(f'Login failed: {str(e)}', 'error')

    except Exception as e:
        logger.error(f"Login error: {str(e)}", exc_info=True)
        flash('An error occurred. Please try again.', 'error')

    return redirect(url_for('login'))


@socketio.on('terminal_open')
def terminal_open(data):
    """Open an SSH-backed terminal for the logged-in user."""
    if 'username' not in session:
        emit('terminal_error', {'message': 'Unauthorized'})
        return

    try:
        info = terminal_manager.open(request.sid, session['username'])
        emit('terminal_ready', info)
        emit('terminal_output', {
            'session_id': request.sid,
            'data': f"\r\nConnected as {info['ssh_user']}@{info['host']}\r\n"
        })
    except Exception as exc:
        logger.error("Failed to open terminal session: %s", exc, exc_info=True)
        emit('terminal_error', {'message': str(exc)})


@socketio.on('terminal_input')
def terminal_input(data):
    """Send user input to the SSH session."""
    if 'username' not in session:
        emit('terminal_error', {'message': 'Unauthorized'})
        return

    payload = data or {}
    terminal_manager.send(request.sid, payload.get('data', ''))


@socketio.on('terminal_resize')
def terminal_resize(data):
    """Resize the terminal PTY to match the browser terminal."""
    if 'username' not in session:
        return

    payload = data or {}
    terminal_manager.resize(
        request.sid,
        int(payload.get('rows', 34) or 34),
        int(payload.get('cols', 132) or 132),
    )


@socketio.on('disconnect')
def terminal_disconnect():
    """Close any SSH session tied to the browser connection."""
    terminal_manager.close(request.sid)

@app.route('/status')
def status():
    """Display the device status page."""
    try:
        # Home must remain local-first; remote registration and heartbeat run
        # in background jobs and must never delay the dashboard render.
        system_status = get_system_status(update_remote=False)
        device_info = device_manager.get_device_info()
        minutes = _capture_timeline_items()
        active_minute = current_minute() if system_status.collection_active else None

        return render_template('status.html',
                            system_status=system_status,
                            device_info=device_info,
                            username=session.get('username'),
                            capture_overview={'minute_count': len(minutes)},
                            minutes=minutes,
                            active_minute=active_minute.name if active_minute else None,
                            pairing_state={
                                key: auth_manager.pairing_session.get(key)
                                for key in ('code', 'device_id', 'expires_at')
                            } if auth_manager.pairing_session else None,
                            hub_paired=(
                                auth_manager.is_authenticated()
                                and not device_manager.pairing_required
                            ),
                            pairing_required=device_manager.pairing_required)

    except Exception as e:
        logger.error(f"Error in status route: {str(e)}", exc_info=True)
        flash('An error occurred while loading the status page.', 'error')
        return redirect(url_for('index'))


@app.route('/api/pairing/start', methods=['POST'])
def start_pairing():
    """Show a one-time code that binds this physical device to thothHUB."""
    try:
        device_name = f"Thoth-{device_manager.device_id[:8]}"
        result = auth_manager.start_pairing(
            device_manager.device_id,
            device_name,
            device_manager._build_hardware_info(),
        )
        return jsonify({'success': True, 'status': 'pending', **result})
    except Exception as exc:
        logger.warning("Unable to start thothHUB pairing: %s", exc)
        return jsonify({'success': False, 'message': str(exc)}), 502


@app.route('/api/pairing/status')
def pairing_status():
    """Complete a claimed pairing without exposing account credentials."""
    try:
        result = auth_manager.pairing_status()
        if result.get('status') == 'paired':
            user = result.get('user') or {}
            session['user_id'] = user.get('user_id')
            _activate_device_session(str(user.get('username') or 'paired-user'), result['token'])
        return jsonify(result)
    except Exception as exc:
        logger.warning("Unable to check thothHUB pairing: %s", exc)
        return jsonify({'success': False, 'message': str(exc)}), 502


@app.route('/api/assistant', methods=['POST'])
def assistant_query():
    """Use the same authenticated Brain assistant as Research Portal."""
    token = (
        getattr(auth_manager, 'token', None)
        or getattr(device_manager, 'auth_token', None)
        or getattr(Config, 'USER_AUTH_TOKEN', None)
    )
    if not token:
        return jsonify({'success': False, 'message': 'Log in to use the assistant'}), 401
    payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    payload = payload if isinstance(payload, dict) else {}
    query = str(payload.get('query') or '').strip()
    attachment = request.files.get('attachment') if not request.is_json else None
    attachment_context = None
    if attachment and attachment.filename:
        filename = Path(attachment.filename).name
        if Path(filename).suffix.lower() != '.txt':
            return jsonify({'success': False, 'message': 'Only .txt attachments are supported'}), 400
        raw = attachment.stream.read(262145)
        if len(raw) > 262144:
            return jsonify({'success': False, 'message': 'Text attachments must be 256 KB or smaller'}), 413
        try:
            content = raw.decode('utf-8')
        except UnicodeDecodeError:
            return jsonify({'success': False, 'message': 'The attached text file must use UTF-8'}), 400
        attachment_context = {'name': filename, 'content': content}
    if not query and not attachment_context:
        return jsonify({'success': False, 'message': 'Write a message or attach a .txt file'}), 400
    if not query:
        query = f"Please read and respond to the attached text file {attachment_context['name']}."
    try:
        context = {
            'surface': 'thoth-device-dashboard',
            'device_id': device_manager.device_id,
            'collection_active': get_system_status(update_remote=False).collection_active,
            'username': session.get('username'),
        }
        if attachment_context:
            context['text_attachment'] = attachment_context
        response = requests.post(
            _brain_api_url('/query'),
            json={
                'query': query,
                'chat_id': payload.get('chat_id'),
                'context': context,
            },
            headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
            timeout=90,
        )
        body = _response_json(response)
        if body is None:
            logger.error(
                'Brain assistant returned non-JSON content: status=%s content_type=%s',
                response.status_code,
                response.headers.get('content-type'),
            )
            return jsonify({
                'success': False,
                'message': f'Assistant service returned an invalid response ({response.status_code})',
            }), 502
        if response.ok:
            nested = body.get('data') if isinstance(body.get('data'), dict) else {}
            answer = body.get('response') or body.get('answer') or body.get('message') or nested.get('response') or nested.get('answer')
            if answer:
                body['response'] = answer
            body['chat_id'] = body.get('chat_id') or nested.get('chat_id') or payload.get('chat_id')
            body.setdefault('success', True)
        return jsonify(body), response.status_code
    except Exception as exc:
        logger.exception('Assistant request failed: %s', exc)
        return jsonify({'success': False, 'message': 'Assistant unavailable'}), 502


def _brain_api_url(path: str) -> str:
    base = str(Config.BRAIN_SERVER_URL or '').rstrip('/')
    suffix = '/' + str(path or '').lstrip('/')
    return f"{base}{suffix}" if base.endswith('/api') else f"{base}/api{suffix}"


def _response_json(response: requests.Response) -> Optional[Dict[str, Any]]:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _brain_profile_request(method: str, path: str, payload: Optional[Dict[str, Any]] = None):
    token = (
        getattr(auth_manager, 'token', None)
        or getattr(device_manager, 'auth_token', None)
        or getattr(Config, 'USER_AUTH_TOKEN', None)
    )
    if not token:
        return jsonify({'success': False, 'message': 'Log in to view this profile'}), 401
    try:
        response = requests.request(
            method,
            _brain_api_url(path),
            json=payload,
            headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
            timeout=20,
        )
        body = _response_json(response)
        if body is None:
            return jsonify({
                'success': False,
                'message': f'Profile service returned an invalid response ({response.status_code})',
            }), 502
        return jsonify(body), response.status_code
    except requests.RequestException as exc:
        logger.warning('Profile request failed: %s', exc)
        return jsonify({'success': False, 'message': 'Profile service is unavailable'}), 502


@app.route('/profile')
def profile():
    if 'username' not in session and not getattr(auth_manager, 'token', None):
        return redirect(url_for('login', next=url_for('profile')))
    return render_template('profile.html', username=session.get('username'))


@app.route('/api/profile', methods=['GET', 'PUT'])
def profile_api():
    payload = request.get_json(silent=True) if request.method == 'PUT' else None
    return _brain_profile_request(request.method, '/profile', payload)


@app.route('/api/profile/resend-verification', methods=['POST'])
def profile_resend_verification():
    payload = request.get_json(silent=True) or {}
    return _brain_profile_request('POST', '/resend-verification', payload)


def _read_json_file(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        with path.open('r') as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else default.copy()
    except (OSError, ValueError):
        return default.copy()


def _write_json_file(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    try:
        with temporary.open('w') as handle:
            json.dump(value, handle, indent=4)
            handle.write('\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@app.route('/presence')
def presence_view():
    """Live browser visualization shared by the device and Research Portal."""
    return render_template('presence.html')


@app.route('/api/radar/occupancy', methods=['GET'])
def api_radar_occupancy():
    state = _read_json_file(RADAR_OCCUPANCY_STATE, {
        'updated_at': 0,
        'occupied': False,
        'target_count': 0,
        'targets': [],
        'shadow': [],
    })
    room = _read_json_file(RADAR_ROOM_CONFIG, {})
    live_room = state.get('room') if isinstance(state.get('room'), dict) else {}
    if room:
        merged_room = dict(live_room)
        merged_room.update(room)
        state['room'] = merged_room
    else:
        state['room'] = live_room
    cones = state['room'].get('radar_cones') if isinstance(state.get('room'), dict) else None
    if isinstance(cones, list) and cones:
        primary = cones[0]
        state['fov'] = {
            'horizontal_deg': primary.get('horizontal_deg', 40),
            'vertical_deg': primary.get('vertical_deg', 65),
            'range_m': primary.get('range_m', 15),
        }
    try:
        updated_at = float(state.get('updated_at') or 0)
    except (TypeError, ValueError):
        updated_at = 0.0
    try:
        configured_hz = max(0.0, float(state.get('configured_hz') or 0))
    except (TypeError, ValueError):
        configured_hz = 0.0

    # Native Example 2 processing can briefly lag capture on a busy Pi, and
    # the minute collector intentionally rotates workers at wall-clock minute
    # boundaries. Three seconds was short enough to mark a healthy stream as
    # stale during either event. Allow at least twelve seconds, or thirty
    # expected frame periods for deliberately low-rate radar configurations.
    stale_after = max(
        RADAR_LIVE_STALE_SECONDS,
        30.0 / configured_hz if configured_hz > 0 else 0.0,
    )
    age_seconds = max(0.0, time.time() - updated_at) if updated_at else None
    collection_paused = COLLECTOR_PAUSE_PATH.exists()
    state['age_seconds'] = round(age_seconds, 3) if age_seconds is not None else None
    state['stale_after_seconds'] = round(stale_after, 3)
    state['collection_paused'] = collection_paused
    state['stale'] = updated_at <= 0 or bool(age_seconds is not None and age_seconds > stale_after)
    state['stale_reason'] = (
        'collection_paused' if collection_paused
        else 'waiting_for_first_frame' if updated_at <= 0
        else 'frame_timeout' if state['stale']
        else None
    )
    response = jsonify(state)
    response.headers['Cache-Control'] = 'no-store, max-age=0'
    return response


@app.route('/api/radar/room', methods=['GET', 'POST'])
def api_radar_room():
    room = _read_json_file(RADAR_ROOM_CONFIG, {})
    if not isinstance(room.get('radar_cones'), list) or not room.get('radar_cones'):
        room['radar_cones'] = [{
            'id': 'radar-1', 'name': 'Radar 1', 'enabled': True,
            'wall': room.get('sensor_wall', 'Back'),
            'position_m': room.get('sensor_position_m', 2.0),
            'height_m': room.get('sensor_height_m', 1.0),
            'horizontal_deg': 40.0, 'vertical_deg': 65.0,
            'range_m': 15.0, 'azimuth_deg': 0.0,
        }]
    for key in ('doors', 'windows', 'furniture', 'zones'):
        if not isinstance(room.get(key), list):
            room[key] = []
    room.pop('sleep_anchor', None)
    if request.method == 'GET':
        return jsonify({'success': True, 'room': room})
    payload = request.get_json(silent=True) or {}
    numeric_fields = {
        'width_m': (1.0, 20.0),
        'depth_m': (1.0, 20.0),
        'height_m': (1.5, 8.0),
        'sensor_position_m': (0.0, 20.0),
        'sensor_height_m': (0.1, 8.0),
        'max_object_height_m': (0.5, 3.0),
        'max_object_width_m': (0.2, 3.0),
        'max_object_depth_m': (0.15, 3.0),
        'max_lying_length_m': (0.8, 3.0),
    }
    for field, (minimum, maximum) in numeric_fields.items():
        if field in payload:
            room[field] = max(minimum, min(maximum, float(payload[field])))
    wall = str(payload.get('sensor_wall', room.get('sensor_wall', 'Back'))).title()
    room['sensor_wall'] = wall if wall in {'Back', 'Front', 'Left', 'Right'} else 'Back'
    if 'floor_anchored_targets' in payload:
        room['floor_anchored_targets'] = bool(payload['floor_anchored_targets'])
    walls = {'Back', 'Front', 'Left', 'Right'}
    if isinstance(payload.get('radar_cones'), list):
        room['radar_cones'] = [{
            'id': str(item.get('id') or f'radar-{index + 1}')[:64],
            'name': str(item.get('name') or f'Radar {index + 1}')[:64],
            'enabled': item.get('enabled') is not False,
            'wall': str(item.get('wall') or 'Back').title() if str(item.get('wall') or 'Back').title() in walls else 'Back',
            'position_m': min(20.0, max(0.0, float(item.get('position_m') or 0.0))),
            'height_m': min(8.0, max(0.1, float(item.get('height_m') or 1.0))),
            'horizontal_deg': min(160.0, max(5.0, float(item.get('horizontal_deg') or 40.0))),
            'vertical_deg': min(120.0, max(5.0, float(item.get('vertical_deg') or 65.0))),
            'range_m': min(30.0, max(0.5, float(item.get('range_m') or 15.0))),
            'azimuth_deg': min(90.0, max(-90.0, float(item.get('azimuth_deg') or 0.0))),
        } for index, item in enumerate(payload['radar_cones'][:8]) if isinstance(item, dict)] or room['radar_cones']
    for key in ('doors', 'windows'):
        if isinstance(payload.get(key), list):
            room[key] = [{
                'id': str(item.get('id') or f'{key[:-1]}-{index + 1}')[:64],
                'wall': str(item.get('wall') or 'Back').title() if str(item.get('wall') or 'Back').title() in walls else 'Back',
                'offset_m': min(20.0, max(0.0, float(item.get('offset_m') or 0.0))),
                'width_m': min(5.0, max(0.2, float(item.get('width_m') or 1.0))),
            } for index, item in enumerate(payload[key][:32]) if isinstance(item, dict)]
    if isinstance(payload.get('furniture'), list):
        allowed = {'chair', 'table', 'bed', 'sofa', 'desk', 'cabinet', 'custom'}
        room['furniture'] = [{
            'id': str(item.get('id') or f'furniture-{index + 1}')[:64],
            'type': str(item.get('type') or 'custom').lower() if str(item.get('type') or 'custom').lower() in allowed else 'custom',
            'x': min(20.0, max(0.0, float(item.get('x') or 0.0))),
            'y': min(20.0, max(0.0, float(item.get('y') or 0.0))),
            'width': min(5.0, max(0.1, float(item.get('width') or 0.8))),
            'depth': min(5.0, max(0.1, float(item.get('depth') or 0.8))),
        } for index, item in enumerate(payload['furniture'][:64]) if isinstance(item, dict)]
    if isinstance(payload.get('zones'), list):
        room['zones'] = [{
            'id': str(item.get('id') or f'zone-{index + 1}')[:64],
            'label': str(item.get('label') or f'Zone {index + 1}')[:64],
            'x': min(float(room.get('width_m', 5.0)), max(0.0, float(item.get('x') or 0.0))),
            'y': min(float(room.get('depth_m', 5.0)), max(0.0, float(item.get('y') or 0.0))),
            'width': min(float(room.get('width_m', 5.0)), max(0.1, float(item.get('width') or 1.0))),
            'depth': min(float(room.get('depth_m', 5.0)), max(0.1, float(item.get('depth') or 1.0))),
            'color': str(item.get('color') or '#22c55e')[:24],
        } for index, item in enumerate(payload['zones'][:64]) if isinstance(item, dict)]
    if room['radar_cones']:
        primary = room['radar_cones'][0]
        room['sensor_wall'] = primary['wall']
        room['sensor_position_m'] = primary['position_m']
        room['sensor_height_m'] = primary['height_m']
    wall_length = room.get('width_m', 5.0) if room['sensor_wall'] in {'Back', 'Front'} else room.get('depth_m', 5.0)
    room['sensor_position_m'] = min(float(room.get('sensor_position_m', 0)), wall_length)
    room['sensor_height_m'] = min(float(room.get('sensor_height_m', 1.4)), float(room.get('height_m', 2.7)))
    _write_json_file(RADAR_ROOM_CONFIG, room)
    return jsonify({'success': True, 'room': room})


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Device settings page."""
    if 'username' not in session:
        return redirect(url_for('login', next=url_for('settings')))

    if request.method == 'POST':
        payload = request.get_json(silent=True) or request.form
        updates = {
            'portal_upload_allowed': str(payload.get('portal_upload_allowed', '')).lower() in {'1', 'true', 'on', 'yes'},
            'cloud_sync_allowed': str(payload.get('cloud_sync_allowed', '')).lower() in {'1', 'true', 'on', 'yes'},
            'radar_detection_threshold_db': payload.get('radar_detection_threshold_db', 8.0),
            'auto_occupancy_label_enabled': str(payload.get('auto_occupancy_label_enabled', '')).lower() in {'1', 'true', 'on', 'yes'},
            'chunk_seconds': payload.get('chunk_seconds', 10.0),
            'system_mode': payload.get('system_mode', 'balanced'),
            'labels': payload.get('labels', []),
            'prediction_label_style': payload.get('prediction_label_style', 'occupancy'),
            'people_count_label_enabled': str(payload.get('people_count_label_enabled', '')).lower() in {'1', 'true', 'on', 'yes'},
            'sleep_study_enabled': str(payload.get('sleep_study_enabled', '')).lower() in {'1', 'true', 'on', 'yes'},
            'csi_device_ids': payload.get('csi_device_ids', {}),
        }
        try:
            saved = device_manager.save_device_settings(updates)
            home_assistant = save_home_assistant_config({
                'enabled': str(payload.get('home_assistant_enabled', '')).lower() in {'1', 'true', 'on', 'yes'},
                'base_url': payload.get('home_assistant_base_url'),
                'entity_id': payload.get('home_assistant_entity_id'),
                'token': payload.get('home_assistant_token'),
            })
        except (OSError, ValueError, TypeError) as exc:
            logger.error('Settings persistence failed: %s', exc)
            if request.is_json:
                return jsonify({'success': False, 'message': f'Unable to persist settings: {exc}'}), 500
            flash(f'Unable to persist settings: {exc}', 'error')
            return redirect(url_for('settings'))
        if request.is_json:
            return jsonify({'success': True, 'settings': saved, 'home_assistant': home_assistant, 'sync_status': saved.get('sync_status')})
        flash('Sync pending' if saved.get('sync_pending') else 'Settings saved', 'success')
        return redirect(url_for('settings'))

    return render_template(
        'settings.html',
        username=session.get('username'),
        device_settings=device_manager.get_device_settings(),
        home_assistant=load_home_assistant_config(),
        csi_devices=next((sensor.get('devices', []) for sensor in detect_sensor_inventory() if sensor.get('key') == 'esp32_csi'), []),
    )


@app.route('/api/settings', methods=['GET', 'PATCH'])
def api_settings():
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    if request.method == 'GET':
        return jsonify({'success': True, 'settings': device_manager.get_device_settings()})
    payload = request.get_json(silent=True) or {}
    try:
        saved = device_manager.save_device_settings(payload)
    except (OSError, ValueError, TypeError) as exc:
        return jsonify({'success': False, 'message': f'Unable to persist settings: {exc}'}), 500
    return jsonify({'success': True, 'settings': saved})


@app.route('/api/settings/home-assistant/test', methods=['POST'])
def api_home_assistant_test():
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    result = test_home_assistant_connection()
    return jsonify(result), (200 if result.get('success') else 502)


@app.route('/api/device/identity', methods=['PUT'])
def api_device_identity():
    payload = request.get_json(silent=True) or {}
    try:
        device_name = device_manager.save_device_name(payload.get('device_id') or payload.get('device_name'))
    except (OSError, ValueError, TypeError) as exc:
        return jsonify({'success': False, 'message': str(exc)}), 422
    return jsonify({'success': True, 'device_id': device_name, 'device_name': device_name})


@app.route('/api/sensors', methods=['GET'])
def api_sensor_inventory():
    """Return hot-plug-aware sensor inventory for dashboard polling."""
    sensors = detect_sensor_inventory()
    return jsonify({'success': True, 'sensors': sensors})


@app.route('/api/internal/capture-chunk', methods=['POST'])
def api_internal_capture_chunk():
    """Forward one compact analyzed chunk to Brain."""
    if request.remote_addr not in {'127.0.0.1', '::1', None}:
        return jsonify({'success': False, 'message': 'Local requests only'}), 403
    payload = request.get_json(silent=True) or {}
    if not payload.get('minute') or payload.get('chunk_index') is None:
        return jsonify({'success': False, 'message': 'minute and chunk_index are required'}), 400
    success = device_manager.publish_capture_chunk(payload)
    return jsonify({'success': success}), (202 if success else 503)


@app.route('/api/internal/home-assistant/publish', methods=['POST'])
def api_internal_home_assistant_publish():
    """Accept a live chunk from the local collector and return immediately."""
    if request.remote_addr not in {'127.0.0.1', '::1', None}:
        return jsonify({'success': False, 'message': 'Local requests only'}), 403
    payload = request.get_json(silent=True) or {}
    minute = str(payload.get('minute') or '')
    occupancy = payload.get('occupancy') if isinstance(payload.get('occupancy'), dict) else {}
    scope = 'minute' if payload.get('scope') == 'minute' else 'chunk'
    try:
        chunk_index = int(payload.get('chunk_index')) if scope == 'chunk' else None
    except (TypeError, ValueError):
        return jsonify({'success': False, 'message': 'A valid chunk_index is required'}), 400

    def record_result(result: Dict[str, Any]) -> None:
        minute_dir = get_minute(minute)
        if not minute_dir:
            return
        manifest_path = minute_dir / 'manifest.json'
        with _home_assistant_manifest_lock:
            status_path = minute_dir / '.home_assistant_status.json'
            statuses = _read_json_file(status_path, {})
            status_key = str(chunk_index) if chunk_index is not None else 'minute'
            statuses[status_key] = {**result, 'updated_at': datetime.now(timezone.utc).isoformat()}
            _write_json_file(status_path, statuses)
            manifest = _read_json_file(manifest_path, {})
            # The collector is the sole manifest writer until finalization. Its
            # next atomic snapshot merges this sidecar status. This avoids a
            # cross-process lost-update race on the live manifest.
            if not manifest.get('folder_minute') or not manifest.get('capture_finished'):
                return
            if chunk_index is None:
                manifest['home_assistant'] = statuses[status_key]
            chunks = (((manifest.get('outputs') or {}).get('radar') or {}).get('chunks') or [])
            for chunk in chunks:
                if chunk_index is not None and isinstance(chunk, dict) and int(chunk.get('chunk_index', -1)) == chunk_index:
                    chunk['home_assistant'] = statuses[status_key]
                    break
            _write_json_file(manifest_path, manifest)

    queued = get_home_assistant_publisher().submit(
        occupancy,
        minute,
        callback=record_result,
        chunk_index=chunk_index,
        location=payload.get('location'),
        confidence=payload.get('confidence'),
        targets=payload.get('targets'),
        scope=scope,
        people_count=payload.get('people_count'),
        labels=payload.get('labels'),
        activity_labels=payload.get('activity_labels'),
        activity=payload.get('activity'),
        timestamp=payload.get('timestamp'),
    )
    return jsonify({'success': queued, 'status': 'queued' if queued else 'queue_full'}), (202 if queued else 503)


@app.route('/captures')
def captures():
    """Keep old capture links working after moving the archive home."""
    return redirect(url_for('status', _anchor='captures'))


def _capture_timeline_items(limit: Optional[int] = None) -> list[Dict[str, Any]]:
    """Return every stored minute in stable timestamp order."""
    minute_dirs = sorted(list_minute_folders(), key=lambda path: path.name)
    if limit is not None:
        normalized_limit = max(1, min(10000, int(limit)))
        minute_dirs = minute_dirs[-normalized_limit:]
    signatures = []
    for minute_dir in minute_dirs:
        manifest_path = minute_dir / 'manifest.json'
        try:
            stat = manifest_path.stat()
            signatures.append((str(manifest_path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            signatures.append((str(manifest_path), 0, 0))
    signature = tuple(signatures)
    if _capture_timeline_cache['signature'] == signature:
        return list(_capture_timeline_cache['items'])

    compact = []
    active_cache_keys = set()
    for minute_dir, manifest_signature in zip(minute_dirs, signatures):
        manifest_path = minute_dir / 'manifest.json'
        cache_key = str(manifest_path)
        active_cache_keys.add(cache_key)
        cached = _capture_manifest_cache.get(cache_key)
        if not cached or cached.get('signature') != manifest_signature[1:]:
            cached = {
                'signature': manifest_signature[1:],
                'summary': _read_timeline_manifest_summary(manifest_path),
            }
            _capture_manifest_cache[cache_key] = cached
        summary = cached['summary']
        compact.append({
            'minute': minute_dir.name,
            'modified': datetime.fromtimestamp(minute_dir.stat().st_mtime).isoformat(),
            'latest_chunk': summary,
        })
    for cache_key in set(_capture_manifest_cache) - active_cache_keys:
        _capture_manifest_cache.pop(cache_key, None)
    _capture_timeline_cache.update({'signature': signature, 'items': compact})
    return list(compact)


def _read_timeline_manifest_summary(path: Path) -> Optional[Dict[str, Any]]:
    """Read only the bounded manifest tail needed for one archive dot."""
    try:
        size = path.stat().st_size
        with path.open('rb') as handle:
            handle.seek(max(0, size - 262144))
            tail = handle.read().decode('utf-8', errors='ignore')
    except OSError:
        return None

    minute_match = re.search(
        r'"minute_summary"\s*:\s*\{.*?"occupancy"\s*:\s*\{.*?'
        r'"label"\s*:\s*"(occupied|empty)"',
        tail,
        flags=re.DOTALL,
    )
    state = minute_match.group(1) if minute_match else None
    chunk_matches = re.findall(
        r'"chunk_index"\s*:\s*(\d+)(?:(?!"chunk_index").){0,6000}?'
        r'"status"\s*:\s*"(occupied|empty)"',
        tail,
        flags=re.DOTALL,
    )
    if chunk_matches:
        chunk_index, chunk_state = max(chunk_matches, key=lambda item: int(item[0]))
        state = chunk_state
        index = int(chunk_index)
    else:
        index = None
    if state not in {'occupied', 'empty'}:
        return None
    return {
        'index': index,
        'state': state,
        'classification': 'green' if state == 'occupied' else 'red',
        'prediction': state,
    }


@app.route('/api/captures/timeline', methods=['GET'])
def api_capture_timeline():
    requested_limit = request.args.get('limit')
    minutes = _capture_timeline_items(int(requested_limit)) if requested_limit else _capture_timeline_items()
    system_status = get_system_status(update_remote=False)
    active = current_minute() if system_status.collection_active else None
    return jsonify({
        'status': 'success',
        'minutes': minutes,
        'count': len(minutes),
        'active_minute': active.name if active else None,
    })


def _capture_runtime_stats() -> Dict[str, Any]:
    status = get_system_status(update_remote=False)
    disk_io = psutil.disk_io_counters()
    net_io = psutil.net_io_counters()
    memory = psutil.virtual_memory()
    latest = list_minutes()[:1]
    latest_summary = latest[0] if latest else None
    latest_metrics = {}
    if latest_summary:
        minute_dir = get_minute(str(latest_summary.get('relative_path') or latest_summary.get('minute')))
        if minute_dir:
            latest_metrics = minute_metrics(minute_dir)

    return {
        'timestamp': datetime.utcnow().isoformat(),
        'collection_active': status.collection_active,
        'cpu_percent': psutil.cpu_percent(interval=None),
        'memory_percent': memory.percent,
        'temperature_c': status.temperature,
        'disk_percent': status.disk_usage.get('percent_used') if status.disk_usage else None,
        'disk_read_mb': round((disk_io.read_bytes if disk_io else 0) / (1024 * 1024), 1),
        'disk_write_mb': round((disk_io.write_bytes if disk_io else 0) / (1024 * 1024), 1),
        'network_rx_mb': round((net_io.bytes_recv if net_io else 0) / (1024 * 1024), 1),
        'network_tx_mb': round((net_io.bytes_sent if net_io else 0) / (1024 * 1024), 1),
        'latest_minute': latest_summary,
        'latest_metrics': latest_metrics,
    }


@app.route('/api/capture-settings', methods=['GET', 'PUT'])
def capture_settings_api():
    """Get or update settings used by ongoing minute collection."""
    if request.method == 'GET':
        return jsonify({'success': True, 'capture_settings': device_manager.load_capture_settings()})

    payload = request.get_json(silent=True) or {}
    try:
        settings = device_manager.save_and_sync_capture_settings(payload)
    except (OSError, ValueError, TypeError) as exc:
        return jsonify({'success': False, 'message': f'Unable to persist capture settings: {exc}'}), 500
    return jsonify({'success': True, 'capture_settings': settings, 'sync_status': settings.get('sync_status'), 'message': 'Capture settings updated'})


@app.route('/api/captures/stats')
def capture_stats_api():
    """Return realtime system and latest capture statistics."""
    return jsonify({'success': True, 'stats': _capture_runtime_stats()})


@app.route('/captures/<minute>')
def capture_detail(minute):
    """Show a single minute capture."""
    minute_dir = get_minute(minute)
    if not minute_dir:
        abort(404, description='Minute folder not found')

    files = capture_files(minute_dir)
    detail = minute_summary(minute_dir)
    for chunk in (detail.get("progress") or {}).get("chunks", []):
        chunk.pop("xy_map", None)
    metrics = minute_metrics(minute_dir)
    video_preview = f"/api/captures/{minute}/file/video" if files.get("video") else None
    camera_preview = (
        f"/api/captures/{minute}/video/frame"
        if files.get("camera_images") or files.get("container") else None
    )
    radar_preview = preview_text(files.get("radar"), 12000)
    csi_preview = preview_text(files.get("csi_timestamped") or files.get("csi_csv"), 12000)
    serial_preview = preview_text(files.get("csi_serial"), 12000)

    return render_template(
        'capture_detail.html',
        minute=minute_dir.name,
        capture=minute_dir,
        minute_info=detail,
        files=files,
        capture_metrics=metrics,
        video_url=video_preview,
        camera_frame_url=camera_preview,
        manifest_url=url_for('api_capture_file', minute=minute_dir.name, kind='manifest'),
        xy_tracking_url=url_for('api_capture_radar_data', minute=minute_dir.name, plot='xy-tracking'),
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


def _requested_capture_minutes() -> tuple[list[str], list[Path], tuple[Response, int] | None]:
    payload = request.get_json(silent=True) or {}
    if not request.is_json and request.form.get('minutes'):
        try:
            payload = {'minutes': json.loads(request.form['minutes'])}
        except (TypeError, json.JSONDecodeError):
            payload = {'minutes': []}
    requested = payload.get('minutes', [])
    if not isinstance(requested, list):
        return [], [], (jsonify({'status': 'error', 'message': 'minutes must be a list'}), 400)

    names = list(dict.fromkeys(str(item).strip() for item in requested if str(item).strip()))
    if not names:
        return [], [], (jsonify({'status': 'error', 'message': 'Select at least one capture'}), 400)
    minute_dirs: list[Path] = []
    missing: list[str] = []
    for name in names:
        minute_dir = get_minute(name)
        if minute_dir is None:
            missing.append(name)
        else:
            minute_dirs.append(minute_dir)
    if missing:
        return [], [], (jsonify({
            'status': 'error',
            'message': f"Capture not found: {', '.join(missing)}",
            'missing': missing,
        }), 404)
    return names, minute_dirs, None


@app.route('/api/captures/bulk-download', methods=['POST'])
def bulk_download_captures():
    """Stream selected capture folders in one ZIP archive."""
    names, minute_dirs, error = _requested_capture_minutes()
    if error:
        return error

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return Response(
        stream_minute_folders(minute_dirs),
        mimetype='application/zip',
        headers={
            'Content-Disposition': f'attachment; filename="thoth-captures-{timestamp}.zip"',
            'Cache-Control': 'no-store',
            'X-Archive-Mode': 'streaming',
        },
        direct_passthrough=True,
    )


@app.route('/api/captures/bulk-delete', methods=['POST'])
def bulk_delete_captures():
    """Delete selected completed capture folders as one validated operation."""
    names, minute_dirs, error = _requested_capture_minutes()
    if error:
        return error

    active = current_minute()
    collection_active = get_system_status(update_remote=False).collection_active
    blocked = []
    for minute_dir in minute_dirs:
        summary = minute_summary(minute_dir)
        if active and active == minute_dir and not summary.get('capture_finished') and collection_active:
            blocked.append(minute_dir.name)
    if blocked:
        return jsonify({
            'status': 'error',
            'message': 'An actively recording capture cannot be deleted.',
            'blocked': blocked,
        }), 409

    deleted = []
    try:
        for name, minute_dir in zip(names, minute_dirs):
            shutil.rmtree(minute_dir)
            deleted.append(name)
    except Exception as exc:
        logger.exception('Bulk capture delete failed: %s', exc)
        return jsonify({
            'status': 'error',
            'message': f'Bulk delete stopped after deleting {len(deleted)} captures: {exc}',
            'deleted': deleted,
        }), 500

    return jsonify({'status': 'success', 'deleted': deleted, 'count': len(deleted)})


@app.route('/api/captures', methods=['GET'])
def api_list_captures():
    """Return the capture minute index."""
    minutes = list_minutes()
    for minute in minutes:
        name = str(minute.get('minute') or '')
        if not name:
            continue
        minute['urls'] = {
            'open': url_for('capture_detail', minute=name),
            'download': url_for('download_capture_minute', minute=name),
            'upload': url_for('upload_capture_minute', minute=name),
            'delete': url_for('delete_capture_minute', minute=name),
        }
    return jsonify({
        'status': 'success',
        'minutes': minutes,
        'count': len(minutes),
        'active_minute': current_minute().name if current_minute() else None,
        'capture_dir': Config.CAPTURE_DATA_DIR,
        'max_disk_percent': Config.CAPTURE_MAX_DISK_PERCENT,
    })


@app.route('/api/captures/<minute>', methods=['GET'])
def api_capture_detail(minute):
    """Return the metadata for a single capture minute."""
    minute_dir = get_minute(minute)
    if not minute_dir:
        return jsonify({'status': 'error', 'message': 'Minute folder not found'}), 404

    files = capture_files(minute_dir)
    detail = minute_summary(minute_dir)
    for chunk in (detail.get('progress') or {}).get('chunks', []):
        chunk.pop('xy_map', None)
    if request.args.get('compact') in {'1', 'true', 'yes'}:
        response = jsonify({'status': 'success', 'capture': {
            'minute': detail.get('minute'),
            'capture_finished': detail.get('capture_finished'),
            'progress': detail.get('progress') or {},
        }})
        response.headers['Cache-Control'] = 'no-store, max-age=0'
        return response
    metrics = minute_metrics(minute_dir)
    detail['files_on_disk'] = {
        key: ([str(path) for path in value] if isinstance(value, list) else str(value) if value else None)
        for key, value in files.items()
    }
    detail['metrics'] = metrics
    return jsonify({'status': 'success', 'capture': detail})


@app.route('/api/captures/<minute>/labels', methods=['PATCH'])
def api_capture_labels(minute):
    """Update labels for a capture minute."""
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Authentication required'}), 401

    minute_dir = get_minute(minute)
    if not minute_dir:
        return jsonify({'status': 'error', 'message': 'Minute folder not found'}), 404

    payload = request.get_json(silent=True) or {}
    labels = payload.get('labels', [])
    replace = bool(payload.get('replace', True))

    try:
        updated = update_minute_labels(minute_dir, labels, replace=replace)
    except Exception as exc:
        logger.exception("Failed to update labels for %s: %s", minute, exc)
        return jsonify({'status': 'error', 'message': 'Failed to update labels'}), 500

    return jsonify({'status': 'success', 'minute': minute, 'labels': updated})


@app.route('/api/captures/<minute>', methods=['DELETE'])
def delete_capture_minute(minute):
    """Delete a completed capture minute folder."""
    minute_dir = get_minute(minute)
    if not minute_dir:
        return jsonify({'status': 'error', 'message': 'Minute folder not found'}), 404

    summary = minute_summary(minute_dir)
    active = current_minute()
    if active and active == minute_dir and not summary.get('capture_finished') and get_system_status(update_remote=False).collection_active:
        return jsonify({
            'status': 'error',
            'message': 'This minute still appears to be recording. Wait for it to finish before deleting.',
        }), 409

    try:
        relative_path = summary.get('relative_path') or minute_dir.name
        shutil.rmtree(minute_dir)
    except Exception as exc:
        logger.exception("Failed to delete capture %s: %s", minute, exc)
        return jsonify({'status': 'error', 'message': f'Failed to delete capture: {exc}'}), 500

    return jsonify({'status': 'success', 'minute': minute, 'relative_path': relative_path})


@app.route('/api/captures/<minute>/video/frame')
def api_capture_video_frame(minute):
    """Render a JPEG preview from a specific minute's video file."""
    minute_dir = get_minute(minute)
    if not minute_dir:
        abort(404, description='Minute folder not found')

    files = capture_files(minute_dir)
    video_path = files.get('video')
    try:
        requested_second = max(0, int(request.args.get('second', 0)))
        container = files.get('container')
        if container and container.exists():
            jpeg_bytes = read_camera_frame(container, requested_second) or first_camera_frame(container)
            if not jpeg_bytes:
                abort(404, description='No camera frame available for this minute')
        elif files.get('camera_images'):
            images = files['camera_images']
            jpeg_bytes = images[min(requested_second, len(images) - 1)].read_bytes()
        elif video_path and video_path.exists():
            jpeg_bytes = _render_video_frame(video_path)
        else:
            abort(404, description='No camera data available for this minute')
    except Exception as exc:
        logger.exception(f'Failed to render minute video frame {minute}: {exc}')
        abort(500, description=str(exc))

    return Response(jpeg_bytes, mimetype='image/jpeg', headers={'Cache-Control': 'no-cache'})


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
        'xy_tracking': files.get('xy_tracking'),
        'csi_csv': files.get('csi_csv'),
        'csi_timestamped': files.get('csi_timestamped'),
        'csi_serial': files.get('csi_serial'),
        'manifest': files.get('manifest'),
        'predictions': files.get('predictions'),
        'log': files.get('video_log'),
        'container': files.get('container'),
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
    elif file_path.suffix == '.npz':
        mime = 'application/x-npz'

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
    path = files.get('csi_csv') or files.get('csi_timestamped') or files.get('csi_serial') or files.get('container')
    if not path or not path.exists():
        abort(404, description='No CSI data available')

    points = _parse_csi_average_series(path)
    svg = _build_csi_svg(points)
    return Response(svg, mimetype='image/svg+xml', headers={'Cache-Control': 'no-cache'})


@app.route('/api/captures/<minute>/csi/data')
def api_capture_csi_data(minute):
    """Return interactive CSI data for a saved minute."""
    minute_dir = get_minute(minute)
    if not minute_dir:
        abort(404, description='Minute folder not found')

    files = capture_files(minute_dir)
    path = files.get('csi_csv') or files.get('csi_timestamped') or files.get('csi_serial') or files.get('container')
    if not path or not path.exists():
        abort(404, description='No CSI data available')

    data = _csi_plot_payload(path)
    data['metadata'] = minute_metrics(minute_dir)
    return jsonify({'status': 'success', 'minute': minute, 'data': data})


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
    radar_path = files.get('radar') or files.get('container')
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
    minute_dir = get_minute(minute)
    if not minute_dir:
        abort(404, description='Minute folder not found')

    plot = plot.lower()
    if plot not in RADAR_PLOTS:
        abort(404, description=f'Unsupported radar plot kind: {plot}')

    try:
        files = capture_files(minute_dir)
        payload = _load_xy_tracking_payload(minute_dir, files)
    except Exception as exc:
        logger.exception(f"Failed to build xy data {plot}: {exc}")
        abort(500, description=str(exc))

    requested_chunk = request.args.get('chunk')
    if requested_chunk is not None:
        try:
            chunk_index = int(requested_chunk)
        except (TypeError, ValueError):
            return jsonify({'status': 'error', 'message': 'chunk must be a non-negative integer'}), 400
        if chunk_index < 0:
            return jsonify({'status': 'error', 'message': 'chunk must be a non-negative integer'}), 400

        frames = []
        for frame in payload.get('frames') or []:
            if not isinstance(frame, dict):
                continue
            try:
                frame_chunk_index = int(frame.get('chunk_index', -1))
            except (TypeError, ValueError):
                continue
            if frame_chunk_index == chunk_index:
                frames.append(frame)
        if not frames:
            radar_bins = files.get('radar_bins') or []
            if chunk_index < len(radar_bins):
                try:
                    chunk_payload = _radar_plot_payload(radar_bins[chunk_index], plot)
                    frames = list((chunk_payload or {}).get('frames') or [])
                    payload = chunk_payload if isinstance(chunk_payload, dict) else payload
                except Exception as exc:
                    logger.debug('Chunk %s is not ready for playback: %s', chunk_index, exc)

        chunk_payload = dict(payload)
        chunk_payload['frames'] = frames[:10]
        chunk_payload['z'] = [] if frames else chunk_payload.get('z', [])
        chunk_payload['chunk_index'] = chunk_index
        chunk_payload['frame_count'] = len(chunk_payload['frames'])
        chunk_payload['sample_count'] = len(chunk_payload['frames'])
        chunk_payload['expected_frame_count'] = 10
        chunk_payload['loading'] = len(chunk_payload['frames']) < 10
        response = jsonify({'status': 'success', 'minute': minute, 'data': chunk_payload})
        response.headers['Cache-Control'] = 'no-store, max-age=0'
        return response

    payload['metadata'] = minute_metrics(minute_dir)
    return jsonify({'status': 'success', 'minute': minute, 'data': payload})


@app.route('/api/captures/<minute>/upload', methods=['POST'])
def upload_capture_minute(minute):
    """Upload all files in a capture minute to Brain."""
    auth_token = (
        getattr(Config, 'USER_AUTH_TOKEN', None)
        or getattr(device_manager, 'auth_token', None)
        or getattr(Config, 'BRAIN_AUTH_TOKEN', None)
    )
    auth_token = auth_token.strip() if isinstance(auth_token, str) else auth_token
    if not auth_token:
        return jsonify({'status': 'error', 'message': 'Not authenticated'}), 401

    minute_dir = get_minute(minute)
    if not minute_dir:
        return jsonify({'status': 'error', 'message': 'Minute folder not found'}), 404

    summary = minute_summary(minute_dir)
    device_id = getattr(device_manager, 'device_id', None)
    uploaded = []
    errors = []
    skipped = []
    import base64

    has_container = (minute_dir / 'capture.npz').exists()
    all_files = sorted(
        path for path in minute_dir.iterdir()
        if path.is_file()
        and (
            path.name in {'manifest.json', 'capture.npz', 'xy-tracking.json', 'predictions.json'}
            or (not has_container and path.name == 'wifi_csi.csv')
            or (not has_container and path.name.startswith('radar_') and path.suffix == '.bin')
            or (not has_container and path.name.startswith('camera_') and path.suffix.lower() in {'.jpg', '.jpeg'})
        )
    )

    for path in all_files:
        if path.name == 'usb_camera.ffmpeg.log' and path.stat().st_size == 0:
            skipped.append(path.name)
            continue
        if path.name == 'xy-tracking.json':
            key = 'xy-tracking'
        elif path.name == 'predictions.json':
            key = 'predictions'
        elif path.name == 'manifest.json':
            key = 'manifest'
        elif path.name == 'capture.npz':
            key = 'synchronized-container'
        elif path.name.startswith('radar_') or path.name.startswith('mmw_radar_raw_'):
            key = 'radar-bin'
        elif path.name.startswith('mmw_radar_xy_'):
            key = 'radar-csv'
        elif path.name.startswith('camera_'):
            key = 'camera-image'
        else:
            key = path.stem
        try:
            with open(path, 'rb') as handle:
                content = base64.b64encode(handle.read()).decode('utf-8')
            suffix = path.suffix.lower()
            if suffix in {'.jpg', '.jpeg'}:
                content_type = 'image/jpeg'
            elif suffix == '.mp4':
                content_type = 'video/mp4'
            elif suffix == '.csv':
                content_type = 'text/csv'
            elif suffix == '.json':
                content_type = 'application/json'
            elif suffix == '.jsonl':
                content_type = 'application/x-ndjson'
            elif suffix == '.npz':
                content_type = 'application/x-npz'
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
                'device_id': device_id,
                'content_type': content_type,
                'metadata': {
                    'source': 'thoth_device',
                    'device_id': device_id,
                    'minute': minute_dir.name,
                    'relative_path': summary.get('relative_path'),
                    'file_kind': key,
                    'size_bytes': path.stat().st_size,
                },
            }
            response = requests.post(f"{Config.BRAIN_SERVER_URL}/api/file/upload", json=payload, headers=headers, timeout=120)
            if response.status_code == 404:
                response = requests.post(f"{Config.BRAIN_SERVER_URL}/file/upload", json=payload, headers=headers, timeout=120)
            if response.status_code in (200, 201):
                uploaded.append({'name': path.name, 'bytes': path.stat().st_size})
            else:
                detail = response.text[:300] if response.text else ''
                errors.append(f"{path.name}: {response.status_code} {detail}".strip())
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    # The minute-named manifest completes an explicit full-upload request and
    # is the only operation that marks the minute as cloud uploaded.
    try:
        raw = json.dumps(summary, default=str).encode('utf-8')
        response = requests.post(
            f"{Config.BRAIN_SERVER_URL}/api/file/upload",
            json={
                'filename': minute_dir.name,
                'content': base64.b64encode(raw).decode('ascii'),
                'is_base64': True,
                'device_id': device_id,
                'content_type': 'application/json',
                'metadata': {'source': 'thoth_device', 'minute': minute_dir.name, 'file_kind': 'capture-manifest'},
            },
            headers={'Authorization': f'Bearer {auth_token}', 'Content-Type': 'application/json'},
            timeout=120,
        )
        if response.status_code in (200, 201):
            uploaded.append({'name': minute_dir.name, 'bytes': len(raw)})
        else:
            errors.append(f'{minute_dir.name}: {response.status_code} {response.text[:300]}')
    except Exception as exc:
        errors.append(f'{minute_dir.name}: {exc}')

    return jsonify({
        'status': 'success' if not errors else 'partial',
        'minute': minute_dir.name,
        'uploaded': uploaded,
        'skipped': skipped,
        'errors': errors,
    })


@app.route('/captures/live/<kind>')
def live_capture_stream(kind):
    """Render the live sensor view for the selected modality."""
    kind = kind.lower()
    if kind not in {'video', 'csi', 'radar'}:
        abort(404, description=f'Unsupported live stream kind: {kind}')

    minute_dir, files = _best_live_minute_for_kind(kind)
    summary = minute_summary(minute_dir) if minute_dir else {}
    summary_files = summary.get('files') if isinstance(summary.get('files'), dict) else {}
    has_video = bool(summary_files.get('video'))
    has_csi = bool(summary_files.get('csi'))
    has_radar = bool(summary_files.get('radar'))
    live_metrics = minute_metrics(minute_dir) if minute_dir else None

    return render_template(
        'live_stream.html',
        kind=kind,
        active_minute=minute_dir.name if minute_dir else None,
        username=session.get('username'),
        live_metrics=live_metrics,
        video_url=url_for('api_live_capture_video'),
        video_frame_url=url_for('api_live_capture_video_frame'),
        csi_data_url=url_for('api_live_capture_csi_data'),
        xy_tracking_url=url_for('api_live_capture_radar_data', plot='xy-tracking'),
        has_video=has_video,
        has_mp4=bool(files.get('video') and files['video'].exists()) if minute_dir else False,
        has_csi=has_csi,
        has_radar=has_radar,
    )


@app.route('/api/captures/live/video')
def api_live_capture_video():
    """Serve the current minute video file."""
    minute_dir, files = _best_live_minute_for_kind('video')
    video_path = files.get('video') if files else None
    if not minute_dir or not video_path or not video_path.exists():
        abort(404, description='No live video available')

    return send_file(video_path, mimetype='video/mp4', conditional=True, max_age=0)


@app.route('/api/captures/live/video/frame')
def api_live_capture_video_frame():
    """Render a live JPEG preview from the current or latest video minute."""
    minute_dir, files = _best_live_minute_for_kind('video')
    video_path = files.get('video') if files else None
    if not minute_dir:
        abort(404, description='No live video available')

    try:
        images = files.get('camera_images') or []
        container = files.get('container')
        if images:
            jpeg_bytes = images[-1].read_bytes()
        elif container and container.exists():
            jpeg_bytes = first_camera_frame(container)
            if not jpeg_bytes:
                abort(404, description='No live camera frame available')
        elif video_path and video_path.exists():
            jpeg_bytes = _render_video_frame(video_path)
        else:
            abort(404, description='No live camera frame available')
    except Exception as exc:
        logger.exception(f"Failed to render live video frame: {exc}")
        abort(500, description=str(exc))

    return Response(jpeg_bytes, mimetype='image/jpeg', headers={'Cache-Control': 'no-cache'})


@app.route('/api/captures/live/csi')
def api_live_capture_csi():
    """Redirect to the live CSI plot view."""
    return redirect(url_for('live_capture_stream', kind='csi'))


@app.route('/api/captures/live/radar')
def api_live_capture_radar():
    """Redirect to the live radar plot view."""
    return redirect(url_for('live_capture_stream', kind='radar'))


@app.route('/api/captures/live/csi/data')
def api_live_capture_csi_data():
    """Return interactive live CSI data."""
    minute_dir, files = _best_live_minute_for_kind('csi')
    if not minute_dir:
        abort(404, description='No live capture minute available')

    path = files.get('csi_csv') or files.get('csi_timestamped') or files.get('csi_serial') or files.get('container')
    if not path or not path.exists():
        abort(404, description='No CSI data available')

    data = _csi_plot_payload(path)
    data['metadata'] = minute_metrics(minute_dir)
    return jsonify({'status': 'success', 'minute': minute_dir.name, 'data': data})


@app.route('/api/captures/live/csi/plot')
def api_live_capture_csi_plot():
    """Render a live CSI amplitude SVG from the active minute."""
    if request.method == 'HEAD':
        return Response(status=200)

    minute_dir, files = _best_live_minute_for_kind('csi')
    if not minute_dir:
        abort(404, description='No live capture minute available')

    path = files.get('csi_csv') or files.get('csi_timestamped') or files.get('csi_serial') or files.get('container')
    if not path or not path.exists():
        abort(404, description='No CSI data available')

    points = _parse_csi_average_series(path)
    svg = _build_csi_svg(points)
    return Response(svg, mimetype='image/svg+xml', headers={'Cache-Control': 'no-cache'})


@app.route('/api/captures/live/radar/data/<plot>')
def api_live_capture_radar_data(plot):
    """Return the bounded, newest live radar window."""
    plot = plot.lower()
    if plot not in RADAR_PLOTS:
        abort(404, description=f'Unsupported radar plot kind: {plot}')

    minute_dir, files = _best_live_minute_for_kind('radar')
    if not minute_dir:
        abort(404, description='No live capture minute available')

    try:
        payload = _live_xy_window(_load_xy_tracking_payload(minute_dir, files))
    except Exception as exc:
        logger.exception(f"Failed to build live xy data {plot}: {exc}")
        abort(500, description=str(exc))

    response = jsonify({'status': 'success', 'minute': minute_dir.name, 'data': payload})
    response.headers['Cache-Control'] = 'no-store, max-age=0'
    return response


@app.route('/api/captures/live/radar/plot/<plot>')
def api_live_capture_plot(plot):
    """Render the current minute radar plot as PNG."""
    if request.method == 'HEAD':
        return Response(status=200)

    plot = plot.lower()
    if plot not in RADAR_PLOTS:
        abort(404, description=f'Unsupported radar plot kind: {plot}')

    minute_dir, files = _best_live_minute_for_kind('radar')
    if not minute_dir:
        abort(404, description='No live capture minute available')

    radar_path = files.get('radar') or files.get('container')
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
    persistent_token = getattr(Config, 'BRAIN_AUTH_TOKEN', None)
    if not persistent_token:
        try:
            device_manager.mark_device_offline()
        except Exception as e:
            logger.error(f"Error updating device status on logout: {e}")

    if not persistent_token:
        try:
            auth_manager.logout()
        except Exception as e:
            logger.error(f"Error clearing auth state on logout: {e}")

    Config.USER_AUTH_TOKEN = persistent_token

    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/wifi/config', methods=['POST'])
def wifi_config():
    """Legacy WiFi configuration endpoint removed."""
    return jsonify({"status": "error", "error": "WiFi setup is handled through Raspberry Pi Imager."}), 410

@app.route('/api/collection/<action>', methods=['POST'])
def collection_action(action):
    """Start or stop data collection."""
    global collection_active, collection_process

    if action == 'start':
        payload = request.get_json(silent=True) or {}
        sensors = payload.get('sensors') or {}
        label = str(payload.get('label') or '').strip()
        if label or sensors:
            device_manager.save_and_sync_capture_settings({'labels': [label] if label else [], 'sensors': sensors})
        COLLECTOR_PAUSE_PATH.unlink(missing_ok=True)

        service_running = subprocess.run(
            ['systemctl', 'is-active', '--quiet', 'thoth-collector.service'],
            capture_output=True,
        ).returncode == 0
        if service_running:
            collection_active = True
            return jsonify({'status': 'success', 'message': 'Continuous collection resumed'})
        if collection_process is not None and collection_process.poll() is None:
            collection_active = True
            return jsonify({'status': 'success', 'message': 'Collection already running'})

        command = [sys.executable, str(THOTH_ROOT / 'src' / 'collector.py')]

        try:
            collection_process = subprocess.Popen(
                command,
                cwd=str(THOTH_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            logger.exception('Failed to start minute collection: %s', exc)
            return jsonify({'status': 'error', 'message': 'Failed to start collection'}), 500

        collection_active = True
        logger.info('Data collection started: %s', ' '.join(command))
        return jsonify({'status': 'success', 'message': 'Collection started', 'pid': collection_process.pid})
    elif action == 'stop':
        COLLECTOR_PAUSE_PATH.parent.mkdir(parents=True, exist_ok=True)
        COLLECTOR_PAUSE_PATH.write_text(datetime.utcnow().isoformat(), encoding='utf-8')
        if collection_process is not None and collection_process.poll() is None:
            try:
                collection_process.terminate()
                collection_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                collection_process.kill()
                collection_process.wait(timeout=5)
            except Exception as exc:
                logger.warning('Failed to stop collection process cleanly: %s', exc)
        collection_process = None
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
