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
from datetime import datetime, timedelta
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
from backend.terminal_manager import SSHTerminalManager
from backend.sensor_detection import detect_sensor_inventory
from backend.capture_manager import (
    list_minutes,
    get_minute,
    capture_files,
    current_minute,
    minute_summary,
    minute_metrics,
    collect_prediction_timelines,
    cleanup_old_minutes,
    zip_minute_folder,
    update_minute_labels,
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

TRACK_EXAMPLE_DIR = MMW_RELEASE / 'example_2_track'
if str(TRACK_EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(TRACK_EXAMPLE_DIR))

try:
    from signal_proc import SigProc
    from utility.helper import parse_radar_cfg, read_uint12, split_samples
except Exception:  # pragma: no cover - import may fail on minimal installs
    SigProc = None
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

RADAR_PLOTS = ('range-doppler', 'azimuth-range', 'azimuth-doppler', 'xy-tracking')
RADAR_PLOT_AXES = {
    'range-doppler': ('Range', 'Doppler'),
    'azimuth-range': ('Azimuth', 'Range'),
    'azimuth-doppler': ('Azimuth', 'Doppler'),
    'xy-tracking': ('Y', 'X'),
}
RADAR_CONFIG_DIR = MMW_RELEASE / 'radar_config' / 'config_3rx_3m'
RADAR_TRACKING_CONFIG = TRACK_EXAMPLE_DIR / 'config' / 'processing_config.json'
CSI_NUMBER_RE = re.compile(r'[-+]?\d+(?:\.\d+)?')
RADAR_CACHE_VERSION = 3
_radar_cache_lock = threading.RLock()

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
terminal_manager = SSHTerminalManager(socketio, Config)

# Global state
collection_active = False
collection_process: Optional[subprocess.Popen] = None
COLLECTOR_PAUSE_PATH = THOTH_ROOT / 'config' / 'collector.pause'
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
        if kind == 'video':
            return bool(files.get('video') and files['video'].exists())
        if kind == 'csi':
            return bool(
                (files.get('csi_csv') and files['csi_csv'].exists())
                or (files.get('csi_timestamped') and files['csi_timestamped'].exists())
                or (files.get('csi_serial') and files['csi_serial'].exists())
            )
        if kind == 'radar':
            return bool(files.get('radar') and files['radar'].exists())
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
    if all((SigProc, parse_radar_cfg, read_uint12, split_samples)) and RADAR_TRACKING_CONFIG.exists():
        tracking_proc = SigProc(str(RADAR_TRACKING_CONFIG), parse_radar_cfg(setting))
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
            location_values = np.asarray(location, dtype=float)
            safe_location = [float(value) if np.isfinite(value) else None for value in location_values]
            xy_image = np.asarray(gui_plot['map'], dtype=float).T
            active = np.argwhere(xy_image > 0)
            frames_by_plot['xy-tracking'].append({
                'seq': seq,
                'index': index,
                'x': np.asarray(gui_plot['x_axis'], dtype=float).tolist(),
                'y': np.asarray(gui_plot['y_axis'], dtype=float).tolist(),
                'z_sparse': [[int(row), int(column), float(xy_image[row, column])] for row, column in active],
                'z_shape': [int(xy_image.shape[0]), int(xy_image.shape[1])],
                'location': safe_location,
                'score': float(score) if np.isfinite(score) else None,
                'detected': bool(detection.get('detected')),
                'snr_db': detection.get('snr_db'),
                'threshold_db': detection.get('threshold_db'),
                'peak_power_db': detection.get('peak_power_db'),
                'noise_floor_db': detection.get('noise_floor_db'),
            })

    bundle: Dict[str, Dict[str, Any]] = {}
    occupancy_ratio = detected_frames / evaluated_frames if evaluated_frames else 0.0
    occupancy = {
        'label': 'occupied' if occupancy_ratio > 0.5 else 'empty',
        'detected_frames': detected_frames,
        'evaluated_frames': evaluated_frames,
        'ratio': occupancy_ratio,
        'rule': 'occupied when more than 50% of radar frames contain a confirmed target',
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
            'frame_interval_ms': 120,
            'updated': datetime.utcnow().isoformat(),
            'occupancy': occupancy,
        }
        if plot == 'xy-tracking':
            bundle[plot]['title'] = 'X-Y Tracking'
            bundle[plot]['x_label'] = 'X (forward, m)'
            bundle[plot]['y_label'] = 'Y (lateral, m)'
            bundle[plot]['location'] = latest.get('location')
            bundle[plot]['score'] = latest.get('score')
            bundle[plot]['detected'] = latest.get('detected', False)
            bundle[plot]['snr_db'] = latest.get('snr_db')
            bundle[plot]['threshold_db'] = latest.get('threshold_db')
            bundle[plot]['peak_power_db'] = latest.get('peak_power_db')
            bundle[plot]['noise_floor_db'] = latest.get('noise_floor_db')
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
        detection_threshold_db = min(40.0, max(0.0, float(settings.get('radar_detection_threshold_db', 8.0))))
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
    if settings.get('auto_occupancy_label_enabled', True):
        occupancy = payload.get('occupancy') or {}
        label = occupancy.get('label')
        manifest_path = radar_path.parent / 'manifest.json'
        if label in {'empty', 'occupied'} and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                if manifest.get('auto_occupancy_label') != occupancy:
                    manifest['labels'] = [label]
                    manifest['primary_label'] = label
                    manifest['auto_occupancy_label'] = occupancy
                    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            except Exception as exc:
                logger.warning('Unable to save automatic occupancy label: %s', exc)
    return payload


def _prewarm_latest_radar_playback() -> None:
    """Process the newest completed minute before a user opens its plots."""
    if collection_active:
        return
    try:
        settings = device_manager.get_device_settings()
        threshold = min(40.0, max(0.0, float(settings.get('radar_detection_threshold_db', 8.0))))
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
    lambda: cleanup_old_minutes(Config.CAPTURE_KEEP_MINUTES),
    'interval',
    minutes=10,
    id='capture_cleanup',
    replace_existing=True
)
device_scheduler.add_job(
    _prewarm_latest_radar_playback,
    'interval',
    seconds=20,
    id='radar_playback_prewarm',
    max_instances=1,
    replace_existing=True,
)
device_scheduler.start()

# Load registration info if available
device_manager.load_registration_info()
cleanup_old_minutes(Config.CAPTURE_KEEP_MINUTES)
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

        wifi_state = get_active_wifi_state()
        capture_overview = get_capture_overview()
        sensors = detect_sensor_inventory()
        recent_minutes = list_minutes()[:6]

        return render_template('status.html',
                            system_status=system_status,
                            device_info=device_info,
                            disk_usage=disk_usage,
                            cpu_temp=cpu_temp,
                            wifi_state=wifi_state,
                            username=session.get('username'),
                            capture_overview=capture_overview,
                            recent_minutes=recent_minutes,
                            sensors=sensors,
                            device_settings=device_manager.get_device_settings())

    except Exception as e:
        logger.error(f"Error in status route: {str(e)}", exc_info=True)
        flash('An error occurred while loading the status page.', 'error')
        return redirect(url_for('index'))


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Device settings page."""
    if 'username' not in session:
        return redirect(url_for('login', next=url_for('settings')))

    if request.method == 'POST':
        payload = request.get_json(silent=True) or request.form
        updates = {
            'portal_upload_allowed': str(payload.get('portal_upload_allowed', '')).lower() in {'1', 'true', 'on', 'yes'},
            'deployment_requests_allowed': str(payload.get('deployment_requests_allowed', '')).lower() in {'1', 'true', 'on', 'yes'},
            'cloud_sync_allowed': str(payload.get('cloud_sync_allowed', '')).lower() in {'1', 'true', 'on', 'yes'},
            'radar_detection_threshold_db': payload.get('radar_detection_threshold_db', 8.0),
            'auto_occupancy_label_enabled': str(payload.get('auto_occupancy_label_enabled', '')).lower() in {'1', 'true', 'on', 'yes'},
        }
        saved = device_manager.save_device_settings(updates)
        if request.is_json:
            return jsonify({'success': True, 'settings': saved})
        flash('Settings saved', 'success')
        return redirect(url_for('settings'))

    return render_template(
        'settings.html',
        username=session.get('username'),
        device_settings=device_manager.get_device_settings(),
    )


@app.route('/models')
def models():
    """Device model management page."""
    if 'username' not in session:
        return redirect(url_for('login', next=url_for('models')))

    pending = device_manager.list_pending_deployments()
    running = device_manager.get_running_models()
    timelines = collect_prediction_timelines()

    return render_template(
        'models.html',
        username=session.get('username'),
        pending_deployments=pending,
        running_models=running,
        prediction_timelines=timelines,
        device_settings=device_manager.get_device_settings(),
    )


@app.route('/api/settings', methods=['GET', 'PATCH'])
def api_settings():
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    if request.method == 'GET':
        return jsonify({'success': True, 'settings': device_manager.get_device_settings()})
    payload = request.get_json(silent=True) or {}
    saved = device_manager.save_device_settings(payload)
    return jsonify({'success': True, 'settings': saved})


@app.route('/api/models/deployments/pending', methods=['GET'])
def api_pending_model_deployments():
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    return jsonify({'success': True, 'deployments': device_manager.list_pending_deployments()})


@app.route('/api/models/prediction-timelines', methods=['GET'])
def api_model_prediction_timelines():
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    timelines = collect_prediction_timelines()
    return jsonify({
        'success': True,
        'timelines': timelines,
        'model_count': len(timelines),
    })


@app.route('/api/models/deployments/<deployment_id>/accept', methods=['POST'])
def api_accept_model_deployment(deployment_id: str):
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    deployment = next((item for item in device_manager.list_pending_deployments() if item.get('deployment_id') == deployment_id), None)
    if not deployment:
        return jsonify({'success': False, 'message': 'Deployment not found'}), 404
    if not device_manager.acknowledge_deployment(deployment, accepted=True):
        return jsonify({'success': False, 'message': 'Failed to accept deployment'}), 500
    return jsonify({'success': True, 'message': 'Deployment accepted'})


@app.route('/api/models/deployments/<deployment_id>/decline', methods=['POST'])
def api_decline_model_deployment(deployment_id: str):
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    deployment = next((item for item in device_manager.list_pending_deployments() if item.get('deployment_id') == deployment_id), None)
    if not deployment:
        return jsonify({'success': False, 'message': 'Deployment not found'}), 404
    if not device_manager.acknowledge_deployment(deployment, accepted=False):
        return jsonify({'success': False, 'message': 'Failed to decline deployment'}), 500
    return jsonify({'success': True, 'message': 'Deployment declined'})

@app.route('/captures')
def captures():
    """Show the minute capture browser."""
    minutes = list_minutes()
    active = current_minute()
    system_status = get_system_status(update_remote=False)
    sensors = detect_sensor_inventory()
    capture_settings = device_manager.load_capture_settings()
    enabled_map = (capture_settings or {}).get("sensors", {})
    for sensor in sensors:
        sensor["enabled"] = bool(enabled_map.get(sensor.get("key"), True))
    return render_template(
        'captures.html',
        minutes=minutes,
        active_minute=active.name if active else None,
        username=session.get('username'),
        system_status=system_status,
        sensors=sensors,
        capture_settings=capture_settings,
    )


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
    settings = device_manager.save_capture_settings(payload)
    return jsonify({'success': True, 'capture_settings': settings, 'message': 'Capture settings updated'})


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
    metrics = minute_metrics(minute_dir)
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
        capture_metrics=metrics,
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
    requested = payload.get('minutes', [])
    if not isinstance(requested, list):
        return [], [], (jsonify({'status': 'error', 'message': 'minutes must be a list'}), 400)

    names = list(dict.fromkeys(str(item).strip() for item in requested if str(item).strip()))
    if not names:
        return [], [], (jsonify({'status': 'error', 'message': 'Select at least one capture'}), 400)
    if len(names) > 100:
        return [], [], (jsonify({'status': 'error', 'message': 'Select no more than 100 captures at once'}), 400)

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
    """Download selected capture folders in one ZIP archive."""
    names, minute_dirs, error = _requested_capture_minutes()
    if error:
        return error

    temp = tempfile.NamedTemporaryFile(prefix='thoth-captures-', suffix='.zip', delete=False)
    zip_path = Path(temp.name)
    temp.close()
    try:
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for minute_dir in minute_dirs:
                for path in sorted(minute_dir.rglob('*')):
                    if path.is_file():
                        archive.write(path, arcname=str(Path(minute_dir.name) / path.relative_to(minute_dir)))
    except Exception:
        zip_path.unlink(missing_ok=True)
        raise

    @after_this_request
    def cleanup_bulk_archive(response):
        zip_path.unlink(missing_ok=True)
        return response

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return send_file(
        zip_path,
        as_attachment=True,
        download_name=f'thoth-captures-{timestamp}.zip',
        mimetype='application/zip',
        conditional=True,
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
    metrics = minute_metrics(minute_dir)
    detail['files_on_disk'] = {key: str(path) if path else None for key, path in files.items()}
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
    if not video_path or not video_path.exists():
        abort(404, description='No video available for this minute')

    try:
        jpeg_bytes = _render_video_frame(video_path)
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
        'csi_csv': files.get('csi_csv'),
        'csi_timestamped': files.get('csi_timestamped'),
        'csi_serial': files.get('csi_serial'),
        'manifest': files.get('manifest'),
        'predictions': files.get('predictions'),
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
    minute_dir = get_minute(minute)
    if not minute_dir:
        abort(404, description='Minute folder not found')

    files = capture_files(minute_dir)
    path = files.get('csi_csv') or files.get('csi_timestamped') or files.get('csi_serial')
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

    files = capture_files(minute_dir)
    summary = minute_summary(minute_dir)
    device_id = getattr(device_manager, 'device_id', None)
    uploaded = []
    errors = []
    skipped = []
    import base64

    for key, path in files.items():
        if not path or not path.exists():
            continue
        if key in {'video_log'} and path.stat().st_size == 0:
            skipped.append(path.name)
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

    # Upload browser-ready products so ResearchPortal does not need the radar
    # hardware libraries or the original multi-megabyte capture to visualize it.
    generated = []
    radar_path = files.get('radar')
    if radar_path and radar_path.exists():
        for plot in RADAR_PLOTS:
            try:
                generated.append((f'{minute_dir.name}_radar-{plot}.json', _radar_plot_payload(radar_path, plot), f'radar-{plot}'))
            except Exception as exc:
                errors.append(f'radar-{plot}: {exc}')
    csi_path = files.get('csi_timestamped') or files.get('csi_csv') or files.get('csi_serial')
    if csi_path and csi_path.exists():
        try:
            generated.append((f'{minute_dir.name}_csi-plot.json', _csi_plot_payload(csi_path), 'csi-plot'))
        except Exception as exc:
            errors.append(f'csi-plot: {exc}')
    # The minute-named manifest completes the DeviceFile upload request in Brain.
    generated.append((minute_dir.name, summary, 'capture-manifest'))

    for filename, document, key in generated:
        try:
            raw = json.dumps(document, default=str).encode('utf-8')
            response = requests.post(
                f"{Config.BRAIN_SERVER_URL}/api/file/upload",
                json={
                    'filename': filename,
                    'content': base64.b64encode(raw).decode('ascii'),
                    'is_base64': True,
                    'device_id': device_id,
                    'content_type': 'application/json',
                    'metadata': {'source': 'thoth_device', 'minute': minute_dir.name, 'file_kind': key},
                },
                headers={'Authorization': f'Bearer {auth_token}', 'Content-Type': 'application/json'},
                timeout=120,
            )
            if response.status_code in (200, 201):
                uploaded.append({'name': filename, 'bytes': len(raw)})
            else:
                errors.append(f'{filename}: {response.status_code} {response.text[:300]}')
        except Exception as exc:
            errors.append(f'{filename}: {exc}')

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
    has_video = bool(files.get('video') and files['video'].exists()) if minute_dir else False
    has_csi = bool(files.get('csi_csv') or files.get('csi_timestamped') or files.get('csi_serial'))
    has_radar = bool(files.get('radar'))
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
        radar_data_urls=[url_for('api_live_capture_radar_data', plot=plot) for plot in RADAR_PLOTS],
        has_video=has_video,
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
    if not minute_dir or not video_path or not video_path.exists():
        abort(404, description='No live video available')

    try:
        jpeg_bytes = _render_video_frame(video_path)
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

    path = files.get('csi_csv') or files.get('csi_timestamped') or files.get('csi_serial')
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

    path = files.get('csi_csv') or files.get('csi_timestamped') or files.get('csi_serial')
    if not path or not path.exists():
        abort(404, description='No CSI data available')

    points = _parse_csi_average_series(path)
    svg = _build_csi_svg(points)
    return Response(svg, mimetype='image/svg+xml', headers={'Cache-Control': 'no-cache'})


@app.route('/api/captures/live/radar/data/<plot>')
def api_live_capture_radar_data(plot):
    """Return interactive live radar data."""
    plot = plot.lower()
    if plot not in RADAR_PLOTS:
        abort(404, description=f'Unsupported radar plot kind: {plot}')

    minute_dir, files = _best_live_minute_for_kind('radar')
    if not minute_dir:
        abort(404, description='No live capture minute available')

    radar_path = files.get('radar')
    if not radar_path or not radar_path.exists():
        abort(404, description='No radar data available')

    try:
        payload = _radar_plot_payload(radar_path, plot)
    except Exception as exc:
        logger.exception(f"Failed to build live radar data {plot}: {exc}")
        abort(500, description=str(exc))

    payload['metadata'] = minute_metrics(minute_dir)
    return jsonify({'status': 'success', 'minute': minute_dir.name, 'data': payload})


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
            device_manager.save_capture_settings({'labels': [label] if label else [], 'sensors': sensors})
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
