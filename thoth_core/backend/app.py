"""Thoth Flask Backend Application (platform-agnostic core).

This module provides the main Flask application for the Thoth device,
including REST API endpoints and WebSocket support for real-time data streaming.
RPi-specific code (SenseHat, PiSugar, captive portal, hotspot) has been removed;
each platform package extends this core as needed.
"""

import os
import sys
import json
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
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
import cv2
import base64

# Load environment variables
load_dotenv()

from flask import (
    Flask, jsonify, request, render_template, 
    redirect, url_for, flash, session, send_from_directory, abort
)
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

from .config import Config, BUTTON_ACTIONS, SENSOR_CONFIG
from .models import SensorReading, SystemStatus, ButtonConfig, UploadResult
from .device_manager import DeviceManager
from .auth_manager import AuthManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config.from_object(Config)

# Register blueprints
from .routes import files as files_bp
app.register_blueprint(files_bp.bp)

# Initialize file manager
from .file_manager import file_manager

# Add request logging
@app.before_request
def log_request():
    logger.info(f"Request: {request.method} {request.path} - {request.remote_addr}")

@app.after_request
def log_response(response):
    logger.info(f"Response: {request.method} {request.path} - {response.status_code}")
    return response

# Load user info from saved credentials before each request
@app.before_request
def populate_session_if_auth_manager_has_credentials():
    """If credentials were loaded at startup but session is empty, populate it."""
    # Disabled: require explicit login on first run
    # if 'username' not in session and auth_manager.is_authenticated() and auth_manager.user_info:
    #     session['username'] = auth_manager.user_info.get('username')
    #     session['user_id'] = auth_manager.user_info.get('user_id')
    #     session['token'] = auth_manager.token
    pass

# Add current date to all templates
@app.context_processor
def inject_now():
    """Inject commonly used globals into every template."""
    user_info = None
    if 'username' in session:
        user_info = {
            'username': session.get('username'),
            'user_id': session.get('user_id')
        }
    elif auth_manager.is_authenticated():
        user_info = auth_manager.user_info
    return {
        'now': datetime.utcnow(),
        'user_info': user_info,
        'config': {
            'app_name': Config.APP_NAME,
            'version': Config.VERSION
        }
    }

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

# Initialize managers
auth_manager = AuthManager(Config)
device_manager = DeviceManager(Config)

# Global state
collection_active = False


def get_system_uptime() -> str:
    """Get system uptime in a human-readable format."""
    try:
        uptime_seconds = time.time() - psutil.boot_time()
        return str(timedelta(seconds=int(uptime_seconds)))
    except Exception:
        return "unknown"


def get_system_status(update_remote: bool = True) -> SystemStatus:
    """Get current system status and optionally update the Brain server."""
    try:
        # Check internet connectivity
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(2)
            s.connect(("8.8.8.8", 80))
            s.close()
            wifi_connected = True
        except Exception:
            wifi_connected = False
        
        # Get battery level
        battery_level = None
        try:
            battery = psutil.sensors_battery()
            if battery:
                battery_level = int(battery.percent)
        except Exception:
            battery_level = None
        
        # Get CPU temperature (Linux only)
        cpu_temp = None
        if platform.system() == "Linux":
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    cpu_temp = float(f.read().strip()) / 1000.0
            except Exception:
                pass
        
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
        
        uptime_output = get_system_uptime()
        
        status = SystemStatus(
            status="ok",
            battery_level=battery_level,
            wifi_connected=wifi_connected,
            ap_mode=False,
            collection_active=collection_active,
            uptime=uptime_output,
            temperature=cpu_temp,
            disk_usage=disk_usage,
            ip_address=ip_address
        )
        
        if update_remote:
            try:
                device_manager.update_status({
                    'battery_level': battery_level,
                    'wifi_connected': wifi_connected,
                    'collection_active': collection_active,
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
        interfaces = {}
        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            if netifaces.AF_LINK in addrs and addrs[netifaces.AF_LINK]:
                mac = addrs[netifaces.AF_LINK][0].get('addr')
                if mac and mac != '00:00:00:00:00:00':
                    interfaces[iface] = mac
        
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


def _process_pending_uploads(filenames: list, auth_token: str):
    """Process pending upload requests by uploading files to Brain server."""
    def upload_files():
        for filename in filenames:
            try:
                file_path = os.path.join(Config.DATA_DIR, filename)
                if not os.path.exists(file_path):
                    logger.warning(f"File not found for upload: {filename}")
                    continue
                with open(file_path, 'rb') as f:
                    content = f.read()
                content_b64 = base64.b64encode(content).decode('utf-8')
                ext = os.path.splitext(filename)[1].lower()
                content_type = 'application/json' if ext == '.json' else 'text/csv' if ext == '.csv' else 'application/octet-stream'
                device_id = getattr(Config, 'DEVICE_ID', None)
                brain_url = f"{Config.BRAIN_SERVER_URL}/file/upload"
                headers = {
                    'Authorization': f'Bearer {auth_token}',
                    'Content-Type': 'application/json'
                }
                payload = {
                    'filename': filename,
                    'content': content_b64,
                    'is_base64': True,
                    'device_id': device_id,
                    'content_type': content_type
                }
                logger.info(f"Uploading {filename} ({len(content)} bytes) to Brain cloud")
                response = requests.post(brain_url, json=payload, headers=headers, timeout=120)
                if response.status_code in (200, 201):
                    result = response.json()
                    logger.info(f"File uploaded successfully: {filename} -> cloud_file_id={result.get('file_id')}")
                else:
                    logger.error(f"Upload failed for {filename}: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error uploading {filename}: {e}")
    thread = threading.Thread(target=upload_files, daemon=True)
    thread.start()


def register_device_periodically():
    """Register device with Brain server every minute (only if user is authenticated)."""
    try:
        if not getattr(Config, 'BRAIN_SERVER_URL', None):
            return
        auth_token = getattr(Config, 'USER_AUTH_TOKEN', None)
        if not auth_token:
            return
        device_info = get_device_info()
        headers = {
            'Authorization': f'Bearer {auth_token.strip()}',
            'Content-Type': 'application/json',
            'User-Agent': 'Thoth-Device/1.0'
        }
        device_id = getattr(Config, 'DEVICE_ID', None)
        if not device_id:
            mac_address = next(iter(device_info.get('network_interfaces', {}).values()), None)
            if mac_address:
                device_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, mac_address))
            else:
                device_id = str(uuid.uuid4())
            Config.DEVICE_ID = device_id
        mac_display = next(iter(device_info.get('network_interfaces', {}).values()), '')[:8]
        files_list = device_manager._get_data_files_list()
        registration_data = {
            'device_id': device_id,
            'device_name': f"Thoth-{mac_display}" if mac_display else f"Thoth-{device_id[:8]}",
            'device_type': 'thoth',
            'hardware_info': device_info,
            'files': files_list
        }
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{Config.BRAIN_SERVER_URL}/device/register",
                    json=registration_data,
                    headers=headers,
                    timeout=30
                )
                if response.status_code in (200, 201):
                    try:
                        result = response.json()
                        pending_uploads = result.get('pending_uploads', [])
                        if pending_uploads:
                            _process_pending_uploads(pending_uploads, auth_token)
                    except ValueError:
                        pass
                    return True
                if attempt == 2:
                    return False
                time.sleep(2)
            except requests.exceptions.RequestException:
                if attempt == 2:
                    return False
                time.sleep(2)
        return False
    except Exception as e:
        logger.error(f"Unexpected error in device registration: {str(e)}", exc_info=True)
        return False


# Start background tasks
socketio.start_background_task(tail_sensor_data)

device_scheduler.add_job(
    register_device_periodically,
    'interval',
    seconds=10,
    id='device_registration',
    replace_existing=True
)
device_scheduler.start()

device_manager.load_registration_info()
if device_manager.registered:
    try:
        device_manager.start_heartbeat(Config.HEARTBEAT_INTERVAL)
    except Exception as e:
        logger.error(f"Failed to start device heartbeat: {e}")


# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def index():
    """Serve the appropriate page based on authentication status."""
    if 'username' in session:
        return redirect(url_for('status'))
    return redirect(url_for('login'))


@app.route('/health')
def health():
    """Get system health status."""
    status = get_system_status()
    return jsonify(status.to_dict())


@app.route('/data/current')
def get_current_data():
    """Get the latest sensor reading."""
    try:
        if os.path.exists(Config.SENSOR_DATA_FILE):
            with open(Config.SENSOR_DATA_FILE, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    data = json.loads(last_line)
                    return jsonify(data)
        return jsonify({"error": "No data available"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/data/history')
def get_data_history():
    """Get historical sensor data with pagination."""
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        if not os.path.exists(Config.SENSOR_DATA_FILE):
            return jsonify([])
        with open(Config.SENSOR_DATA_FILE, 'r') as f:
            lines = f.readlines()
        start_idx = max(0, len(lines) - offset - limit)
        end_idx = len(lines) - offset
        data = []
        for line in lines[start_idx:end_idx]:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_data():
    """Upload collected data to remote server."""
    try:
        upload_url = request.json.get('upload_url') if request.json else Config.UPLOAD_URL
        if not upload_url:
            return jsonify({"success": False, "error": "No upload URL configured"}), 400
        if not os.path.exists(Config.SENSOR_DATA_FILE):
            return jsonify({"success": False, "error": "No data file found"}), 404
        data = []
        with open(Config.SENSOR_DATA_FILE, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        if not data:
            return jsonify({"success": False, "error": "No data to upload"}), 400
        headers = {'Content-Type': 'application/json'}
        if Config.API_KEY:
            headers['Authorization'] = f'Bearer {Config.API_KEY}'
        response = requests.post(upload_url, json=data, headers=headers, timeout=30)
        if response.status_code == 200:
            return jsonify({"success": True, "uploaded": len(data), "message": "Data uploaded successfully"})
        else:
            return jsonify({"success": False, "error": f"Upload failed: {response.status_code}"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/config/button', methods=['GET', 'POST'])
def button_config():
    """Get or set button configuration."""
    if request.method == 'GET':
        return jsonify(BUTTON_ACTIONS)
    try:
        new_config = request.json
        BUTTON_ACTIONS.update(new_config)
        config_file = os.path.join(os.path.dirname(__file__), 'button_config.json')
        with open(config_file, 'w') as f:
            json.dump(BUTTON_ACTIONS, f)
        return jsonify({"updated": BUTTON_ACTIONS, "message": "Button configuration updated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login and device registration."""
    if 'username' in session:
        return redirect(url_for('status'))
    if request.method == 'GET':
        return render_template('login.html', next=request.args.get('next', ''))
    try:
        username = request.form.get('username')
        password = request.form.get('password')
        next_page = request.form.get('next', '')
        if not username or not password:
            flash('Username and password are required', 'error')
            return redirect(url_for('login', next=next_page))
        try:
            result = auth_manager.login(username, password)
            if result.get('success'):
                session['user_id'] = result['user'].get('user_id')
                session['username'] = username
                session['token'] = result['token']
                Config.USER_AUTH_TOKEN = result['token']
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


@app.route('/api/setup/login', methods=['POST'])
def api_setup_login():
    """Handle login from setup page."""
    try:
        data = request.get_json() or request.form
        username = data.get('username')
        password = data.get('user_password')
        if not username or not password:
            return jsonify({'status': 'error', 'error': 'Username and password are required'}), 400
        result = auth_manager.login(username, password)
        if result.get('success'):
            session['user_id'] = result['user'].get('user_id')
            session['username'] = username
            session['token'] = result['token']
            Config.USER_AUTH_TOKEN = result['token']
            success, message = device_manager.register_device(result['token'])
            if success:
                device_manager.start_heartbeat(Config.HEARTBEAT_INTERVAL)
            return jsonify({
                'status': 'success',
                'message': 'Login successful',
                'redirect': url_for('info_page')
            })
        else:
            return jsonify({'status': 'error', 'error': 'Invalid username or password'}), 401
    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/status')
def status():
    """Display the device status page."""
    return redirect(url_for('info_page'))


# ============================================================================
# Thoth Web App Pages: Info, Media, Models
# ============================================================================

def get_available_sensors():
    """Get list of available sensors on this device."""
    sensors = []
    # Check for camera
    try:
        cap = cv2.VideoCapture(0)
        camera_available = cap.isOpened()
        cap.release()
        sensors.append({
            'sensor_type': 'camera',
            'name': 'Camera',
            'available': camera_available,
            'error': None if camera_available else 'Camera not detected'
        })
    except Exception as e:
        sensors.append({
            'sensor_type': 'camera',
            'name': 'Camera',
            'available': False,
            'error': str(e)
        })
    # Check for microphone
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        mic_available = p.get_device_count() > 0
        p.terminate()
    except Exception:
        mic_available = False
    sensors.append({
        'sensor_type': 'microphone',
        'name': 'Microphone',
        'available': mic_available,
        'error': None if mic_available else 'Microphone not detected'
    })
    return sensors


def get_media_files():
    """Get list of media files in the data directory."""
    media_files = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    video_extensions = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}
    audio_extensions = {'.wav', '.mp3', '.m4a', '.aac', '.ogg', '.flac'}
    media_extensions = image_extensions | video_extensions | audio_extensions
    labels_dir = os.path.join(Config.DATA_DIR, 'labels')
    cloud_status = {}
    cloud_status_file = os.path.join(Config.DATA_DIR, 'cloud_status.json')
    if os.path.exists(cloud_status_file):
        try:
            with open(cloud_status_file, 'r') as f:
                cloud_status = json.load(f)
        except:
            pass
    try:
        data_dir = Config.DATA_DIR
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext in media_extensions:
                    file_path = os.path.join(data_dir, filename)
                    stat = os.stat(file_path)
                    if ext in image_extensions:
                        file_type = 'image'
                    elif ext in video_extensions:
                        file_type = 'video'
                    elif ext in audio_extensions:
                        file_type = 'audio'
                    else:
                        file_type = 'other'
                    size = stat.st_size
                    if size < 1024:
                        size_formatted = f"{size} B"
                    elif size < 1024 * 1024:
                        size_formatted = f"{size / 1024:.1f} KB"
                    else:
                        size_formatted = f"{size / (1024 * 1024):.1f} MB"
                    labels = []
                    base_name = os.path.splitext(filename)[0]
                    labels_file = os.path.join(labels_dir, f"{base_name}.json")
                    if os.path.exists(labels_file):
                        try:
                            with open(labels_file, 'r') as f:
                                label_data = json.load(f)
                                if isinstance(label_data, dict):
                                    labels = label_data.get('labels', [])
                                elif isinstance(label_data, list):
                                    labels = label_data
                        except:
                            pass
                    media_files.append({
                        'filename': filename,
                        'type': file_type,
                        'size': size,
                        'size_formatted': size_formatted,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'on_cloud': filename in cloud_status,
                        'labels': labels
                    })
        media_files.sort(key=lambda x: x['modified'], reverse=True)
    except Exception as e:
        logger.error(f"Error getting media files: {e}")
    return media_files


@app.route('/info')
def info_page():
    """Display the device info page."""
    if 'username' not in session:
        return redirect(url_for('login', next=url_for('info_page')))
    try:
        system_status = get_system_status()
        device_info = device_manager.get_device_info()
        hardware_info = get_device_info()
        sensors = get_available_sensors()
        return render_template('info.html',
                            active_page='info',
                            system_status=system_status,
                            device_info=device_info,
                            hardware_info=hardware_info,
                            sensors=sensors)
    except Exception as e:
        logger.error(f"Error in info page: {str(e)}", exc_info=True)
        flash('An error occurred while loading the info page.', 'error')
        return redirect(url_for('login'))


@app.route('/media')
def media_page():
    """Display the media gallery page."""
    if 'username' not in session:
        return redirect(url_for('login', next=url_for('media_page')))
    try:
        media_files = get_media_files()
        return render_template('media.html',
                            active_page='media',
                            media_files=media_files)
    except Exception as e:
        logger.error(f"Error in media page: {str(e)}", exc_info=True)
        flash('An error occurred while loading the media page.', 'error')
        return redirect(url_for('info_page'))


@app.route('/models')
def models_page():
    """Display the deployed models page."""
    if 'username' not in session:
        return redirect(url_for('login', next=url_for('models_page')))
    try:
        models = []
        total_inferences = 0
        return render_template('models.html',
                            active_page='models',
                            models=models,
                            total_inferences=total_inferences)
    except Exception as e:
        logger.error(f"Error in models page: {str(e)}", exc_info=True)
        flash('An error occurred while loading the models page.', 'error')
        return redirect(url_for('info_page'))


# ============================================================================
# Media API Endpoints
# ============================================================================

@app.route('/api/media/list', methods=['GET'])
def api_media_list():
    """Get list of media files."""
    try:
        media_files = get_media_files()
        return jsonify({'status': 'success', 'files': media_files})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/media/serve/<filename>')
def api_media_serve(filename):
    """Serve a media file."""
    try:
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'status': 'error', 'error': 'Invalid filename'}), 400
        return send_from_directory(Config.DATA_DIR, filename)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 404


@app.route('/api/media/delete/<filename>', methods=['DELETE'])
def api_media_delete(filename):
    """Delete a media file locally and from cloud."""
    if 'username' not in session:
        return jsonify({'status': 'error', 'error': 'Unauthorized'}), 401
    try:
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'status': 'error', 'error': 'Invalid filename'}), 400
        file_path = os.path.join(Config.DATA_DIR, filename)
        cloud_deleted = False
        cloud_status_file = os.path.join(Config.DATA_DIR, 'cloud_status.json')
        cloud_status = {}
        if os.path.exists(cloud_status_file):
            try:
                with open(cloud_status_file, 'r') as f:
                    cloud_status = json.load(f)
            except:
                pass
        if filename in cloud_status:
            file_id = cloud_status[filename].get('file_id')
            if file_id:
                auth_token = session.get('token') or getattr(Config, 'USER_AUTH_TOKEN', None)
                if auth_token:
                    try:
                        headers = {'Authorization': f'Bearer {auth_token}'}
                        response = requests.delete(
                            f"{Config.BRAIN_SERVER_URL}/file/{file_id}",
                            headers=headers, timeout=30
                        )
                        if response.status_code in (200, 204):
                            cloud_deleted = True
                    except Exception as e:
                        logger.warning(f"Error deleting from cloud: {e}")
            del cloud_status[filename]
            with open(cloud_status_file, 'w') as f:
                json.dump(cloud_status, f, indent=2)
        if os.path.exists(file_path):
            os.remove(file_path)
            labels_file = os.path.join(Config.DATA_DIR, 'labels', os.path.splitext(filename)[0] + '.json')
            if os.path.exists(labels_file):
                os.remove(labels_file)
            msg = f'Deleted {filename}'
            if cloud_deleted:
                msg += ' (also removed from cloud)'
            return jsonify({'status': 'success', 'message': msg})
        else:
            return jsonify({'status': 'error', 'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/media/upload', methods=['POST'])
def api_media_upload():
    """Upload a media file."""
    if 'username' not in session:
        return jsonify({'status': 'error', 'error': 'Unauthorized'}), 401
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'error': 'No file selected'}), 400
        filename = file.filename
        file_path = os.path.join(Config.DATA_DIR, filename)
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        file.save(file_path)
        return jsonify({'status': 'success', 'filename': filename})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/media/upload/<filename>', methods=['POST'])
def api_media_upload_to_cloud(filename):
    """Upload a specific media file to the cloud."""
    if 'username' not in session:
        return jsonify({'status': 'error', 'error': 'Unauthorized'}), 401
    try:
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'status': 'error', 'error': 'Invalid filename'}), 400
        file_path = os.path.join(Config.DATA_DIR, filename)
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'error': 'File not found'}), 404
        ext = os.path.splitext(filename)[1].lower()
        content_types = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
            '.gif': 'image/gif', '.mp4': 'video/mp4', '.avi': 'video/avi',
            '.mov': 'video/quicktime', '.webm': 'video/webm',
            '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.m4a': 'audio/mp4'
        }
        content_type = content_types.get(ext, 'application/octet-stream')
        auth_token = session.get('token') or getattr(Config, 'USER_AUTH_TOKEN', None)
        if not auth_token:
            return jsonify({'status': 'error', 'error': 'Not authenticated.'}), 401
        device_id = device_manager.device_id if device_manager else None
        labels = []
        labels_dir = os.path.join(Config.DATA_DIR, 'labels')
        base_name = os.path.splitext(filename)[0]
        labels_file = os.path.join(labels_dir, f"{base_name}.json")
        if os.path.exists(labels_file):
            try:
                with open(labels_file, 'r') as f:
                    label_data = json.load(f)
                    if isinstance(label_data, dict):
                        labels = label_data.get('labels', [])
                    elif isinstance(label_data, list):
                        labels = label_data
            except:
                pass
        headers = {'Authorization': f'Bearer {auth_token}'}
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f, content_type)}
            data = {}
            if device_id:
                data['device_id'] = device_id
            if labels:
                data['labels'] = json.dumps(labels)
            response = requests.post(
                f"{Config.BRAIN_SERVER_URL}/file/upload-multipart",
                files=files, data=data, headers=headers, timeout=120
            )
        if response.status_code in (200, 201):
            result = response.json()
            cloud_status_file = os.path.join(Config.DATA_DIR, 'cloud_status.json')
            cloud_status = {}
            if os.path.exists(cloud_status_file):
                try:
                    with open(cloud_status_file, 'r') as f:
                        cloud_status = json.load(f)
                except:
                    pass
            cloud_status[filename] = {
                'file_id': result.get('file_id'),
                'uploaded_at': datetime.utcnow().isoformat()
            }
            with open(cloud_status_file, 'w') as f:
                json.dump(cloud_status, f, indent=2)
            return jsonify({'status': 'success', 'message': f'Uploaded {filename}', 'file_id': result.get('file_id')})
        elif response.status_code == 401:
            return jsonify({'status': 'error', 'error': 'Authentication expired.'}), 401
        else:
            return jsonify({'status': 'error', 'error': f'Upload failed: {response.status_code}'}), 500
    except requests.exceptions.Timeout:
        return jsonify({'status': 'error', 'error': 'Upload timed out.'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/media/rename/<filename>', methods=['POST'])
def api_media_rename(filename):
    """Rename a media file."""
    if 'username' not in session:
        return jsonify({'status': 'error', 'error': 'Unauthorized'}), 401
    try:
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'status': 'error', 'error': 'Invalid filename'}), 400
        data = request.get_json()
        new_name = data.get('new_name', '').strip()
        if not new_name:
            return jsonify({'status': 'error', 'error': 'New name is required'}), 400
        if '..' in new_name or '/' in new_name or '\\' in new_name:
            return jsonify({'status': 'error', 'error': 'Invalid new filename'}), 400
        old_path = os.path.join(Config.DATA_DIR, filename)
        new_path = os.path.join(Config.DATA_DIR, new_name)
        if not os.path.exists(old_path):
            return jsonify({'status': 'error', 'error': 'File not found'}), 404
        if os.path.exists(new_path):
            return jsonify({'status': 'error', 'error': 'A file with that name already exists'}), 400
        os.rename(old_path, new_path)
        old_labels = get_labels_file_path(filename)
        if os.path.exists(old_labels):
            new_labels = get_labels_file_path(new_name)
            os.rename(old_labels, new_labels)
        return jsonify({'status': 'success', 'message': f'Renamed to {new_name}', 'new_name': new_name})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ============================================================================
# Labels API Endpoints
# ============================================================================

def get_labels_file_path(media_filename):
    """Get the path to the labels JSON file for a media file."""
    labels_dir = os.path.join(Config.DATA_DIR, 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    base_name = os.path.splitext(media_filename)[0]
    return os.path.join(labels_dir, f"{base_name}.json")


@app.route('/api/media/labels/<filename>', methods=['GET'])
def api_get_labels(filename):
    """Get labels for a media file."""
    try:
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'status': 'error', 'error': 'Invalid filename'}), 400
        labels_path = get_labels_file_path(filename)
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                data = json.load(f)
            return jsonify({'status': 'success', 'labels': data.get('labels', [])})
        else:
            return jsonify({'status': 'success', 'labels': []})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/media/labels/<filename>', methods=['POST'])
def api_save_labels(filename):
    """Save labels for a media file."""
    if 'username' not in session:
        return jsonify({'status': 'error', 'error': 'Unauthorized'}), 401
    try:
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'status': 'error', 'error': 'Invalid filename'}), 400
        data = request.get_json()
        labels = data.get('labels', [])
        labels_path = get_labels_file_path(filename)
        with open(labels_path, 'w') as f:
            json.dump({
                'filename': filename,
                'labels': labels,
                'updated_at': datetime.utcnow().isoformat(),
                'updated_by': session.get('username')
            }, f, indent=2)
        return jsonify({'status': 'success', 'message': 'Labels saved'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/logout')
def do_logout():
    """Log out the current user."""
    try:
        device_manager.update_status({
            'online': False,
            'last_seen': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Error updating device status on logout: {e}")
    device_manager.stop_heartbeat()
    Config.USER_AUTH_TOKEN = None
    try:
        auth_manager.logout()
    except Exception as e:
        logger.error(f"Error in auth_manager logout: {e}")
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


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
        return jsonify({'status': 'success', 'message': f'Service {service} restart initiated'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/system/shutdown', methods=['POST'])
def system_shutdown():
    """Shut down the device."""
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
    try:
        return jsonify({'status': 'success', 'message': 'Shutdown initiated.'})
    except Exception as e:
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
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ============================================================================
# Camera capture API
# ============================================================================

@app.route('/api/camera/list', methods=['GET'])
def camera_list():
    """List available cameras on the system."""
    cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            backend = cap.getBackendName()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cameras.append({
                'index': i,
                'name': f'Camera {i}',
                'backend': backend,
                'resolution': f'{width}x{height}'
            })
            cap.release()
    return jsonify({'status': 'success', 'cameras': cameras})


@app.route('/api/camera/capture', methods=['GET'])
def camera_capture():
    """Capture an image from the selected camera and return as base64 data URI."""
    try:
        camera_index = int(request.args.get('camera_index', 0))
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return jsonify({'status': 'error', 'message': f'Camera {camera_index} not available.'}), 400
        for _ in range(10):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return jsonify({'status': 'error', 'message': 'Failed to capture.'}), 500
        width = int(frame.shape[1])
        height = int(frame.shape[0])
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return jsonify({'status': 'error', 'message': 'Failed to encode image'}), 500
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"img_{timestamp}.jpg"
        file_path = os.path.join(Config.DATA_DIR, filename)
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        cv2.imwrite(file_path, frame)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        data_uri = f'data:image/jpeg;base64,{img_b64}'
        return jsonify({
            'status': 'success',
            'image': data_uri,
            'file': filename,
            'camera_index': camera_index,
            'resolution': f'{width}x{height}'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/camera/record', methods=['POST'])
def camera_record():
    """Record a video from the selected camera."""
    try:
        data = request.get_json(force=True) or {}
        duration = int(data.get('duration', 5))
        camera_index = int(data.get('camera_index', 0))
        if duration <= 0 or duration > 30:
            return jsonify({'status': 'error', 'message': 'Duration must be 1-30 seconds'}), 400
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"vid_{timestamp}.mp4"
        file_path = os.path.join(Config.DATA_DIR, filename)
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return jsonify({'status': 'error', 'message': f'Camera {camera_index} not available.'}), 400
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0:
            fps = 30.0
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
        frame_count = 0
        start_time = time.time()
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1
        cap.release()
        out.release()
        return jsonify({'status': 'success', 'file': filename, 'frames': frame_count})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/audio/record', methods=['POST'])
def audio_record():
    """Record audio from the microphone."""
    try:
        data = request.get_json(force=True) or {}
        duration = int(data.get('duration', 5))
        if duration <= 0 or duration > 60:
            return jsonify({'status': 'error', 'message': 'Duration must be 1-60 seconds'}), 400
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"audio_{timestamp}.wav"
        file_path = os.path.join(Config.DATA_DIR, filename)
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        try:
            cmd = ['rec', '-q', file_path, 'trim', '0', str(duration)]
            result = subprocess.run(cmd, capture_output=True, timeout=duration + 5)
            if result.returncode != 0:
                raise Exception("sox failed")
        except Exception:
            try:
                if platform.system() == 'Darwin':
                    cmd = [
                        'ffmpeg', '-y', '-f', 'avfoundation', '-i', ':0',
                        '-t', str(duration), '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1',
                        file_path
                    ]
                else:
                    cmd = [
                        'ffmpeg', '-y', '-f', 'pulse', '-i', 'default',
                        '-t', str(duration), '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1',
                        file_path
                    ]
                result = subprocess.run(cmd, capture_output=True, timeout=duration + 10)
                if result.returncode != 0:
                    raise Exception(f"ffmpeg failed: {result.stderr.decode()}")
            except FileNotFoundError:
                return jsonify({
                    'status': 'error',
                    'message': 'Audio recording requires sox or ffmpeg.'
                }), 500
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return jsonify({'status': 'success', 'file': filename})
        else:
            return jsonify({'status': 'error', 'message': 'Audio recording failed'}), 500
    except subprocess.TimeoutExpired:
        return jsonify({'status': 'error', 'message': 'Recording timed out'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ============================================================================

def run_server(host=None, port=None, debug=False):
    """Start the Flask-SocketIO server (called by platform entry points)."""
    if not device_scheduler.running:
        device_scheduler.start()
    socketio.run(
        app,
        host=host or Config.HOST,
        port=port or Config.PORT,
        debug=debug,
        use_reloader=False,
        allow_unsafe_werkzeug=True
    )


if __name__ == '__main__':
    run_server(host='0.0.0.0', debug=True)
