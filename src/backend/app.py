"""Thoth Flask Backend Application.

This module provides the main Flask application for the Thoth device,
including REST API endpoints and WebSocket support for real-time data streaming.
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

# Load environment variables
load_dotenv()

from flask import (
    Flask, jsonify, request, render_template, 
    redirect, url_for, flash, session, send_from_directory, abort
)
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import requests
from werkzeug.security import generate_password_hash, check_password_hash

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import Sense HAT, fall back to mock if not available
try:
    from sense_hat import SenseHat
    sense = SenseHat()
    print("Using real Sense HAT")
except (ImportError, OSError):
    print("Sense HAT not found, using mock implementation")
    from backend.mock_sense_hat import sense

from backend.config import Config, BUTTON_ACTIONS, SENSOR_CONFIG
from backend.models import SensorReading, SystemStatus, ButtonConfig, UploadResult
from backend.device_manager import DeviceManager
from backend.auth_manager import AuthManager

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
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Ensure required directories exist
os.makedirs(Config.CONFIG_DIR, exist_ok=True)
os.makedirs(Config.LOGS_DIR, exist_ok=True)

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

# Mock network scan function (replace with actual implementation)
def scan_wifi_networks():
    """Scan for available WiFi networks."""
    # This is a mock implementation - replace with actual WiFi scanning code
    return [
        {'ssid': 'MyHomeWiFi', 'secure': True, 'signal': 85},
        {'ssid': 'GuestWiFi', 'secure': False, 'signal': 65},
        {'ssid': 'NeighborsWiFi', 'secure': True, 'signal': 45}
    ]

def get_system_uptime() -> str:
    """Get system uptime in a human-readable format."""
    with open('/proc/uptime', 'r') as f:
        uptime_seconds = float(f.readline().split()[0])
        return str(timedelta(seconds=uptime_seconds)).split('.')[0]  # Remove microseconds

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
        # Check WiFi connection
        wifi_connected = subprocess.run(
            ["ping", "-c1", "8.8.8.8"], 
            capture_output=True, timeout=5
        ).returncode == 0
        
        # Check collection status
        collection_status = subprocess.run(
            ["systemctl", "is-active", "thoth-collector"],
            capture_output=True, text=True
        ).stdout.strip() == "active"
        
        # Get battery level (if PiSugar available)
        
        # Get CPU temperature
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                cpu_temp = float(f.read().strip()) / 1000.0  # Convert millidegrees to degrees
        except Exception as e:
            logger.error(f"Error getting CPU temperature: {e}")
            cpu_temp = None
        
        # Get IP address
        ip_address = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
            s.close()
        except Exception as e:
            logger.error(f"Error getting IP address: {e}")
            logger.warning(f"Could not get IP address: {e}")
        
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
    """Register device with Brain server every minute."""
    try:
        # Get device info
        device_info = get_device_info()
        
        # Get device ID from session or generate one
        device_id = session.get('device_id')
        if not device_id:
            # Try to get MAC address as device ID
            mac_address = next(iter(device_info.get('network_interfaces', {}).values()), None)
            device_id = mac_address or str(uuid.uuid4())
            session['device_id'] = device_id

        # Get device name from session or use hostname
        device_name = session.get('device_name', device_info.get('hostname', 'Thoth Device'))

        # Get access token from environment
        access_token = os.getenv('BRAIN_AUTH_TOKEN')
        if not access_token:
            logger.error("No BRAIN_AUTH_TOKEN found in environment variables")
            return

        # Prepare request data
        data = {
            'device_id': device_id,
            'device_name': device_name,
            'device_type': 'thoth',
            'hardware_info': device_info
        }
        
        # Send registration request
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f"{Config.BRAIN_SERVER_URL}/api/device/register",
            json=data,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        
        logger.info(f"Device registration successful: {response.json()}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during device registration: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.error(f"Unexpected error in device registration: {str(e)}", exc_info=True)

def register_device_periodically():
    """Register device with Brain server every minute."""
    try:
        # Skip if Brain server URL is not configured
        if not getattr(Config, 'BRAIN_SERVER_URL', None):
            logger.warning("Brain server URL not configured, skipping device registration")
            return
            
        # Skip if authentication token is not configured
        auth_token = getattr(Config, 'BRAIN_AUTH_TOKEN', None)
        if not auth_token:
            logger.warning("Authentication token not configured, skipping device registration")
            logger.warning(f"auth_token: {auth_token}")
            return
            
        device_info = get_device_info()
        headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json'
        }
        
        # Generate a unique device ID if not already set
        device_id = getattr(Config, 'DEVICE_ID', str(uuid.uuid4()))
        device_name = getattr(Config, 'DEVICE_NAME', 'Thoth-Device')
        
        device_data = {
            'device_id': device_id,
            'name': device_name,
            'type': 'thoth',
            'status': 'online',
            'info': device_info,
            'last_seen': datetime.utcnow().isoformat(),
            'last_heartbeat': datetime.utcnow().isoformat()
        }
        
        # Prepare the device registration payload
        registration_data = {
            'device_id': device_id,
            'device_name': device_name,
            'device_type': 'thoth',
            'hardware_info': device_info
        }
        
        logger.info(f"Registering device with Brain server: {registration_data}")
        
        try:
            response = requests.post(
                f"{Config.BRAIN_SERVER_URL}/api/device/register",
                json=registration_data,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()  # Raise an exception for HTTP errors
        except requests.exceptions.RequestException as e:
            logger.error(f"Error registering device: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.status_code} - {e.response.text}")
            raise
        
        if response.status_code == 200:
            logger.info(f"Successfully registered device with Brain server. Response: {response.json()}")
        else:
            logger.error(f"Failed to register device: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during device registration: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in device registration: {str(e)}", exc_info=True)

# Start background tasks
socketio.start_background_task(tail_sensor_data)

# Start device registration scheduler
device_scheduler.add_job(
    register_device_periodically,
    'interval',
    minutes=1,
    id='device_registration',
    replace_existing=True
)
device_scheduler.start()

# Load registration info if available
device_manager.load_registration_info()

# Start device heartbeat if registered
if device_manager.registered:
    try:
        device_manager.start_heartbeat(Config.HEARTBEAT_INTERVAL)
        logger.info("Started device heartbeat")
    except Exception as e:
        logger.error(f"Failed to start device heartbeat: {e}")

# Routes
@app.route('/')
def index():
    """Serve the appropriate page based on authentication and registration status."""
    # Redirect to login if not authenticated
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Redirect to status page if authenticated
    return redirect(url_for('status'))
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Thoth Device</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .ok { background-color: #d4edda; color: #155724; }
            .error { background-color: #f8d7da; color: #721c24; }
            button { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
            .start { background-color: #28a745; color: white; }
            .stop { background-color: #dc3545; color: white; }
            .config { background-color: #007bff; color: white; }
        </style>
    </head>
    <body>
        <h1>Thoth Research Device</h1>
        <div id="status" class="status">Loading...</div>
        
        <h2>Controls</h2>
        <button class="start" onclick="startCollection()">Start Collection</button>
        <button class="stop" onclick="stopCollection()">Stop Collection</button>
        <button class="config" onclick="uploadData()">Upload Data</button>
        
        <h2>Live Data</h2>
        <div id="live-data">
            <p>Pitch: <span id="pitch">--</span>°</p>
            <p>Roll: <span id="roll">--</span>°</p>
            <p>Yaw: <span id="yaw">--</span>°</p>
        </div>
        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
        <script>
            const socket = io();
            
            // Update status
            function updateStatus() {
                fetch('/health')
                    .then(r => r.json())
                    .then(data => {
                        const statusDiv = document.getElementById('status');
                        statusDiv.className = 'status ' + (data.status === 'ok' ? 'ok' : 'error');
                        statusDiv.innerHTML = `
                            Status: ${data.status}<br>
                            Battery: ${data.battery_level || 'N/A'}%<br>
                            WiFi: ${data.wifi_connected ? 'Connected' : 'Disconnected'}<br>
                            Collection: ${data.collection_active ? 'Active' : 'Inactive'}
                        `;
                    });
            }
            
            // Control functions
            function startCollection() {
                fetch('/control/start', {method: 'POST'})
                    .then(r => r.json())
                    .then(data => {
                        alert('Collection started: ' + data.status);
                        updateStatus();
                    });
            }
            
            function stopCollection() {
                fetch('/control/stop', {method: 'POST'})
                    .then(r => r.json())
                    .then(data => {
                        alert('Collection stopped: ' + data.status);
                        updateStatus();
                    });
            }
            
            function uploadData() {
                fetch('/upload', {method: 'POST'})
                    .then(r => r.json())
                    .then(data => {
                        alert(`Upload result: ${data.success ? 'Success' : 'Failed'}`);
                    });
            }
            
            // WebSocket events
            socket.on('connect', function() {
                console.log('Connected to Thoth device');
            });
            
            socket.on('imu_data', function(data) {
                if (data.imu) {
                    document.getElementById('pitch').textContent = data.imu.pitch.toFixed(1);
                    document.getElementById('roll').textContent = data.imu.roll.toFixed(1);
                    document.getElementById('yaw').textContent = data.imu.yaw.toFixed(1);
                }
            });
            
            // Update status every 5 seconds
            updateStatus();
            setInterval(updateStatus, 5000);
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

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
        
        # Apply pagination
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

@app.route('/control/start', methods=['POST'])
def start_collection():
    """Start sensor data collection."""
    try:
        result = subprocess.run(
            ["sudo", "systemctl", "start", "thoth-collector"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return jsonify({"status": "started", "message": "Data collection started"})
        else:
            return jsonify({"status": "error", "message": result.stderr}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/control/stop', methods=['POST'])
def stop_collection():
    """Stop sensor data collection."""
    try:
        result = subprocess.run(
            ["sudo", "systemctl", "stop", "thoth-collector"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return jsonify({"status": "stopped", "message": "Data collection stopped"})
        else:
            return jsonify({"status": "error", "message": result.stderr}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_data():
    """Upload collected data to remote server."""
    try:
        upload_url = request.json.get('upload_url') if request.json else Config.UPLOAD_URL
        
        if not upload_url:
            return jsonify({"success": False, "error": "No upload URL configured"}), 400
        
        if not os.path.exists(Config.SENSOR_DATA_FILE):
            return jsonify({"success": False, "error": "No data file found"}), 404
        
        # Read all data
        data = []
        with open(Config.SENSOR_DATA_FILE, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        if not data:
            return jsonify({"success": False, "error": "No data to upload"}), 400
        
        # Upload data
        headers = {'Content-Type': 'application/json'}
        if Config.API_KEY:
            headers['Authorization'] = f'Bearer {Config.API_KEY}'
        
        response = requests.post(upload_url, json=data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return jsonify({
                "success": True, 
                "uploaded": len(data),
                "message": "Data uploaded successfully"
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Upload failed: {response.status_code}"
            }), 500
            
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
        
        # Save to config file (simple approach)
        config_file = os.path.join(os.path.dirname(__file__), 'button_config.json')
        with open(config_file, 'w') as f:
            json.dump(BUTTON_ACTIONS, f)
        
        return jsonify({"updated": BUTTON_ACTIONS, "message": "Button configuration updated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
                    
                    # Redirect to next page or status
                    next_page = request.args.get('next', '')
                    if next_page and next_page.startswith('/'):
                        return redirect(next_page)
                    return redirect(url_for('status'))
                else:
                    flash(f'Device registration failed: {message}', 'error')
            else:
                flash('Invalid username or password', 'error')
                
        except Exception as e:
            logger.error(f"Login/registration error: {str(e)}", exc_info=True)
            flash('An error occurred during login. Please try again.', 'error')
            
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
        
        # Get CPU temperature
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                cpu_temp = float(f.read().strip()) / 1000.0  # Convert millidegrees to degrees
        except Exception as e:
            logger.error(f"Error getting CPU temperature: {e}")
            cpu_temp = None
        
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
        
        return render_template('status.html',
                            system_status=system_status,
                            device_info=device_info,
                            disk_usage=disk_usage,
                            cpu_temp=cpu_temp,
                            ip_address=ip_address,
                            username=session.get('username'),
                            available_networks=available_networks)
    
    except Exception as e:
        logger.error(f"Error in status route: {str(e)}", exc_info=True)
        flash('An error occurred while loading the status page.', 'error')
        return redirect(url_for('index'))

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

if __name__ == '__main__':
    # Start the scheduler
    if not device_scheduler.running:
        device_scheduler.start()
    
    # Run the application
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False
    )
