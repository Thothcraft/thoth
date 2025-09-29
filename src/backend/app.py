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
from datetime import datetime
from typing import Dict, List, Optional, Any

from flask import Flask, jsonify, request, render_template_string
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import requests

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import Config, BUTTON_ACTIONS, SENSOR_CONFIG
from backend.models import SensorReading, SystemStatus, ButtonConfig, UploadResult

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global state
collection_active = False
wifi_manager = None

def get_system_status() -> SystemStatus:
    """Get current system status."""
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
        battery_level = None
        try:
            # This would use the actual PiSugar library
            # For now, return a placeholder
            battery_level = 85  # Placeholder
        except Exception:
            pass
        
        # Get uptime
        uptime_output = subprocess.run(
            ["uptime", "-p"], capture_output=True, text=True
        ).stdout.strip()
        
        return SystemStatus(
            status="ok",
            battery_level=battery_level,
            wifi_connected=wifi_connected,
            ap_mode=not wifi_connected,
            collection_active=collection_status,
            uptime=uptime_output
        )
    except Exception as e:
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

# Start background task
socketio.start_background_task(tail_sensor_data)

# Routes
@app.route('/')
def index():
    """Serve basic web interface."""
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

@app.route('/wifi-config', methods=['POST'])
def wifi_config():
    """Handle WiFi configuration from captive portal."""
    try:
        ssid = request.form.get('ssid')
        password = request.form.get('password')
        
        if not ssid:
            return jsonify({"error": "SSID required"}), 400
        
        # This would call the WiFi manager
        # For now, return success
        return jsonify({"status": "connecting", "ssid": ssid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('status', {'message': 'Connected to Thoth device'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')

if __name__ == '__main__':
    # Ensure data directory exists
    os.makedirs(Config.LOGS_DIR, exist_ok=True)
    
    # Run the application
    socketio.run(
        app, 
        host=Config.HOST, 
        port=Config.PORT, 
        debug=Config.DEBUG
    )
