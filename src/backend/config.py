"""Configuration module for Thoth backend."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = os.path.join(Path(__file__).parent.parent.parent, '.env')
load_dotenv(env_path)

class Config:
    """Base configuration class for Thoth device."""
    
    # Application info
    APP_NAME = 'Thoth Device'
    VERSION = '1.0.0'
    
    # Flask configuration
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'thoth-dev-secret-key')
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT = int(os.getenv('FLASK_PORT', 5000))
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Device configuration
    DEVICE_NAME = os.getenv('DEVICE_NAME', 'Thoth-Device')
    DEVICE_TYPE = 'thoth'
    
    # Brain server configuration
    BRAIN_SERVER_URL = os.getenv('BRAIN_SERVER_URL', 'https://web-production-d7d37.up.railway.app')
    BRAIN_AUTH_TOKEN = os.getenv('BRAIN_AUTH_TOKEN', '')
    
    # WiFi configuration
    WIFI_SSID = os.getenv('WIFI_SSID', '')
    WIFI_PASSWORD = os.getenv('WIFI_PASSWORD', '')
    AP_SSID = os.getenv('AP_SSID', 'Thoth-AP')
    AP_PASSWORD = os.getenv('AP_PASSWORD', 'thoth123')
    AP_IP = '192.168.4.1'
    AP_NETMASK = '255.255.255.0'
    
    # Upload configuration
    UPLOAD_URL = os.getenv('UPLOAD_URL', '')
    API_KEY = os.getenv('API_KEY', '')
    
    # PiSugar configuration
    PISUGAR_MODEL = os.getenv('PISUGAR_MODEL', 'PiSugar 2 Pro')
    
    # Data collection
    COLLECTION_RATE = float(os.getenv('COLLECTION_RATE', 1.0))  # Hz
    DATA_RETENTION_DAYS = int(os.getenv('DATA_RETENTION_DAYS', 30))
    
    # File paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    CONFIG_DIR = os.path.join(DATA_DIR, 'config')
    SENSOR_DATA_FILE = os.path.join(DATA_DIR, 'sensor_data.json')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.path.join(LOGS_DIR, 'thoth.log')
    
    # Device management
    HEARTBEAT_INTERVAL = int(os.getenv('HEARTBEAT_INTERVAL', 10))  # seconds
    MAX_HEARTBEAT_FAILURES = int(os.getenv('MAX_HEARTBEAT_FAILURES', 3))
    
    # Captive portal
    CAPTIVE_PORTAL_PORT = int(os.getenv('CAPTIVE_PORTAL_PORT', 80))
    CAPTIVE_PORTAL_TIMEOUT = int(os.getenv('CAPTIVE_PORTAL_TIMEOUT', 300))  # seconds

# Button action configuration (can be modified via API)
BUTTON_ACTIONS = {
    "single": "toggle_collection",
    "double": "start_ap", 
    "long": "shutdown"
}

# Sensor configuration
SENSOR_CONFIG = {
    "imu_enabled": True,
    "compass_enabled": True,
    "gyro_enabled": True,
    "accel_enabled": True,
    "sample_rate": 1.0,  # Hz
    "calibration_required": False
}
