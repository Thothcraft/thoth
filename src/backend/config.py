"""Configuration module for Thoth backend."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration class."""
    
    # Flask configuration
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'thoth-default-secret-key')
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT = int(os.getenv('FLASK_PORT', 5000))
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # WiFi configuration
    WIFI_SSID = os.getenv('WIFI_SSID', '')
    WIFI_PASSWORD = os.getenv('WIFI_PASSWORD', '')
    AP_SSID = os.getenv('AP_SSID', 'Thoth-AP')
    AP_PASSWORD = os.getenv('AP_PASSWORD', 'thoth123')
    
    # Upload configuration
    UPLOAD_URL = os.getenv('UPLOAD_URL', '')
    API_KEY = os.getenv('API_KEY', '')
    
    # PiSugar configuration
    PISUGAR_MODEL = os.getenv('PISUGAR_MODEL', 'PiSugar 2 Pro')
    
    # Data collection
    COLLECTION_RATE = float(os.getenv('COLLECTION_RATE', 1.0))
    DATA_RETENTION_DAYS = int(os.getenv('DATA_RETENTION_DAYS', 30))
    
    # File paths
    BASE_DIR = '/opt/thoth'
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    LOGS_DIR = os.path.join(DATA_DIR, 'logs')
    SENSOR_DATA_FILE = os.path.join(LOGS_DIR, 'sensors.json')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', os.path.join(LOGS_DIR, 'thoth.log'))

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
