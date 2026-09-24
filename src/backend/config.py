"""Configuration module for Thoth backend."""

import os
import secrets
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = os.path.join(Path(__file__).parent.parent.parent, '.env')
load_dotenv(env_path)


def _persisted_secret(config_dir: str, filename: str, length: int = 32) -> str:
    """Return a per-device secret, generating and persisting it on first use.

    Avoids shipping a shared hardcoded credential while keeping the value
    stable across restarts. The file is created with owner-only permissions.
    """
    path = os.path.join(config_dir, filename)
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as handle:
                value = handle.read().strip()
            if value:
                return value
        os.makedirs(config_dir, exist_ok=True)
        value = secrets.token_urlsafe(length)
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(value)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return value
    except OSError:
        # Last resort: ephemeral secret. Sessions reset on restart but no
        # shared credential is ever used.
        return secrets.token_urlsafe(length)


def _resolve_bind_host() -> str:
    """Resolve the Flask bind host.

    Default is loopback-only. LAN exposure requires the explicit
    ``THOTH_BIND_MODE=lan`` opt-in (or a direct ``FLASK_HOST`` override).
    """
    explicit_host = os.getenv('FLASK_HOST')
    if explicit_host:
        return explicit_host
    mode = os.getenv('THOTH_BIND_MODE', 'loopback').strip().lower()
    return '0.0.0.0' if mode == 'lan' else '127.0.0.1'

class Config:
    """Base configuration class for Thoth device."""

    # Application info
    APP_NAME = 'Thoth Device'
    VERSION = '1.0.0'

    # Flask configuration
    # SECRET_KEY: env override, else a per-device persisted random secret.
    # Never falls back to a shared hardcoded value.
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY') or _persisted_secret(
        os.getenv('THOTH_CONFIG_DIR', os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'config')),
        'flask_secret_key')
    # Loopback by default; set THOTH_BIND_MODE=lan (or FLASK_HOST) to expose.
    BIND_MODE = os.getenv('THOTH_BIND_MODE', 'loopback').strip().lower()
    HOST = _resolve_bind_host()
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
    AP_SSID = os.getenv('AP_SSID', 'Thoth')
    # Per-device generated AP password unless explicitly configured.
    AP_PASSWORD = os.getenv('AP_PASSWORD') or _persisted_secret(
        os.getenv('THOTH_CONFIG_DIR', os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'config')),
        'ap_password', length=12)
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
    CAPTURE_DATA_DIR = os.getenv('CAPTURE_DATA_DIR', os.path.join(BASE_DIR, 'data'))
    LEGACY_CONFIG_DIR = os.path.join(DATA_DIR, 'config')
    LEGACY_DEVICE_ID_FILE = os.path.join(DATA_DIR, 'device_id.txt')
    CAPTURE_MAX_DISK_PERCENT = min(99.0, max(1.0, float(os.getenv('CAPTURE_MAX_DISK_PERCENT', 95.0))))
    CAPTURE_CAMERA_DEVICE = os.getenv('CAPTURE_CAMERA_DEVICE', '/dev/video0')
    CAPTURE_CAMERA_WIDTH = int(os.getenv('CAPTURE_CAMERA_WIDTH', 640))
    CAPTURE_CAMERA_HEIGHT = int(os.getenv('CAPTURE_CAMERA_HEIGHT', 480))
    CAPTURE_CAMERA_FPS = int(os.getenv('CAPTURE_CAMERA_FPS', 30))
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    CONFIG_DIR = os.getenv('THOTH_CONFIG_DIR', os.path.join(BASE_DIR, 'config'))
    DEVICE_ID_FILE = os.path.join(CONFIG_DIR, 'device_id.txt')
    SENSOR_DATA_FILE = os.path.join(DATA_DIR, 'sensor_data.json')

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.path.join(LOGS_DIR, 'thoth.log')

    # Install/runtime profile: "full" (Pi 4/5) or "lite" (Pi 3 / 1 GB boards).
    # Set by setup/first-boot.sh into the systemd units; overridable via env.
    PROFILE = os.getenv('THOTH_PROFILE', 'full').strip().lower() or 'full'
    IS_LITE = PROFILE == 'lite'

    # Device management
    # Lite mode backs off the Brain registration retry + heartbeat to cut the
    # network/thread churn that stalls a 1 GB Pi when registration is failing.
    HEARTBEAT_INTERVAL = int(os.getenv('HEARTBEAT_INTERVAL', 30 if IS_LITE else 10))  # seconds
    REGISTER_INTERVAL = int(os.getenv('THOTH_REGISTER_INTERVAL_S', 30 if IS_LITE else 10))  # seconds
    MAX_HEARTBEAT_FAILURES = int(os.getenv('MAX_HEARTBEAT_FAILURES', 3))

    # Captive portal
    CAPTIVE_PORTAL_PORT = int(os.getenv('CAPTIVE_PORTAL_PORT', 5000))
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
