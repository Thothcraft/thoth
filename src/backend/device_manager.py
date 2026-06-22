"""
Device Manager for Thoth Device

This module handles device registration and status updates with the Brain server.
It manages the device's lifecycle, including registration, authentication, and
periodic status updates.
"""

import os
import json
import logging
import uuid
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .capture_manager import list_minutes, list_minute_folders, capture_files, minute_summary

# Set up logging
logger = logging.getLogger(__name__)

class DeviceManager:
    """Manages device registration and status updates with the Brain server."""

    def __init__(self, config: 'Config'):
        """Initialize the DeviceManager with configuration.

        Args:
            config: Application configuration object
        """
        self.config = config
        self.device_id = self._get_device_id()
        self.auth_token = None
        self.registered = False
        self.session = self._create_session()
        self.stop_event = threading.Event()
        self.heartbeat_thread = None
        self.device_settings = self.load_device_settings()

        # Device status
        self.status = {
            'online': False,
            'battery_level': None,
            'wifi_connected': False,
            'collection_active': False,
            'last_seen': None,
            'ip_address': None,
            'mac_address': self._get_mac_address()
        }

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _get_device_id(self) -> str:
        """Get or generate a persistent device ID."""
        device_id_file = os.path.join(self.config.DATA_DIR, 'device_id.txt')

        try:
            # Try to read existing device ID
            if os.path.exists(device_id_file):
                with open(device_id_file, 'r') as f:
                    device_id = f.read().strip()
                    if device_id:
                        return device_id

            # Generate new device ID if not found
            device_id = str(uuid.uuid4())

            # Ensure data directory exists
            os.makedirs(os.path.dirname(device_id_file), exist_ok=True)

            # Save device ID to file
            with open(device_id_file, 'w') as f:
                f.write(device_id)

            return device_id

        except Exception as e:
            logger.error(f"Error getting/generating device ID: {e}")
            # Fallback to a random UUID if file operations fail
            return str(uuid.uuid4())

    def _get_mac_address(self) -> Optional[str]:
        """Get the device's MAC address."""
        try:
            # Common interface names for Raspberry Pi
            interfaces = ['eth0', 'wlan0']

            for iface in interfaces:
                try:
                    with open(f'/sys/class/net/{iface}/address', 'r') as f:
                        return f.read().strip()
                except FileNotFoundError:
                    continue

            return None
        except Exception as e:
            logger.error(f"Error getting MAC address: {e}")
            return None

    def _config_dir(self) -> Path:
        base_dir = getattr(self.config, 'CAPTURE_DATA_DIR', None) or getattr(self.config, 'DATA_DIR', None) or self.config.DATA_DIR
        return Path(base_dir).expanduser() / 'config'

    def _settings_path(self) -> Path:
        return self._config_dir() / 'device_settings.json'

    def _model_registry_path(self) -> Path:
        return self._config_dir() / 'deployed_models.json'

    def _capture_settings_path(self) -> Path:
        return self._config_dir() / 'capture_settings.json'

    def _default_device_settings(self) -> Dict[str, Any]:
        return {
            'portal_upload_allowed': True,
            'deployment_requests_allowed': True,
            'cloud_sync_allowed': True,
            'auto_registration_enabled': True,
        }

    def _coerce_setting_value(self, key: str, value: Any) -> Any:
        if key.endswith('_allowed') or key.startswith('auto_'):
            if isinstance(value, bool):
                return value
            if value is None:
                return False
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                return value.strip().lower() in {'1', 'true', 'yes', 'on'}
            return bool(value)
        return value

    def load_device_settings(self) -> Dict[str, Any]:
        settings = self._default_device_settings()
        try:
            path = self._settings_path()
            if path.exists():
                with open(path, 'r', encoding='utf-8') as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    settings.update({k: self._coerce_setting_value(k, v) for k, v in loaded.items()})
        except Exception as e:
            logger.error(f"Error loading device settings: {e}")
        self.device_settings = settings
        return settings

    def save_device_settings(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        settings = self.load_device_settings()
        settings.update({k: self._coerce_setting_value(k, v) for k, v in (updates or {}).items()})
        try:
            path = self._settings_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(settings, handle, indent=2)
            self.device_settings = settings
            if self.registered and self.auth_token:
                self.update_status({'online': True, 'hardware_info': self._build_hardware_info()})
        except Exception as e:
            logger.error(f"Error saving device settings: {e}")
        return settings

    def _build_hardware_info(self) -> Dict[str, Any]:
        import platform
        hardware_info = {
            'local_ip': self._get_local_ip(),
            'hostname': platform.node(),
            'system': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'platform': platform.platform(),
            'is_raspberry_pi': platform.system() == 'Linux' and (
                'arm' in platform.machine().lower() or os.path.exists('/proc/device-tree/model')
            ),
            'portal_upload_allowed': bool(self.device_settings.get('portal_upload_allowed', True)),
            'deployment_requests_allowed': bool(self.device_settings.get('deployment_requests_allowed', True)),
            'cloud_sync_allowed': bool(self.device_settings.get('cloud_sync_allowed', True)),
            'capture_settings': self.load_capture_settings(),
        }
        try:
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'rb') as handle:
                    hardware_info['raspberry_pi_model'] = handle.read().decode('utf-8', errors='replace').strip('\x00\r\n ')
        except Exception:
            pass
        return hardware_info

    def get_device_settings(self) -> Dict[str, Any]:
        return self.load_device_settings()

    def default_capture_settings(self) -> Dict[str, Any]:
        return {
            'labels': [],
            'sensors': {
                'usb_camera': True,
                'dreamhat_radar': True,
                'esp32_csi': True,
                'sense_hat': True,
            },
        }

    def normalize_capture_settings(self, settings: Dict[str, Any] | None) -> Dict[str, Any]:
        source = settings if isinstance(settings, dict) else {}
        default = self.default_capture_settings()
        labels_value = source.get('labels')
        if isinstance(labels_value, str):
            labels = [item.strip() for item in labels_value.split(',') if item.strip()]
        elif isinstance(labels_value, list):
            labels = [str(item).strip() for item in labels_value if str(item).strip()]
        else:
            label = str(source.get('label') or '').strip()
            labels = [label] if label else []

        raw_sensors = source.get('sensors') if isinstance(source.get('sensors'), dict) else {}
        sensors = dict(default['sensors'])
        for key in sensors:
            if key in raw_sensors:
                sensors[key] = bool(raw_sensors.get(key))
        return {'labels': labels, 'sensors': sensors}

    def load_capture_settings(self) -> Dict[str, Any]:
        try:
            path = self._capture_settings_path()
            if path.exists():
                loaded = json.loads(path.read_text(encoding='utf-8'))
                return self.normalize_capture_settings(loaded)
        except Exception as e:
            logger.error(f"Error loading capture settings: {e}")
        return self.default_capture_settings()

    def save_capture_settings(self, settings: Dict[str, Any] | None) -> Dict[str, Any]:
        normalized = self.normalize_capture_settings(settings)
        try:
            path = self._capture_settings_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(normalized, indent=2), encoding='utf-8')
        except Exception as e:
            logger.error(f"Error saving capture settings: {e}")
        return normalized

    def _apply_response_settings(self, result: Dict[str, Any]) -> None:
        if not isinstance(result, dict):
            return
        settings = result.get('capture_settings')
        data = result.get('data')
        if settings is None and isinstance(data, dict):
            settings = data.get('capture_settings')
        if isinstance(settings, dict):
            self.save_capture_settings(settings)

    def load_model_registry(self) -> Dict[str, Any]:
        default = {'models': []}
        try:
            path = self._model_registry_path()
            if path.exists():
                return json.loads(path.read_text(encoding='utf-8'))
        except Exception as e:
            logger.error(f"Error loading model registry: {e}")
        return default

    def save_model_registry(self, registry: Dict[str, Any]) -> Dict[str, Any]:
        try:
            path = self._model_registry_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(registry, indent=2), encoding='utf-8')
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")
        return registry

    def get_running_models(self) -> List[Dict[str, Any]]:
        registry = self.load_model_registry()
        models = []
        for item in registry.get('models', []):
            if not isinstance(item, dict):
                continue
            if item.get('status') not in {'running', 'delivered'}:
                continue
            models.append(item)
        return models

    def list_pending_deployments(self) -> List[Dict[str, Any]]:
        if not self.registered or not self.auth_token:
            return []
        try:
            response = self.session.get(
                f"{self.config.BRAIN_SERVER_URL}/api/datasets/models/deployments",
                headers={"Authorization": f"Bearer {self.auth_token}"},
                timeout=15,
            )
            if response.status_code != 200:
                return []
            data = response.json()
            deployments = data.get('deployments', []) if isinstance(data, dict) else []
            pending = []
            for deployment in deployments:
                if isinstance(deployment, dict) and deployment.get('device_id') == self.device_id and deployment.get('status') == 'pending':
                    pending.append(deployment)
            return pending
        except Exception as e:
            logger.error(f"Error listing pending deployments: {e}")
            return []

    def acknowledge_deployment(self, deployment: Dict[str, Any], accepted: bool = True) -> bool:
        deployment_id = str(deployment.get('deployment_id') or '')
        if not deployment_id:
            return False
        try:
            url = f"{self.config.BRAIN_SERVER_URL}/api/device/{self.device_id}/deployment/{deployment_id}/ack"
            response = self.session.post(url, params={'status': 'delivered' if accepted else 'declined'}, timeout=15)
            if response.status_code not in (200, 201):
                logger.error("Failed to acknowledge deployment %s: %s", deployment_id, response.text)
                return False
            registry = self.load_model_registry()
            models = [item for item in registry.get('models', []) if item.get('deployment_id') != deployment_id]
            models.append({
                'deployment_id': deployment_id,
                'model_name': deployment.get('model_name') or 'Unknown model',
                'model_type': deployment.get('model_type') or 'unknown',
                'device_id': self.device_id,
                'device_name': deployment.get('device_name') or self.device_id,
                'status': 'running' if accepted else 'declined',
                'accepted_at': datetime.utcnow().isoformat() if accepted else None,
                'timeline': [],
            })
            registry['models'] = models
            self.save_model_registry(registry)
            return True
        except Exception as e:
            logger.error(f"Error acknowledging deployment: {e}")
            return False

    def _get_local_ip(self) -> Optional[str]:
        """Get the device's local IP address."""
        try:
            # Try to get IP address by connecting to a known IP and checking the local address
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                # Doesn't need to be reachable
                s.connect(('10.254.254.254', 1))
                ip = s.getsockname()[0]
            except Exception:
                ip = '127.0.0.1'
            finally:
                s.close()

            # If we got a non-loopback address, return it
            if ip != '127.0.0.1':
                return ip

            # Fallback: try to get IP from network interfaces
            import netifaces
            for iface in netifaces.interfaces():
                addrs = netifaces.ifaddresses(iface)
                if netifaces.AF_INET in addrs:
                    for addr in addrs[netifaces.AF_INET]:
                        if 'addr' in addr and addr['addr'] != '127.0.0.1':
                            return addr['addr']

            return None
        except Exception as e:
            logger.error(f"Error getting local IP address: {e}")
            return None

    def register_device(self, user_token: str) -> Tuple[bool, str]:
        """Register the device with the Brain server.

        Args:
            user_token: User authentication token from login

        Returns:
            Tuple of (success, message)
        """
        if not self.config.BRAIN_SERVER_URL:
            return False, "Brain server URL not configured"

        url = f"{self.config.BRAIN_SERVER_URL}/api/device/register"

        # Get device information
        try:
            # Get OS information
            with open('/etc/os-release', 'r') as f:
                os_info = dict(
                    line.strip().replace('"', '').split('=', 1)
                    for line in f if '=' in line
                )
            os_name = os_info.get('PRETTY_NAME', 'Raspberry Pi OS')
            os_version = os_info.get('VERSION_ID', '')

            # Get Python version
            import platform
            python_version = platform.python_version()
            is_raspberry_pi = platform.system() == 'Linux' and (
                'arm' in platform.machine().lower() or os.path.exists('/proc/device-tree/model')
            )
            raspberry_pi_model = None
            try:
                if os.path.exists('/proc/device-tree/model'):
                    with open('/proc/device-tree/model', 'rb') as handle:
                        raspberry_pi_model = handle.read().decode('utf-8', errors='replace').strip('\x00\r\n ')
            except Exception:
                raspberry_pi_model = None


            # Get local IP address
            local_ip = self._get_local_ip()

            # Prepare registration data
            hardware_info = self._build_hardware_info()
            data = {
                "device_id": self.device_id,
                "device_name": f"Thoth-{self.device_id[:8]}",
                "device_type": "thoth",
                "os_version": f"{os_name} {os_version}",
                "app_version": self.config.VERSION if hasattr(self.config, 'VERSION') else "1.0.0",
                "mac_address": self.status['mac_address'],
                "ip_address": local_ip,
                "hardware_info": hardware_info,
            }

            # Send registration request
            headers = {
                "Authorization": f"Bearer {user_token}",
                "Content-Type": "application/json"
            }

            response = self.session.post(
                url,
                json=data,
                headers=headers,
                timeout=10
            )

            if response.status_code in (200, 201):
                result = response.json()
                self._apply_response_settings(result)
                self.registered = True
                self.auth_token = user_token

                if 'device_name' in result:
                    data['device_name'] = result['device_name']

                self._save_registration_info(data, user_token)
                self.status['online'] = True

                logger.info(f"Device registered successfully: {self.device_id}")
                return True, "Device registered successfully"

            error_msg = f"Registration failed: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return False, error_msg

        except requests.exceptions.RequestException as e:
            error_msg = f"Error connecting to Brain server: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Error during device registration: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    def _get_data_files_list(self) -> List[Dict[str, Any]]:
        """Get list of capture files in the data directory.

        Returns:
            List of file records compatible with the Brain device registry.
        """
        files_list = []
        try:
            for minute_dir in list_minute_folders():
                summary = minute_summary(minute_dir)
                relative_path = str(summary.get('relative_path') or minute_dir.name)
                for _kind, file_path in capture_files(minute_dir).items():
                    if not file_path or not file_path.exists():
                        continue
                    stat = file_path.stat()
                    files_list.append({
                        'name': f"{relative_path}/{file_path.name}",
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'type': file_path.suffix.lstrip('.') or 'file',
                    })

            logger.info(f"Found {len(files_list)} capture files to report")

        except Exception as e:
            logger.error(f"Error getting data files list: {e}")

        return files_list

    def _save_registration_info(self, device_info: Dict[str, Any], auth_token: str) -> None:
        """Save device registration information to disk."""
        try:
            config_dir = os.path.join(self.config.DATA_DIR, 'config')
            os.makedirs(config_dir, exist_ok=True)

            config_file = os.path.join(config_dir, 'device_config.json')

            config_data = {
                'device_id': device_info['device_id'],
                'device_name': device_info.get('device_name', f"Thoth-{device_info['device_id'][:8]}"),
                'device_type': device_info.get('device_type', 'thoth'),
                'registered_at': datetime.utcnow().isoformat(),
                'auth_token': auth_token,
                'brain_server_url': self.config.BRAIN_SERVER_URL
            }

            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving registration info: {e}")

    def load_registration_info(self) -> Optional[Dict[str, Any]]:
        """Load device registration information from disk."""
        try:
            config_file = os.path.join(self.config.DATA_DIR, 'config', 'device_config.json')

            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config_data = json.load(f)

                    # Update instance state
                    self.device_id = config_data.get('device_id', self.device_id)
                    self.auth_token = None
                    self.registered = False

                    # Keep only the stable device identity; online state must
                    # come from a fresh login in the current session.
                    config_data.pop('auth_token', None)
                    return config_data

            return None

        except Exception as e:
            logger.error(f"Error loading registration info: {e}")
            return None

    def mark_device_offline(self) -> bool:
        """Notify Brain that the device is offline and stop local registration state."""
        try:
            if self.heartbeat_thread and self.heartbeat_thread.is_alive():
                self.stop_heartbeat()

            self.status['online'] = False
            self.status['last_seen'] = datetime.utcnow().isoformat()
            self.registered = False

            if not self.config.BRAIN_SERVER_URL:
                return False

            url = f"{self.config.BRAIN_SERVER_URL}/api/device/{self.device_id}/offline"
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            response = self.session.post(url, headers=headers, timeout=10)
            if response.status_code in (200, 201, 204):
                logger.info(f"Device {self.device_id} marked offline on Brain")
                return True

            logger.warning(
                "Failed to mark device offline on Brain: %s - %s",
                response.status_code,
                response.text,
            )
            return False
        except Exception as e:
            logger.error(f"Error marking device offline: {e}", exc_info=True)
            return False

    def update_status(self, status_updates: Dict[str, Any]) -> bool:
        """Update device status on the Brain server.

        Args:
            status_updates: Dictionary of status fields to update

        Returns:
            bool: True if update was successful, False otherwise
        """
        if not self.registered or not self.auth_token:
            logger.warning("Cannot update status: Device not registered")
            return False

        # Update local status
        self.status.update(status_updates)
        self.status['last_seen'] = datetime.utcnow().isoformat()

        # Prepare heartbeat data
        data = {
            "device_id": self.device_id,
            "files": self._get_data_files_list(),
            **status_updates
        }

        # Add timestamp if not provided
        if 'timestamp' not in data:
            data['timestamp'] = datetime.utcnow().isoformat()

        try:
            url = f"{self.config.BRAIN_SERVER_URL}/api/device/heartbeat"

            headers = {
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json"
            }

            response = self.session.post(
                url,
                json=data,
                headers=headers,
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                self._apply_response_settings(result)
                logger.debug(f"Status update successful: {result}")
                return True
            else:
                logger.error(f"Status update failed: {response.status_code} - {response.text}")
                # If unauthorized, mark as unregistered to trigger re-registration
                if response.status_code == 401:
                    self.registered = False
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending status update: {str(e)}")
            return False

    def start_heartbeat(self, interval: int = 60) -> None:
        """Start periodic status updates to the Brain server.

        Args:
            interval: Heartbeat interval in seconds (default: 60)
        """
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            logger.warning("Heartbeat thread already running")
            return

        self.stop_event.clear()

        def heartbeat_loop():
            while not self.stop_event.is_set():
                try:
                    # Update status with current system information
                    self.update_status({
                        'battery_level': self.status.get('battery_level'),
                        'wifi_connected': self.status.get('wifi_connected', False),
                        'collection_active': self.status.get('collection_active', False),
                        'online': True,
                        'hardware_info': self._build_hardware_info(),
                    })
                except Exception as e:
                    logger.error(f"Error in heartbeat loop: {e}")

                # Wait for the next heartbeat
                self.stop_event.wait(interval)

        self.heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            name="DeviceHeartbeat",
            daemon=True
        )
        self.heartbeat_thread.start()
        logger.info(f"Started heartbeat thread (interval: {interval}s)")

    def stop_heartbeat(self) -> None:
        """Stop the periodic status updates."""
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.stop_event.set()
            self.heartbeat_thread.join(timeout=5)
            logger.info("Stopped heartbeat thread")

    def get_device_info(self) -> Dict[str, Any]:
        """Get current device information."""
        return {
            'device_id': self.device_id,
            'registered': self.registered,
            'status': self.status,
            'brain_server': self.config.BRAIN_SERVER_URL if hasattr(self.config, 'BRAIN_SERVER_URL') else None
        }

    def sync_files_to_cloud(self) -> Tuple[int, int, list]:
        """Sync local data files to the Brain server.

        Returns:
            Tuple of (uploaded_count, skipped_count, errors)
        """
        if not self.registered or not self.auth_token:
            logger.warning("Cannot sync files: Device not registered")
            return 0, 0, ["Device not registered"]

        if not bool(self.device_settings.get('cloud_sync_allowed', True)):
            logger.info("Cloud sync is disabled in device settings")
            return 0, 0, ["Cloud sync disabled"]

        import base64

        uploaded = 0
        skipped = 0
        errors = []

        try:
            data_dir = self.config.CAPTURE_DATA_DIR if hasattr(self.config, 'CAPTURE_DATA_DIR') else self.config.DATA_DIR
            if not os.path.exists(data_dir):
                return 0, 0, ["Data directory not found"]

            # Get list of minute folders and their files
            local_files = []
            for minute_dir in list_minute_folders():
                summary = minute_summary(minute_dir)
                files = capture_files(minute_dir)
                for name, file_path in files.items():
                    if file_path and file_path.exists():
                        local_files.append((minute_dir.name, file_path, summary))

            if not local_files:
                logger.info("No data files to sync")
                return 0, 0, []

            # Get list of files already on cloud
            url = f"{self.config.BRAIN_SERVER_URL}/file/files"
            headers = {
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json"
            }

            cloud_files = set()
            try:
                response = self.session.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    for f in data.get('files', []):
                        cloud_files.add(f.get('filename', ''))
            except Exception as e:
                logger.warning(f"Could not fetch cloud files: {e}")

            # Upload files not already on cloud
            for minute_name, file_path, summary in local_files:
                relative_path = str(summary.get("relative_path") or minute_name)
                filename_prefix = relative_path.replace("/", "_")
                filename = f"{filename_prefix}_{file_path.name}"
                if filename in cloud_files:
                    skipped += 1
                    continue

                try:
                    # Read and encode file
                    with open(file_path, 'rb') as f:
                        content = base64.b64encode(f.read()).decode('utf-8')

                    file_size = os.path.getsize(file_path)

                    # Upload to Brain server
                    upload_url = f"{self.config.BRAIN_SERVER_URL}/file/upload"
                    upload_data = {
                        "filename": filename,
                        "content": content,
                        "device_id": self.device_id,
                        "metadata": {
                            "source": "thoth_device",
                            "device_id": self.device_id,
                            "original_size": file_size,
                            "minute": minute_name,
                            "relative_path": relative_path,
                            "label": summary.get("label"),
                            "labels": summary.get("labels", []),
                        }
                    }

                    response = self.session.post(
                        upload_url,
                        json=upload_data,
                        headers=headers,
                        timeout=60
                    )

                    if response.status_code in [200, 201]:
                        uploaded += 1
                        logger.info(f"Uploaded {filename} to cloud")
                    else:
                        errors.append(f"{filename}: {response.status_code}")
                        logger.error(f"Failed to upload {filename}: {response.status_code}")

                except Exception as e:
                    errors.append(f"{filename}: {str(e)}")
                    logger.error(f"Error uploading {filename}: {e}")

            logger.info(f"File sync complete: {uploaded} uploaded, {skipped} skipped, {len(errors)} errors")
            return uploaded, skipped, errors

        except Exception as e:
            logger.error(f"Error during file sync: {e}")
            return uploaded, skipped, [str(e)]

    def __del__(self):
        """Clean up resources."""
        self.stop_heartbeat()
        if hasattr(self, 'session'):
            self.session.close()
