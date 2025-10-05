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
from typing import Dict, Optional, Any, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
    
    def register_device(self, user_token: str) -> Tuple[bool, str]:
        """Register the device with the Brain server.
        
        Args:
            user_token: User authentication token from login
            
        Returns:
            Tuple of (success, message)
        """
        if not self.config.BRAIN_SERVER_URL:
            return False, "Brain server URL not configured"
        
        url = f"{self.config.BRAIN_SERVER_URL}/device/register"
        
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
            

            # Prepare registration data
            data = {
                "device_id": self.device_id,
                "device_name": f"Thoth-{self.device_id[:8]}",
                "device_type": "thoth",
                "os_version": f"{os_name} {os_version}",
                "app_version": self.config.VERSION if hasattr(self.config, 'VERSION') else "1.0.0",
                "mac_address": self.status['mac_address']
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
            
            if response.status_code == 200 or response.status_code == 201:
                result = response.json()
                self.registered = True
                self.auth_token = user_token
                
                # Update device name if provided in response
                if 'device_name' in result:
                    data['device_name'] = result['device_name']
                
                # Save registration info
                self._save_registration_info(data, user_token)
                
                logger.info(f"Device registered successfully: {self.device_id}")
                return True, "Device registered successfully"
            else:
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
                    self.auth_token = config_data.get('auth_token')
                    self.registered = True
                    
                    return config_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading registration info: {e}")
            return None
    
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
            **status_updates
        }
        
        # Add timestamp if not provided
        if 'timestamp' not in data:
            data['timestamp'] = datetime.utcnow().isoformat()
        
        try:
            url = f"{self.config.BRAIN_SERVER_URL}/device/heartbeat"
            
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
                        'online': True
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
    
    def __del__(self):
        """Clean up resources."""
        self.stop_heartbeat()
        if hasattr(self, 'session'):
            self.session.close()
