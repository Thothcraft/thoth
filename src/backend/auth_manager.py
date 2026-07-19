"""
Authentication Manager for Thoth Device

This module handles authentication with the Brain server, including token management
and session handling for the Thoth device.
"""

import os
import json
import logging
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)


def _decode_unverified_jwt(token: str) -> Dict[str, Any]:
    """Decode a JWT payload without verifying the signature."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode("utf-8")).decode("utf-8")
        return json.loads(decoded)
    except Exception as exc:
        raise ValueError(f"Invalid JWT payload: {exc}") from exc

class AuthManager:
    """Manages authentication with the Brain server."""

    def __init__(self, config: 'Config'):
        """Initialize the AuthManager with configuration.

        Args:
            config: Application configuration object
        """
        self.config = config
        self.token = None
        self.token_expiry = None
        self.refresh_token = None
        self.user_info = None
        self.pairing_session = None

        # Load saved auth data if available
        self._load_auth_data()
        self._load_pairing_session()

    def _pairing_file(self) -> str:
        return os.path.join(self.config.CONFIG_DIR, 'pairing.json')

    def _load_pairing_session(self) -> None:
        try:
            pairing_file = self._pairing_file()
            if os.path.exists(pairing_file):
                with open(pairing_file, 'r') as handle:
                    value = json.load(handle)
                if isinstance(value, dict) and value.get('pairing_secret'):
                    self.pairing_session = value
        except Exception as exc:
            logger.warning("Unable to restore pairing session: %s", exc)
            self._clear_pairing_session()

    def _save_pairing_session(self) -> None:
        os.makedirs(self.config.CONFIG_DIR, exist_ok=True)
        pairing_file = self._pairing_file()
        with open(pairing_file, 'w') as handle:
            json.dump(self.pairing_session or {}, handle, indent=2)
        os.chmod(pairing_file, 0o600)

    def _clear_pairing_session(self) -> None:
        self.pairing_session = None
        try:
            pairing_file = self._pairing_file()
            if os.path.exists(pairing_file):
                os.remove(pairing_file)
        except OSError:
            logger.warning("Unable to remove pairing session", exc_info=True)

    def _load_auth_data(self) -> None:
        """Load authentication data from disk."""
        try:
            auth_file = os.path.join(self.config.CONFIG_DIR, 'auth.json')
            legacy_auth_file = os.path.join(getattr(self.config, 'LEGACY_CONFIG_DIR', os.path.join(self.config.DATA_DIR, 'config')), 'auth.json')
            auth_data = None
            source_file = None
            if os.path.exists(auth_file):
                source_file = auth_file
                with open(auth_file, 'r') as f:
                    auth_data = json.load(f)
            elif os.path.exists(legacy_auth_file):
                source_file = legacy_auth_file
                with open(legacy_auth_file, 'r') as f:
                    auth_data = json.load(f)
            elif getattr(self.config, 'BRAIN_AUTH_TOKEN', None):
                auth_data = {'token': self.config.BRAIN_AUTH_TOKEN}

            if not isinstance(auth_data, dict) or not auth_data.get('token'):
                return

            token = str(auth_data.get('token') or '').strip()
            token_data = _decode_unverified_jwt(token)
            expiry = datetime.utcfromtimestamp(token_data['exp']) if token_data.get('exp') else None
            if expiry is None:
                logger.info("Saved authentication token has no expiry")
                self.logout()
                return
            if expiry and datetime.utcnow() >= expiry:
                logger.info("Saved authentication token expired")
                self.logout()
                return

            self.token = token
            self.refresh_token = auth_data.get('refresh_token')
            self.token_expiry = expiry
            self.user_info = auth_data.get('user_info') or {
                'username': token_data.get('username') or token_data.get('sub'),
                'user_id': token_data.get('user_id') or token_data.get('sub'),
                'email': token_data.get('email'),
                'role': token_data.get('role'),
                'scopes': token_data.get('scopes', []),
            }
            self.config.USER_AUTH_TOKEN = token
            self.config.BRAIN_AUTH_TOKEN = token
            if source_file != auth_file:
                self._save_auth_data()
            logger.info("Loaded saved Brain authentication token for headless heartbeat")

        except Exception as e:
            logger.error(f"Error loading auth data: {e}")
            # A malformed or otherwise unusable token must not remain in the
            # runtime config, where the registration scheduler would retry it.
            self.logout()

    def _save_auth_data(self) -> None:
        """Save authentication data to disk."""
        try:
            os.makedirs(self.config.CONFIG_DIR, exist_ok=True)
            auth_file = os.path.join(self.config.CONFIG_DIR, 'auth.json')
            legacy_auth_file = os.path.join(getattr(self.config, 'LEGACY_CONFIG_DIR', os.path.join(self.config.DATA_DIR, 'config')), 'auth.json')

            auth_data = {
                'token': self.token,
                'refresh_token': self.refresh_token,
                'token_expiry': self.token_expiry.isoformat() if self.token_expiry else None,
                'user_info': self.user_info
            }

            with open(auth_file, 'w') as f:
                json.dump(auth_data, f, indent=2)
            if legacy_auth_file != auth_file:
                try:
                    with open(legacy_auth_file, 'w') as f:
                        json.dump(auth_data, f, indent=2)
                except Exception:
                    pass

            logger.debug("Saved authentication data to disk")

        except Exception as e:
            logger.error(f"Error saving auth data: {e}")

    def is_authenticated(self) -> bool:
        """Check if the user is currently authenticated.

        Returns:
            bool: True if authenticated, False otherwise
        """
        if not self.token:
            return False

        # Check if token is expired
        if self.token_expiry and datetime.utcnow() >= self.token_expiry:
            logger.info("Authentication token expired")
            self.logout()
            return False

        return True

    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests.

        Returns:
            Dict containing Authorization header
        """
        if not self.is_authenticated():
            return {}

        return {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }

    def start_pairing(
        self,
        device_id: str,
        device_name: str,
        hardware_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Open a one-time thothHUB pairing session for this device."""
        import requests

        response = requests.post(
            f"{self.config.BRAIN_SERVER_URL}/api/device/pairing/start",
            json={
                'device_id': device_id,
                'device_name': device_name,
                'device_type': 'thoth',
                'hardware_info': hardware_info or {},
            },
            headers={
                'Authorization': f'Bearer {self.token}',
                'Content-Type': 'application/json',
            } if self.token else {'Content-Type': 'application/json'},
            timeout=(5, 20),
        )
        data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
        if not response.ok:
            raise Exception(data.get('detail') or f"Pairing service returned HTTP {response.status_code}")
        self.pairing_session = {
            'code': data['code'],
            'pairing_secret': data['pairing_secret'],
            'device_id': data['device_id'],
            'expires_at': data['expires_at'],
        }
        self._save_pairing_session()
        return {key: value for key, value in self.pairing_session.items() if key != 'pairing_secret'}

    def pairing_status(self) -> Dict[str, Any]:
        """Poll Brain and persist the device-scoped token after a claim."""
        import requests

        if not self.pairing_session or not self.pairing_session.get('pairing_secret'):
            return {'success': True, 'status': 'idle'}
        response = requests.get(
            f"{self.config.BRAIN_SERVER_URL}/api/device/pairing/status",
            headers={'X-Pairing-Secret': self.pairing_session['pairing_secret']},
            timeout=(5, 15),
        )
        data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
        if response.status_code in (404, 410):
            self._clear_pairing_session()
            return {'success': False, 'status': 'expired', 'message': data.get('detail') or 'Pairing code expired'}
        if not response.ok:
            raise Exception(data.get('detail') or f"Pairing status returned HTTP {response.status_code}")
        if data.get('status') != 'paired':
            return {
                'success': True,
                'status': 'pending',
                'code': self.pairing_session.get('code'),
                'expires_at': self.pairing_session.get('expires_at'),
            }

        token = str(data.get('access_token') or '')
        token_data = _decode_unverified_jwt(token)
        self.token = token
        self.token_expiry = datetime.utcfromtimestamp(token_data['exp'])
        self.user_info = data.get('user') or {
            'username': token_data.get('username'),
            'user_id': token_data.get('sub'),
            'email': token_data.get('email'),
        }
        self.config.USER_AUTH_TOKEN = token
        self.config.BRAIN_AUTH_TOKEN = token
        self._save_auth_data()
        self._clear_pairing_session()
        return {
            'success': True,
            'status': 'paired',
            'token': token,
            'user': self.user_info,
            'device_id': data.get('device_id'),
            'device_name': data.get('device_name'),
        }

    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate with the Brain server.

        Args:
            username: User's username or email
            password: User's password

        Returns:
            Dict containing login result and user info

        Raises:
            Exception: If login fails
        """
        import requests
        from requests.adapters import HTTPAdapter
        from requests.exceptions import RequestException
        from urllib3.util.retry import Retry

        if not self.config.BRAIN_SERVER_URL:
            raise Exception("Brain server URL not configured")

        try:
            # Railway may need several seconds to wake after an idle period.
            # Keep connection failures bounded, but allow a cold login request
            # enough time to complete.
            client = requests.Session()
            client.mount('https://', HTTPAdapter(max_retries=Retry(
                total=1,
                connect=1,
                read=0,
                status=1,
                backoff_factor=0.4,
                status_forcelist=(502, 503, 504),
                allowed_methods=frozenset({'POST'}),
                raise_on_status=False,
            )))
            response = client.post(
                f"{self.config.BRAIN_SERVER_URL}/api/token",
                json={
                    'username': username,
                    'password': password
                },
                headers={
                    'accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                timeout=(5, 35),
            )

            if response.status_code == 200:
                result = response.json()

                # Parse token to get expiry
                token_data = _decode_unverified_jwt(result['access_token'])

                # Update auth state
                self.token = result['access_token']
                self.refresh_token = result.get('refresh_token')
                self.token_expiry = datetime.utcfromtimestamp(token_data['exp'])
                self.user_info = {
                    'username': result.get('username', token_data.get('sub', username)),
                    'user_id': result.get('user_id'),
                    'email': token_data.get('email'),
                    'role': result.get('role', token_data.get('role')),
                    'scopes': token_data.get('scopes', [])
                }

                self.config.USER_AUTH_TOKEN = self.token
                self.config.BRAIN_AUTH_TOKEN = self.token

                # Save auth data
                self._save_auth_data()

                logger.info(f"Successfully logged in as {self.user_info.get('username')}")

                return {
                    'success': True,
                    'user': self.user_info,
                    'token': self.token,
                    'expires_in': (self.token_expiry - datetime.utcnow()).total_seconds()
                }
            else:
                try:
                    detail = response.json().get('detail')
                except (ValueError, AttributeError):
                    detail = None
                error_msg = str(detail or f"Brain login returned HTTP {response.status_code}")
                logger.error(error_msg)
                raise Exception(error_msg)

        except RequestException as e:
            error_msg = f"Error connecting to Brain server: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception:
            raise

    def refresh_auth_token(self) -> bool:
        """Refresh the authentication token using the refresh token.

        Returns:
            bool: True if token was refreshed successfully, False otherwise
        """
        if not self.refresh_token:
            logger.warning("No refresh token available")
            return False

        import requests
        from requests.exceptions import RequestException

        try:
            url = f"{self.config.BRAIN_SERVER_URL}/auth/refresh-token"

            response = requests.post(
                url,
                json={"refresh_token": self.refresh_token},
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()

                # Parse new token
                token_data = _decode_unverified_jwt(result['access_token'])

                # Update auth state
                self.token = result['access_token']
                self.token_expiry = datetime.utcfromtimestamp(token_data['exp'])
                self.config.USER_AUTH_TOKEN = self.token
                self.config.BRAIN_AUTH_TOKEN = self.token

                # Save the new refresh token if provided
                if 'refresh_token' in result:
                    self.refresh_token = result['refresh_token']

                # Save auth data
                self._save_auth_data()

                logger.info("Successfully refreshed authentication token")
                return True
            else:
                logger.error(f"Token refresh failed: {response.status_code} - {response.text}")
                self.logout()
                return False

        except RequestException as e:
            logger.error(f"Error refreshing token: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error during token refresh: {str(e)}", exc_info=True)
            self.logout()
            return False

    def logout(self) -> None:
        """Clear authentication data and logout the user."""
        # Clear in-memory data
        self.token = None
        self.refresh_token = None
        self.token_expiry = None
        self.user_info = None
        # The scheduler reads these class attributes directly. Clearing only
        # auth.json would leave an expired bearer token active until restart.
        self.config.USER_AUTH_TOKEN = ''
        self.config.BRAIN_AUTH_TOKEN = ''

        # Remove auth file
        try:
            auth_file = os.path.join(self.config.CONFIG_DIR, 'auth.json')
            legacy_auth_file = os.path.join(getattr(self.config, 'LEGACY_CONFIG_DIR', os.path.join(self.config.DATA_DIR, 'config')), 'auth.json')
            if os.path.exists(auth_file):
                os.remove(auth_file)
            if legacy_auth_file != auth_file and os.path.exists(legacy_auth_file):
                os.remove(legacy_auth_file)
            logger.info("Cleared authentication data")
        except Exception as e:
            logger.error(f"Error clearing auth data: {e}")

    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently authenticated user.

        Returns:
            Dict with user information or None if not authenticated
        """
        if not self.is_authenticated():
            return None

        return self.user_info
