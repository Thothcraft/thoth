"""
Authentication Manager for Thoth Device

This module handles authentication with the Brain server, including token management
and session handling for the Thoth device.
"""

import os
import json
import logging
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

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
        
        # Load saved auth data if available
        self._load_auth_data()
    
    def _load_auth_data(self) -> None:
        """Load authentication data from disk."""
        try:
            auth_file = os.path.join(self.config.CONFIG_DIR, 'auth.json')
            
            if os.path.exists(auth_file):
                with open(auth_file, 'r') as f:
                    auth_data = json.load(f)
                    
                    self.token = auth_data.get('token')
                    self.refresh_token = auth_data.get('refresh_token')
                    self.token_expiry = datetime.fromisoformat(auth_data.get('token_expiry')) \
                        if auth_data.get('token_expiry') else None
                    self.user_info = auth_data.get('user_info')
                    
                    # Set token on Config for device registration
                    if self.token:
                        self.config.USER_AUTH_TOKEN = self.token
                        logger.info("Loaded authentication data from disk and set USER_AUTH_TOKEN")
                    else:
                        logger.info("Loaded authentication data from disk (no token)")
                    
        except Exception as e:
            logger.error(f"Error loading auth data: {e}")
    
    def _save_auth_data(self) -> None:
        """Save authentication data to disk."""
        try:
            os.makedirs(self.config.CONFIG_DIR, exist_ok=True)
            auth_file = os.path.join(self.config.CONFIG_DIR, 'auth.json')
            
            auth_data = {
                'token': self.token,
                'refresh_token': self.refresh_token,
                'token_expiry': self.token_expiry.isoformat() if self.token_expiry else None,
                'user_info': self.user_info
            }
            
            with open(auth_file, 'w') as f:
                json.dump(auth_data, f, indent=2)
                
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
            # Try to refresh the token if we have a refresh token
            if self.refresh_token:
                return self.refresh_auth_token()
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
        from requests.exceptions import RequestException
        
        if not self.config.BRAIN_SERVER_URL:
            raise Exception("Brain server URL not configured")
            
        url = f"{self.config.BRAIN_SERVER_URL}/token"
        
        try:
            # Send login request
            response = requests.post(
                url,
                json={
                    'username': username,
                    'password': password
                },
                headers={
                    'accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Parse token to get expiry
                token_data = jwt.decode(
                    result['access_token'],
                    options={"verify_signature": False},
                    algorithms=["HS256"]
                )
                
                # Update auth state
                self.token = result['access_token']
                self.refresh_token = result.get('refresh_token')
                self.token_expiry = datetime.utcfromtimestamp(token_data['exp'])
                self.user_info = {
                    'username': token_data.get('sub'),
                    'user_id': token_data.get('user_id'),
                    'email': token_data.get('email'),
                    'scopes': token_data.get('scopes', [])
                }
                
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
                error_msg = f"Login failed: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        except RequestException as e:
            error_msg = f"Error connecting to Brain server: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Error during login: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)
    
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
                token_data = jwt.decode(
                    result['access_token'],
                    options={"verify_signature": False},
                    algorithms=["HS256"]
                )
                
                # Update auth state
                self.token = result['access_token']
                self.token_expiry = datetime.utcfromtimestamp(token_data['exp'])
                
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
        
        # Remove auth file
        try:
            auth_file = os.path.join(self.config.CONFIG_DIR, 'auth.json')
            if os.path.exists(auth_file):
                os.remove(auth_file)
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
