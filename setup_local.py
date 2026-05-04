#!/usr/bin/env python3
"""
Thoth Local Setup Script

This script helps set up Thoth for local use without requiring a Brain server connection.
It creates a default local user and configures the system for standalone operation.
"""

import os
import json
import hashlib
from datetime import datetime, timedelta

# Add thoth_core to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from thoth_core.backend.config import Config

def setup_local_mode():
    """Set up Thoth for local mode with default credentials."""
    
    print("Thoth Local Setup")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.CONFIG_DIR, exist_ok=True)
    os.makedirs(os.path.join(Config.DATA_DIR, 'models'), exist_ok=True)
    
    # Create local auth file
    auth_file = os.path.join(Config.CONFIG_DIR, 'auth.json')
    
    # Default credentials
    default_username = "admin"
    default_password = "thoth123"  # Change this in production!
    
    # Create a simple token (in production, use JWT)
    token_data = {
        "username": default_username,
        "created_at": datetime.utcnow().isoformat(),
        "expires_at": (datetime.utcnow() + timedelta(days=365)).isoformat()
    }
    simple_token = hashlib.md5(json.dumps(token_data, sort_keys=True).encode()).hexdigest()
    
    auth_data = {
        "token": simple_token,
        "refresh_token": simple_token,
        "token_expiry": token_data["expires_at"],
        "user_info": {
            "user_id": "local-user",
            "username": default_username,
            "email": "local@thoth.local"
        }
    }
    
    with open(auth_file, 'w') as f:
        json.dump(auth_data, f, indent=2)
    
    print(f"✓ Created local authentication file")
    print(f"✓ Default username: {default_username}")
    print(f"✓ Default password: {default_password}")
    print()
    
    # Create device config
    device_config_file = os.path.join(Config.CONFIG_DIR, 'device_config.json')
    device_config = {
        "device_id": "local-device-001",
        "device_name": "Thoth Local Device",
        "brain_server_url": "http://localhost:8000",  # Local mode
        "local_mode": True
    }
    
    with open(device_config_file, 'w') as f:
        json.dump(device_config, f, indent=2)
    
    print(f"✓ Created device configuration")
    print()
    
    # Update .env to ensure local mode
    env_file = os.path.join(os.path.dirname(Config.BASE_DIR), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            env_content = f.read()
        
        # Add local mode setting if not present
        if 'LOCAL_MODE' not in env_content:
            with open(env_file, 'a') as f:
                f.write('\n# Local mode settings\nLOCAL_MODE=True\n')
    
    print("✓ Configuration complete!")
    print()
    print("You can now:")
    print("1. Start Thoth: python -m thoth_core.backend.app")
    print("2. Open http://localhost:8000")
    print(f"3. Login with username: {default_username}")
    print(f"4. Password: {default_password}")
    print()
    print("To uninstall, run:")
    print("  - From system tray: Right-click → Uninstall Thoth")
    print("  - From PowerShell: .\\thoth\\thoth_win\\uninstall.ps1")
    print("  - From Settings: Apps & features → Uninstall Thoth")

if __name__ == "__main__":
    setup_local_mode()
