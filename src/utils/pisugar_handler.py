"""PiSugar button and power management module.

This module handles PiSugar button events and power management,
including configurable button actions and battery monitoring.
"""

import os
import sys
import time
import json
import logging
import subprocess
import threading
from typing import Dict, Callable, Optional

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import Config, BUTTON_ACTIONS

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from sugarpie import PiSugar
    PISUGAR_AVAILABLE = True
except ImportError:
    logger.warning("PiSugar library not available. Using mock implementation.")
    PISUGAR_AVAILABLE = False

class MockPiSugar:
    """Mock PiSugar for testing without hardware."""
    
    def __init__(self):
        self.battery_level = 85
        self.charging = False
        self.button_pressed = False
        self.last_button_time = 0
    
    def get_battery_level(self) -> int:
        """Mock battery level."""
        return self.battery_level
    
    def get_battery_voltage(self) -> float:
        """Mock battery voltage."""
        return 3.7 + (self.battery_level / 100) * 0.5
    
    def is_charging(self) -> bool:
        """Mock charging status."""
        return self.charging
    
    def get_button_event(self) -> Optional[str]:
        """Mock button event (returns None most of the time)."""
        # Simulate occasional button press for testing
        current_time = time.time()
        if current_time - self.last_button_time > 30:  # Every 30 seconds
            self.last_button_time = current_time
            return "single"
        return None
    
    def safe_shutdown(self):
        """Mock safe shutdown."""
        logger.info("Mock PiSugar safe shutdown requested")

class PiSugarHandler:
    """Handles PiSugar button events and power management."""
    
    def __init__(self):
        # Initialize PiSugar
        if PISUGAR_AVAILABLE:
            try:
                self.pisugar = PiSugar()
                logger.info("PiSugar initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize PiSugar: {e}")
                self.pisugar = MockPiSugar()
        else:
            self.pisugar = MockPiSugar()
        
        # Button state tracking
        self.last_button_time = 0
        self.button_press_count = 0
        self.long_press_threshold = 2.0  # seconds
        self.double_tap_window = 0.5  # seconds
        
        # Load button configuration
        self.button_actions = BUTTON_ACTIONS.copy()
        self.load_button_config()
        
        # Action handlers
        self.action_handlers = {
            "toggle_collection": self.toggle_collection,
            "start_collection": self.start_collection,
            "stop_collection": self.stop_collection,
            "start_ap": self.start_ap_mode,
            "stop_ap": self.stop_ap_mode,
            "shutdown": self.safe_shutdown,
            "reboot": self.reboot_system,
            "status": self.show_status
        }
        
        # Monitoring thread
        self.monitoring = False
        self.monitor_thread = None
    
    def load_button_config(self):
        """Load button configuration from file."""
        config_file = os.path.join(os.path.dirname(__file__), '..', 'backend', 'button_config.json')
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    saved_config = json.load(f)
                    self.button_actions.update(saved_config)
                    logger.info(f"Loaded button configuration: {self.button_actions}")
        except Exception as e:
            logger.error(f"Error loading button config: {e}")
    
    def save_button_config(self):
        """Save current button configuration to file."""
        config_file = os.path.join(os.path.dirname(__file__), '..', 'backend', 'button_config.json')
        try:
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(self.button_actions, f)
            logger.info("Button configuration saved")
        except Exception as e:
            logger.error(f"Error saving button config: {e}")
    
    def get_battery_status(self) -> Dict[str, any]:
        """Get current battery status."""
        try:
            return {
                "level": self.pisugar.get_battery_level(),
                "voltage": self.pisugar.get_battery_voltage(),
                "charging": self.pisugar.is_charging(),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error getting battery status: {e}")
            return {"level": None, "error": str(e)}
    
    def toggle_collection(self):
        """Toggle sensor data collection."""
        try:
            # Check if collector is running
            result = subprocess.run(
                ["systemctl", "is-active", "thoth-collector"],
                capture_output=True, text=True
            )
            
            if result.stdout.strip() == "active":
                self.stop_collection()
            else:
                self.start_collection()
                
        except Exception as e:
            logger.error(f"Error toggling collection: {e}")
    
    def start_collection(self):
        """Start sensor data collection."""
        try:
            subprocess.run(["sudo", "systemctl", "start", "thoth-collector"])
            logger.info("Data collection started via button")
        except Exception as e:
            logger.error(f"Error starting collection: {e}")
    
    def stop_collection(self):
        """Stop sensor data collection."""
        try:
            subprocess.run(["sudo", "systemctl", "stop", "thoth-collector"])
            logger.info("Data collection stopped via button")
        except Exception as e:
            logger.error(f"Error stopping collection: {e}")
    
    def start_ap_mode(self):
        """Start AP mode for WiFi configuration."""
        try:
            # Stop collection first
            self.stop_collection()
            
            # Start AP mode via WiFi manager
            subprocess.run([
                "python3", "/opt/thoth/src/network/wifi_manager.py", "--start-ap"
            ])
            logger.info("AP mode started via button")
        except Exception as e:
            logger.error(f"Error starting AP mode: {e}")
    
    def stop_ap_mode(self):
        """Stop AP mode."""
        try:
            subprocess.run([
                "python3", "/opt/thoth/src/network/wifi_manager.py", "--stop-ap"
            ])
            logger.info("AP mode stopped via button")
        except Exception as e:
            logger.error(f"Error stopping AP mode: {e}")
    
    def safe_shutdown(self):
        """Perform safe system shutdown."""
        try:
            logger.info("Safe shutdown initiated via button")
            # Stop all services gracefully
            subprocess.run(["sudo", "systemctl", "stop", "thoth-collector"])
            subprocess.run(["sudo", "systemctl", "stop", "thoth-backend"])
            
            # Use PiSugar safe shutdown if available
            if hasattr(self.pisugar, 'safe_shutdown'):
                self.pisugar.safe_shutdown()
            else:
                subprocess.run(["sudo", "shutdown", "-h", "now"])
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def reboot_system(self):
        """Reboot the system."""
        try:
            logger.info("System reboot initiated via button")
            subprocess.run(["sudo", "reboot"])
        except Exception as e:
            logger.error(f"Error during reboot: {e}")
    
    def show_status(self):
        """Show system status (could flash LEDs or log status)."""
        try:
            battery = self.get_battery_status()
            logger.info(f"System status - Battery: {battery['level']}%, Charging: {battery['charging']}")
            
            # Could add Sense HAT LED display here
            # self.display_status_on_leds(battery['level'])
            
        except Exception as e:
            logger.error(f"Error showing status: {e}")
    
    def handle_button_event(self, event_type: str):
        """Handle a button event based on configuration."""
        action = self.button_actions.get(event_type)
        if not action:
            logger.warning(f"No action configured for button event: {event_type}")
            return
        
        handler = self.action_handlers.get(action)
        if handler:
            logger.info(f"Executing button action: {event_type} -> {action}")
            try:
                handler()
            except Exception as e:
                logger.error(f"Error executing button action {action}: {e}")
        else:
            logger.error(f"Unknown action: {action}")
    
    def detect_button_pattern(self, press_time: float) -> Optional[str]:
        """Detect button press pattern (single, double, long)."""
        current_time = time.time()
        
        # Check for long press
        if press_time >= self.long_press_threshold:
            self.button_press_count = 0
            return "long"
        
        # Check for double tap
        if current_time - self.last_button_time <= self.double_tap_window:
            self.button_press_count += 1
            if self.button_press_count >= 2:
                self.button_press_count = 0
                return "double"
        else:
            self.button_press_count = 1
        
        self.last_button_time = current_time
        
        # Wait to see if it's a double tap
        time.sleep(self.double_tap_window + 0.1)
        
        if self.button_press_count == 1:
            self.button_press_count = 0
            return "single"
        
        return None
    
    def monitor_button(self):
        """Monitor button events in a loop."""
        logger.info("Starting button monitoring")
        
        while self.monitoring:
            try:
                # Check for button event
                event = self.pisugar.get_button_event()
                
                if event:
                    # For real PiSugar, we might get different event types
                    # For now, assume we get basic press events and detect patterns
                    pattern = self.detect_button_pattern(0.1)  # Short press by default
                    
                    if pattern:
                        self.handle_button_event(pattern)
                
                time.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                logger.error(f"Error in button monitoring: {e}")
                time.sleep(1)
    
    def start_monitoring(self):
        """Start button monitoring in a separate thread."""
        if self.monitoring:
            logger.warning("Button monitoring already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_button, daemon=True)
        self.monitor_thread.start()
        logger.info("Button monitoring started")
    
    def stop_monitoring(self):
        """Stop button monitoring."""
        if not self.monitoring:
            logger.warning("Button monitoring not running")
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Button monitoring stopped")
    
    def update_button_config(self, new_config: Dict[str, str]):
        """Update button configuration."""
        self.button_actions.update(new_config)
        self.save_button_config()
        logger.info(f"Button configuration updated: {self.button_actions}")

def main():
    """Main function for running as a service."""
    logger.info("Starting PiSugar button handler service")
    
    handler = PiSugarHandler()
    
    try:
        handler.start_monitoring()
        
        # Keep the service running
        while True:
            # Log battery status periodically
            battery = handler.get_battery_status()
            if battery.get('level') is not None:
                logger.debug(f"Battery: {battery['level']}%, Charging: {battery.get('charging', False)}")
            
            time.sleep(60)  # Log every minute
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        handler.stop_monitoring()
        logger.info("PiSugar handler service stopped")

if __name__ == '__main__':
    main()
