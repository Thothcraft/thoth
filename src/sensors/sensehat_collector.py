"""Sense HAT sensor data collection module.

This module handles continuous collection of IMU data from the Sense HAT,
including accelerometer, gyroscope, and magnetometer readings.
"""

import os
import sys
import json
import time
import threading
import logging
from datetime import datetime
from typing import Optional

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sense_hat import SenseHat
    SENSE_HAT_AVAILABLE = True
except ImportError:
    print("Sense HAT library not available. Using mock data.")
    SENSE_HAT_AVAILABLE = False

from backend.config import Config, SENSOR_CONFIG
from backend.models import SensorReading, IMUReading, AccelerometerReading, GyroscopeReading, MagnetometerReading

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MockSenseHat:
    """Mock Sense HAT for testing without hardware."""
    
    def __init__(self):
        self.time_offset = time.time()
    
    def set_imu_config(self, compass=True, gyro=True, accel=True):
        """Mock IMU configuration."""
        pass
    
    def get_orientation(self):
        """Mock orientation data with sine wave patterns."""
        t = time.time() - self.time_offset
        return {
            'pitch': 10 * math.sin(t * 0.5),
            'roll': 5 * math.cos(t * 0.3),
            'yaw': 15 * math.sin(t * 0.2)
        }
    
    def get_accelerometer_raw(self):
        """Mock accelerometer data."""
        t = time.time() - self.time_offset
        return {
            'x': 0.1 * math.sin(t),
            'y': 0.1 * math.cos(t),
            'z': 1.0 + 0.05 * math.sin(t * 2)
        }
    
    def get_gyroscope_raw(self):
        """Mock gyroscope data."""
        t = time.time() - self.time_offset
        return {
            'x': 0.01 * math.cos(t * 3),
            'y': 0.01 * math.sin(t * 2),
            'z': 0.005 * math.sin(t)
        }
    
    def get_compass_raw(self):
        """Mock magnetometer data."""
        t = time.time() - self.time_offset
        return {
            'x': 20 + 5 * math.sin(t * 0.1),
            'y': -10 + 3 * math.cos(t * 0.15),
            'z': 45 + 2 * math.sin(t * 0.2)
        }

class SensorCollector:
    """Handles sensor data collection from Sense HAT."""
    
    def __init__(self, data_file: str = None):
        self.data_file = data_file or Config.SENSOR_DATA_FILE
        self.collection_rate = SENSOR_CONFIG['sample_rate']
        self.running = False
        self.collection_thread = None
        
        # Initialize Sense HAT
        if SENSE_HAT_AVAILABLE:
            try:
                self.sense = SenseHat()
                logger.info("Sense HAT initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Sense HAT: {e}")
                self.sense = MockSenseHat()
        else:
            import math  # Import here for mock
            self.sense = MockSenseHat()
            logger.info("Using mock Sense HAT data")
        
        # Configure IMU
        self.configure_imu()
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
    
    def configure_imu(self):
        """Configure the IMU sensors."""
        try:
            self.sense.set_imu_config(
                compass=SENSOR_CONFIG['compass_enabled'],
                gyro=SENSOR_CONFIG['gyro_enabled'],
                accel=SENSOR_CONFIG['accel_enabled']
            )
            logger.info("IMU configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure IMU: {e}")
    
    def collect_reading(self) -> Optional[SensorReading]:
        """Collect a single sensor reading."""
        try:
            # Get sensor data
            orientation = self.sense.get_orientation()
            accel = self.sense.get_accelerometer_raw()
            gyro = self.sense.get_gyroscope_raw()
            mag = self.sense.get_compass_raw()
            
            # Create reading object
            reading = SensorReading(
                timestamp=datetime.now().isoformat(),
                imu=IMUReading(
                    pitch=float(orientation['pitch']),
                    roll=float(orientation['roll']),
                    yaw=float(orientation['yaw'])
                ),
                accel=AccelerometerReading(
                    x=float(accel['x']),
                    y=float(accel['y']),
                    z=float(accel['z'])
                ),
                gyro=GyroscopeReading(
                    x=float(gyro['x']),
                    y=float(gyro['y']),
                    z=float(gyro['z'])
                ),
                mag=MagnetometerReading(
                    x=float(mag['x']),
                    y=float(mag['y']),
                    z=float(mag['z'])
                )
            )
            
            return reading
            
        except Exception as e:
            logger.error(f"Error collecting sensor reading: {e}")
            return None
    
    def save_reading(self, reading: SensorReading):
        """Save a reading to the data file."""
        try:
            with open(self.data_file, 'a') as f:
                f.write(reading.to_json() + '\n')
        except Exception as e:
            logger.error(f"Error saving reading: {e}")
    
    def collection_loop(self):
        """Main collection loop."""
        logger.info(f"Starting sensor collection at {self.collection_rate} Hz")
        
        while self.running:
            start_time = time.time()
            
            # Collect and save reading
            reading = self.collect_reading()
            if reading:
                self.save_reading(reading)
                logger.debug(f"Collected reading: pitch={reading.imu.pitch:.2f}°")
            
            # Calculate sleep time to maintain sample rate
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0 / self.collection_rate) - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def start_collection(self):
        """Start sensor data collection."""
        if self.running:
            logger.warning("Collection already running")
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self.collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Sensor collection started")
    
    def stop_collection(self):
        """Stop sensor data collection."""
        if not self.running:
            logger.warning("Collection not running")
            return
        
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Sensor collection stopped")
    
    def is_collecting(self) -> bool:
        """Check if collection is active."""
        return self.running and self.collection_thread and self.collection_thread.is_alive()

def main():
    """Main function for running as a service."""
    logger.info("Starting Thoth sensor collector service")
    
    collector = SensorCollector()
    
    try:
        collector.start_collection()
        
        # Keep the service running
        while True:
            if not collector.is_collecting():
                logger.error("Collection stopped unexpectedly, restarting...")
                collector.start_collection()
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        collector.stop_collection()
        logger.info("Sensor collector service stopped")

if __name__ == '__main__':
    main()
