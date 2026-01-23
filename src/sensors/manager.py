"""Sensor Manager for coordinating multiple sensors.

This module provides a unified interface for managing all sensors
on the Thoth device, including discovery, collection, and data saving.
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from .base import (
    BaseSensor,
    SensorRegistry,
    SensorType,
    SensorStatus,
    SensorConfig,
    CollectionSession,
)
from ..data_manager import DataManager, DataFileMetadata

logger = logging.getLogger(__name__)


class SensorManager:
    """Manages all sensors on the Thoth device.
    
    Provides:
    - Sensor discovery and initialization
    - Coordinated data collection
    - Automatic data saving with metadata
    - Integration with DataManager
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the sensor manager.
        
        Args:
            data_dir: Path to data directory
        """
        self._sensors: Dict[SensorType, BaseSensor] = {}
        self._active_sessions: Dict[str, CollectionSession] = {}
        self._lock = threading.Lock()
        
        # Initialize data manager
        self.data_manager = DataManager(data_dir=data_dir)
        
        logger.info("SensorManager initialized")
    
    def discover_sensors(self) -> List[Dict[str, Any]]:
        """Discover available sensors on this device.
        
        Returns:
            List of sensor info dictionaries
        """
        available = SensorRegistry.discover_available()
        
        for sensor in available:
            self._sensors[sensor.sensor_type] = sensor
        
        return [s.get_info() for s in available]
    
    def get_sensor(self, sensor_type: SensorType) -> Optional[BaseSensor]:
        """Get a sensor by type.
        
        Args:
            sensor_type: Type of sensor
        
        Returns:
            Sensor instance or None
        """
        if sensor_type not in self._sensors:
            # Try to create and initialize
            sensor = SensorRegistry.create(sensor_type)
            if sensor and sensor.initialize():
                self._sensors[sensor_type] = sensor
        
        return self._sensors.get(sensor_type)
    
    def list_sensors(self) -> List[Dict[str, Any]]:
        """List all initialized sensors.
        
        Returns:
            List of sensor info dictionaries
        """
        return [s.get_info() for s in self._sensors.values()]
    
    def start_collection(
        self,
        sensor_type: str,
        duration_seconds: Optional[float] = None,
        labels: Optional[Dict[str, Any]] = None,
        auto_save: bool = True,
    ) -> Optional[CollectionSession]:
        """Start data collection from a sensor.
        
        Args:
            sensor_type: Type of sensor (string or SensorType)
            duration_seconds: Collection duration (None for continuous)
            labels: Labels for the collected data
            auto_save: Whether to automatically save data when collection stops
        
        Returns:
            CollectionSession or None if failed
        """
        # Convert string to SensorType
        if isinstance(sensor_type, str):
            try:
                sensor_type = SensorType(sensor_type)
            except ValueError:
                logger.error(f"Unknown sensor type: {sensor_type}")
                return None
        
        sensor = self.get_sensor(sensor_type)
        if not sensor:
            logger.error(f"Sensor not available: {sensor_type}")
            return None
        
        if sensor.status == SensorStatus.COLLECTING:
            logger.warning(f"Sensor {sensor_type} already collecting")
            return None
        
        # Start collection
        session = sensor.start_collection(
            duration_seconds=duration_seconds,
            labels=labels,
        )
        
        with self._lock:
            self._active_sessions[session.session_id] = session
        
        # If duration specified, schedule auto-stop and save
        if duration_seconds and auto_save:
            def auto_stop():
                import time
                time.sleep(duration_seconds + 0.5)  # Small buffer
                self.stop_collection(session.session_id, save=True)
            
            thread = threading.Thread(target=auto_stop, daemon=True)
            thread.start()
        
        return session
    
    def stop_collection(
        self,
        session_id: str,
        save: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Stop a collection session.
        
        Args:
            session_id: Session ID to stop
            save: Whether to save the collected data
        
        Returns:
            Result dictionary with session info and saved file
        """
        with self._lock:
            session = self._active_sessions.get(session_id)
            if not session:
                logger.warning(f"Session not found: {session_id}")
                return None
        
        # Find the sensor
        sensor = self._sensors.get(session.sensor_type)
        if not sensor:
            return None
        
        # Stop collection
        completed_session = sensor.stop_collection()
        if not completed_session:
            return None
        
        result = {
            "session": completed_session.to_dict(),
            "saved": False,
            "filename": None,
        }
        
        # Save data if requested
        if save:
            data = sensor.get_buffer()
            if len(data) > 0:
                try:
                    data_path, metadata = self.data_manager.save_data(
                        data=data,
                        sensor_type=completed_session.sensor_type.value,
                        labels=completed_session.labels,
                        collection_info={
                            "duration_seconds": completed_session.duration_seconds,
                            "sample_rate_hz": sensor.config.sample_rate_hz,
                            "num_samples": completed_session.num_samples,
                            "start_time": completed_session.start_time.isoformat(),
                            "end_time": completed_session.end_time.isoformat() if completed_session.end_time else None,
                        },
                    )
                    result["saved"] = True
                    result["filename"] = data_path.name
                    completed_session.data_file = data_path.name
                    logger.info(f"Saved {len(data)} samples to {data_path.name}")
                except Exception as e:
                    logger.error(f"Failed to save data: {e}")
                    result["error"] = str(e)
        
        # Clear buffer
        sensor.clear_buffer()
        
        # Remove from active sessions
        with self._lock:
            self._active_sessions.pop(session_id, None)
        
        return result
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all active collection sessions.
        
        Returns:
            List of session dictionaries
        """
        with self._lock:
            return [s.to_dict() for s in self._active_sessions.values()]
    
    def stop_all(self, save: bool = True) -> List[Dict[str, Any]]:
        """Stop all active collection sessions.
        
        Args:
            save: Whether to save collected data
        
        Returns:
            List of results for each stopped session
        """
        results = []
        
        with self._lock:
            session_ids = list(self._active_sessions.keys())
        
        for session_id in session_ids:
            result = self.stop_collection(session_id, save=save)
            if result:
                results.append(result)
        
        return results
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected data.
        
        Returns:
            Statistics dictionary
        """
        return self.data_manager.get_statistics()
    
    def cleanup(self):
        """Cleanup all sensors."""
        # Stop all active sessions
        self.stop_all(save=True)
        
        # Cleanup sensors
        for sensor in self._sensors.values():
            try:
                sensor.cleanup()
            except Exception as e:
                logger.warning(f"Sensor cleanup error: {e}")
        
        self._sensors.clear()
        logger.info("SensorManager cleanup complete")
