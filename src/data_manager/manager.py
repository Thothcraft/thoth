"""Data Manager for Thoth Device.

This module provides the main interface for managing data files and metadata
on the Thoth device, including saving, loading, and querying data for
ML/DL and Federated Learning training.
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterator
import numpy as np

from .protocol import (
    DataFileMetadata,
    CollectionInfo,
    LabelInfo,
    PreprocessingInfo,
    FLStatus,
    QualityInfo,
    SensorType,
    DataType,
    FileFormat,
    generate_filename,
    get_metadata_filename,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)


class DataManager:
    """Manages data files and metadata on the Thoth device.
    
    This class provides:
    - Saving data with automatic metadata generation
    - Loading data and metadata
    - Querying data by labels, sensor type, FL status
    - Preparing data for FL training
    """
    
    def __init__(self, data_dir: str = None, device_id: str = None):
        """Initialize the data manager.
        
        Args:
            data_dir: Path to data directory (defaults to thoth/data/)
            device_id: Device identifier (loaded from config if not provided)
        """
        if data_dir is None:
            # Default to thoth/data/ relative to this file
            self.data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        self.device_id = device_id or self._load_device_id()
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataManager initialized: data_dir={self.data_dir}, device_id={self.device_id}")
    
    def _load_device_id(self) -> str:
        """Load device ID from config file."""
        device_id_file = self.data_dir / "device_id.txt"
        if device_id_file.exists():
            try:
                return device_id_file.read_text().strip()
            except Exception as e:
                logger.warning(f"Failed to read device_id: {e}")
        
        # Generate new device ID
        import uuid
        device_id = f"thoth-{str(uuid.uuid4())[:8]}"
        try:
            device_id_file.write_text(device_id)
        except Exception as e:
            logger.warning(f"Failed to save device_id: {e}")
        
        return device_id
    
    def save_data(
        self,
        data: np.ndarray,
        sensor_type: str,
        labels: Optional[Dict[str, Any]] = None,
        collection_info: Optional[Dict[str, Any]] = None,
        file_format: str = "npy",
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, DataFileMetadata]:
        """Save data with automatically generated metadata.
        
        Args:
            data: NumPy array of data
            sensor_type: Type of sensor (csi, imu, etc.)
            labels: Label information dictionary
            collection_info: Collection information dictionary
            file_format: File format (npy, csv, json)
            custom_metadata: Additional custom metadata
        
        Returns:
            Tuple of (data_path, metadata)
        """
        # Generate filename
        filename = generate_filename(sensor_type, file_format)
        data_path = self.data_dir / filename
        
        # Save data based on format
        if file_format == "npy":
            np.save(data_path, data)
        elif file_format == "npz":
            np.savez_compressed(data_path, data=data)
        elif file_format == "csv":
            if data.ndim == 1:
                np.savetxt(data_path, data, delimiter=",")
            else:
                np.savetxt(data_path, data, delimiter=",")
        elif file_format == "json":
            with open(data_path, 'w') as f:
                json.dump(data.tolist(), f)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        # Create metadata
        metadata = DataFileMetadata(
            filename=filename,
            sensor_type=sensor_type,
            data_type=self._infer_data_type(sensor_type),
            file_format=file_format,
            device_id=self.device_id,
            collection=CollectionInfo(
                num_samples=data.shape[0] if data.ndim >= 1 else 1,
                **(collection_info or {})
            ),
            labels=LabelInfo(**(labels or {})),
            quality=QualityInfo(
                completeness=1.0,
                validated=True,
                file_size_bytes=data_path.stat().st_size,
                checksum=self._compute_checksum(data_path),
            ),
            custom_metadata=custom_metadata or {},
        )
        
        # Save metadata
        meta_path = self.data_dir / get_metadata_filename(filename)
        metadata.save(meta_path)
        
        logger.info(f"Saved data: {filename} ({data.shape})")
        return data_path, metadata
    
    def load_data(self, filename: str) -> Tuple[Optional[np.ndarray], Optional[DataFileMetadata]]:
        """Load data and metadata.
        
        Args:
            filename: Data filename
        
        Returns:
            Tuple of (data, metadata) or (None, None) if not found
        """
        data_path = self.data_dir / filename
        meta_path = self.data_dir / get_metadata_filename(filename)
        
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return None, None
        
        # Load metadata
        metadata = None
        if meta_path.exists():
            metadata = DataFileMetadata.load(meta_path)
        
        # Load data based on format
        file_format = data_path.suffix.lstrip('.')
        try:
            if file_format == "npy":
                data = np.load(data_path)
            elif file_format == "npz":
                loaded = np.load(data_path)
                data = loaded['data'] if 'data' in loaded else loaded[loaded.files[0]]
            elif file_format == "csv":
                data = np.loadtxt(data_path, delimiter=",")
            elif file_format == "json":
                with open(data_path, 'r') as f:
                    data = np.array(json.load(f))
            else:
                logger.warning(f"Unknown format {file_format}, attempting numpy load")
                data = np.load(data_path, allow_pickle=True)
            
            return data, metadata
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {e}")
            return None, metadata
    
    def list_files(
        self,
        sensor_type: Optional[str] = None,
        activity: Optional[str] = None,
        available_for_fl: Optional[bool] = None,
    ) -> List[DataFileMetadata]:
        """List data files with optional filtering.
        
        Args:
            sensor_type: Filter by sensor type
            activity: Filter by activity label
            available_for_fl: Filter by FL availability
        
        Returns:
            List of metadata for matching files
        """
        results = []
        
        for meta_path in self.data_dir.glob("*.meta.json"):
            metadata = DataFileMetadata.load(meta_path)
            if metadata is None:
                continue
            
            # Apply filters
            if sensor_type and metadata.sensor_type != sensor_type:
                continue
            if activity and metadata.labels.activity != activity:
                continue
            if available_for_fl is not None and metadata.fl_status.available_for_training != available_for_fl:
                continue
            
            results.append(metadata)
        
        return results
    
    def get_fl_dataset(
        self,
        sensor_type: Optional[str] = None,
        activity: Optional[str] = None,
        max_files: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, DataFileMetadata]]:
        """Get data files available for FL training.
        
        Args:
            sensor_type: Filter by sensor type
            activity: Filter by activity label
            max_files: Maximum number of files to return
        
        Returns:
            List of (data, metadata) tuples
        """
        files = self.list_files(
            sensor_type=sensor_type,
            activity=activity,
            available_for_fl=True,
        )
        
        if max_files:
            files = files[:max_files]
        
        results = []
        for metadata in files:
            data, _ = self.load_data(metadata.filename)
            if data is not None:
                results.append((data, metadata))
        
        return results
    
    def mark_used_in_fl(self, filename: str, round_num: int) -> bool:
        """Mark a data file as used in an FL round.
        
        Args:
            filename: Data filename
            round_num: FL round number
        
        Returns:
            True if successful
        """
        meta_path = self.data_dir / get_metadata_filename(filename)
        metadata = DataFileMetadata.load(meta_path)
        
        if metadata is None:
            return False
        
        metadata.fl_status.mark_used(round_num)
        return metadata.save(meta_path)
    
    def update_labels(
        self,
        filename: str,
        labels: Dict[str, Any],
    ) -> bool:
        """Update labels for a data file.
        
        Args:
            filename: Data filename
            labels: New label values
        
        Returns:
            True if successful
        """
        meta_path = self.data_dir / get_metadata_filename(filename)
        metadata = DataFileMetadata.load(meta_path)
        
        if metadata is None:
            return False
        
        # Update labels
        for key, value in labels.items():
            if hasattr(metadata.labels, key):
                setattr(metadata.labels, key, value)
            else:
                metadata.labels.custom_labels[key] = value
        
        return metadata.save(meta_path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored data.
        
        Returns:
            Dictionary with statistics
        """
        files = self.list_files()
        
        stats = {
            "total_files": len(files),
            "total_size_bytes": sum(f.quality.file_size_bytes for f in files),
            "by_sensor_type": {},
            "by_activity": {},
            "fl_available": 0,
            "fl_used": 0,
        }
        
        for f in files:
            # By sensor type
            st = f.sensor_type
            if st not in stats["by_sensor_type"]:
                stats["by_sensor_type"][st] = 0
            stats["by_sensor_type"][st] += 1
            
            # By activity
            act = f.labels.activity or "unlabeled"
            if act not in stats["by_activity"]:
                stats["by_activity"][act] = 0
            stats["by_activity"][act] += 1
            
            # FL status
            if f.fl_status.available_for_training:
                stats["fl_available"] += 1
            if f.fl_status.used_in_rounds:
                stats["fl_used"] += 1
        
        return stats
    
    def _infer_data_type(self, sensor_type: str) -> str:
        """Infer data type from sensor type."""
        mapping = {
            "csi": "wifi_csi",
            "imu": "motion_imu",
            "camera": "video",
            "mic": "audio",
            "radar": "radar_point_cloud",
        }
        return mapping.get(sensor_type, "time_series")
    
    def _compute_checksum(self, path: Path) -> str:
        """Compute MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
