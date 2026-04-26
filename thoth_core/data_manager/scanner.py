"""Data Scanner for discovering and indexing data files.

This module scans the data directory for existing data files and
creates/updates metadata files as needed.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .protocol import (
    DataFileMetadata,
    CollectionInfo,
    QualityInfo,
    get_metadata_filename,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)


class DataScanner:
    """Scans and indexes data files in the data directory."""
    
    # Supported data file extensions
    DATA_EXTENSIONS = {'.csv', '.json', '.npy', '.npz', '.wav', '.mp4', '.jpg', '.png', '.pkl'}
    
    # Sensor type inference from filename prefix
    SENSOR_PREFIXES = {
        'csi': 'csi',
        'imu': 'imu',
        'cam': 'camera',
        'mic': 'mic',
        'radar': 'radar',
        'gps': 'gps',
    }
    
    def __init__(self, data_dir: str = None, device_id: str = None):
        """Initialize the scanner.
        
        Args:
            data_dir: Path to data directory
            device_id: Device identifier
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        self.device_id = device_id or self._load_device_id()
    
    def _load_device_id(self) -> str:
        """Load device ID from file."""
        device_id_file = self.data_dir / "device_id.txt"
        if device_id_file.exists():
            try:
                return device_id_file.read_text().strip()
            except Exception:
                pass
        return "unknown"
    
    def scan(self, create_missing_metadata: bool = True) -> Dict[str, Any]:
        """Scan the data directory for data files.
        
        Args:
            create_missing_metadata: Whether to create metadata for files without it
        
        Returns:
            Scan results dictionary
        """
        results = {
            "scanned_at": datetime.utcnow().isoformat() + "Z",
            "data_dir": str(self.data_dir),
            "total_files": 0,
            "with_metadata": 0,
            "without_metadata": 0,
            "metadata_created": 0,
            "errors": [],
            "files": [],
        }
        
        if not self.data_dir.exists():
            results["errors"].append(f"Data directory does not exist: {self.data_dir}")
            return results
        
        # Find all data files
        for path in self.data_dir.iterdir():
            if not path.is_file():
                continue
            
            # Skip metadata files
            if path.suffix == '.json' and path.stem.endswith('.meta'):
                continue
            
            # Skip non-data files
            if path.suffix.lower() not in self.DATA_EXTENSIONS:
                continue
            
            results["total_files"] += 1
            
            # Check for metadata
            meta_path = self.data_dir / get_metadata_filename(path.name)
            has_metadata = meta_path.exists()
            
            if has_metadata:
                results["with_metadata"] += 1
                metadata = DataFileMetadata.load(meta_path)
            else:
                results["without_metadata"] += 1
                
                if create_missing_metadata:
                    try:
                        metadata = self._create_metadata_for_file(path)
                        metadata.save(meta_path)
                        results["metadata_created"] += 1
                        logger.info(f"Created metadata for {path.name}")
                    except Exception as e:
                        results["errors"].append(f"Failed to create metadata for {path.name}: {e}")
                        metadata = None
                else:
                    metadata = None
            
            results["files"].append({
                "filename": path.name,
                "has_metadata": has_metadata or (create_missing_metadata and metadata is not None),
                "size_bytes": path.stat().st_size,
                "sensor_type": self._infer_sensor_type(path.name),
            })
        
        logger.info(f"Scan complete: {results['total_files']} files, "
                   f"{results['with_metadata']} with metadata, "
                   f"{results['metadata_created']} created")
        
        return results
    
    def _create_metadata_for_file(self, path: Path) -> DataFileMetadata:
        """Create metadata for an existing data file.
        
        Args:
            path: Path to data file
        
        Returns:
            DataFileMetadata instance
        """
        sensor_type = self._infer_sensor_type(path.name)
        file_format = path.suffix.lstrip('.')
        
        # Try to get sample count
        num_samples = self._count_samples(path)
        
        # Try to infer timestamp from filename
        created_at = self._infer_timestamp(path.name)
        if created_at is None:
            created_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat() + "Z"
        
        return DataFileMetadata(
            filename=path.name,
            sensor_type=sensor_type,
            data_type=self._infer_data_type(sensor_type),
            file_format=file_format,
            created_at=created_at,
            device_id=self.device_id,
            collection=CollectionInfo(
                num_samples=num_samples,
            ),
            quality=QualityInfo(
                file_size_bytes=path.stat().st_size,
                validated=False,
            ),
        )
    
    def _infer_sensor_type(self, filename: str) -> str:
        """Infer sensor type from filename."""
        name_lower = filename.lower()
        for prefix, sensor_type in self.SENSOR_PREFIXES.items():
            if name_lower.startswith(prefix):
                return sensor_type
        return "custom"
    
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
    
    def _infer_timestamp(self, filename: str) -> Optional[str]:
        """Try to infer timestamp from filename."""
        import re
        
        # Pattern: YYYY-MM-DDTHH-MM-SS or YYYY-MM-DD
        patterns = [
            r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})',
            r'(\d{4}-\d{2}-\d{2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                ts_str = match.group(1)
                try:
                    if 'T' in ts_str:
                        dt = datetime.strptime(ts_str, "%Y-%m-%dT%H-%M-%S")
                    else:
                        dt = datetime.strptime(ts_str, "%Y-%m-%d")
                    return dt.isoformat() + "Z"
                except ValueError:
                    pass
        
        return None
    
    def _count_samples(self, path: Path) -> int:
        """Count samples in a data file."""
        try:
            ext = path.suffix.lower()
            
            if ext == '.npy':
                data = np.load(path)
                return data.shape[0] if data.ndim >= 1 else 1
            
            elif ext == '.npz':
                data = np.load(path)
                arr = data[data.files[0]]
                return arr.shape[0] if arr.ndim >= 1 else 1
            
            elif ext == '.csv':
                with open(path, 'r') as f:
                    return sum(1 for _ in f)
            
            elif ext == '.json':
                with open(path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return len(data)
                return 1
            
        except Exception as e:
            logger.debug(f"Could not count samples in {path}: {e}")
        
        return 0
    
    def validate_all(self) -> Dict[str, Any]:
        """Validate all data files and their metadata.
        
        Returns:
            Validation results
        """
        results = {
            "validated_at": datetime.utcnow().isoformat() + "Z",
            "total_files": 0,
            "valid": 0,
            "invalid": 0,
            "issues": [],
        }
        
        for meta_path in self.data_dir.glob("*.meta.json"):
            results["total_files"] += 1
            
            metadata = DataFileMetadata.load(meta_path)
            if metadata is None:
                results["invalid"] += 1
                results["issues"].append({
                    "file": meta_path.name,
                    "error": "Failed to load metadata",
                })
                continue
            
            # Check data file exists
            data_path = self.data_dir / metadata.filename
            if not data_path.exists():
                results["invalid"] += 1
                results["issues"].append({
                    "file": metadata.filename,
                    "error": "Data file not found",
                })
                continue
            
            # Validate metadata
            errors = metadata.validate()
            if errors:
                results["invalid"] += 1
                results["issues"].append({
                    "file": metadata.filename,
                    "errors": errors,
                })
            else:
                results["valid"] += 1
        
        return results
