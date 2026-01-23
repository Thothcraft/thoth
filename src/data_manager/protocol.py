"""Data Storage Protocol Definitions.

This module defines the data structures and schema for storing sensor data
with metadata that enables categorization, labeling, and training in both
centralized and federated learning settings.

Schema Version: 1.0.0
"""

import uuid
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"


class SensorType(str, Enum):
    """Supported sensor types."""
    CSI = "csi"           # WiFi Channel State Information
    IMU = "imu"           # Accelerometer, Gyroscope, Magnetometer
    CAMERA = "camera"     # Video/Image
    MICROPHONE = "mic"    # Audio
    RADAR = "radar"       # mmWave radar
    GPS = "gps"           # Location
    CUSTOM = "custom"     # Custom sensor


class DataType(str, Enum):
    """Data type categories for ML/FL training."""
    WIFI_CSI = "wifi_csi"
    MOTION_IMU = "motion_imu"
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    RADAR_POINT_CLOUD = "radar_point_cloud"
    TIME_SERIES = "time_series"
    TABULAR = "tabular"


class FileFormat(str, Enum):
    """Supported file formats."""
    CSV = "csv"
    JSON = "json"
    NPY = "npy"           # NumPy array
    NPZ = "npz"           # Compressed NumPy
    WAV = "wav"
    MP4 = "mp4"
    JPG = "jpg"
    PNG = "png"
    PICKLE = "pkl"


@dataclass
class CollectionInfo:
    """Information about data collection."""
    duration_seconds: float = 0.0
    sample_rate_hz: float = 0.0
    num_samples: int = 0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    collection_method: str = "automatic"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollectionInfo":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LabelInfo:
    """Label information for supervised learning."""
    activity: Optional[str] = None
    subject_id: Optional[str] = None
    environment: Optional[str] = None
    session_id: Optional[str] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    confidence: float = 1.0
    labeler: str = "manual"
    verified: bool = False
    custom_labels: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LabelInfo":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PreprocessingInfo:
    """Information about applied preprocessing."""
    applied: bool = False
    pipeline: Optional[List[Dict[str, Any]]] = None
    output_shape: Optional[List[int]] = None
    normalization_params: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessingInfo":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class FLStatus:
    """Federated Learning status for this data file."""
    available_for_training: bool = True
    used_in_rounds: List[int] = field(default_factory=list)
    last_used: Optional[str] = None
    assigned_partition: Optional[int] = None
    server_url: Optional[str] = None
    experiment_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FLStatus":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def mark_used(self, round_num: int):
        """Mark this data as used in an FL round."""
        if round_num not in self.used_in_rounds:
            self.used_in_rounds.append(round_num)
        self.last_used = datetime.utcnow().isoformat() + "Z"


@dataclass
class QualityInfo:
    """Data quality information."""
    completeness: float = 1.0
    has_gaps: bool = False
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)
    checksum: Optional[str] = None
    file_size_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityInfo":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DataFileMetadata:
    """Complete metadata for a data file.
    
    This is the main schema for metadata files (.meta.json).
    """
    file_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    sensor_type: str = "custom"
    data_type: str = "time_series"
    file_format: str = "csv"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    device_id: str = ""
    
    collection: CollectionInfo = field(default_factory=CollectionInfo)
    labels: LabelInfo = field(default_factory=LabelInfo)
    preprocessing: PreprocessingInfo = field(default_factory=PreprocessingInfo)
    fl_status: FLStatus = field(default_factory=FLStatus)
    quality: QualityInfo = field(default_factory=QualityInfo)
    
    schema_version: str = SCHEMA_VERSION
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "sensor_type": self.sensor_type,
            "data_type": self.data_type,
            "file_format": self.file_format,
            "created_at": self.created_at,
            "device_id": self.device_id,
            "collection": self.collection.to_dict(),
            "labels": self.labels.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
            "fl_status": self.fl_status.to_dict(),
            "quality": self.quality.to_dict(),
            "schema_version": self.schema_version,
            "custom_metadata": self.custom_metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataFileMetadata":
        """Create from dictionary."""
        return cls(
            file_id=data.get("file_id", str(uuid.uuid4())),
            filename=data.get("filename", ""),
            sensor_type=data.get("sensor_type", "custom"),
            data_type=data.get("data_type", "time_series"),
            file_format=data.get("file_format", "csv"),
            created_at=data.get("created_at", datetime.utcnow().isoformat() + "Z"),
            device_id=data.get("device_id", ""),
            collection=CollectionInfo.from_dict(data.get("collection", {})),
            labels=LabelInfo.from_dict(data.get("labels", {})),
            preprocessing=PreprocessingInfo.from_dict(data.get("preprocessing", {})),
            fl_status=FLStatus.from_dict(data.get("fl_status", {})),
            quality=QualityInfo.from_dict(data.get("quality", {})),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            custom_metadata=data.get("custom_metadata", {}),
        )
    
    def save(self, path: Path) -> bool:
        """Save metadata to JSON file.
        
        Args:
            path: Path to save metadata file
        
        Returns:
            True if successful
        """
        try:
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.debug(f"Saved metadata to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save metadata to {path}: {e}")
            return False
    
    @classmethod
    def load(cls, path: Path) -> Optional["DataFileMetadata"]:
        """Load metadata from JSON file.
        
        Args:
            path: Path to metadata file
        
        Returns:
            DataFileMetadata instance or None if failed
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load metadata from {path}: {e}")
            return None
    
    def get_metadata_path(self, data_dir: Path) -> Path:
        """Get the metadata file path for this data file.
        
        Args:
            data_dir: Data directory
        
        Returns:
            Path to metadata file
        """
        base_name = Path(self.filename).stem
        return data_dir / f"{base_name}.meta.json"
    
    def validate(self) -> List[str]:
        """Validate the metadata.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not self.filename:
            errors.append("filename is required")
        
        if not self.device_id:
            errors.append("device_id is required")
        
        if self.collection.num_samples <= 0:
            errors.append("collection.num_samples must be positive")
        
        if self.quality.completeness < 0 or self.quality.completeness > 1:
            errors.append("quality.completeness must be between 0 and 1")
        
        return errors


def generate_filename(
    sensor_type: str,
    file_format: str = "csv",
    timestamp: Optional[datetime] = None,
) -> str:
    """Generate a filename following the naming convention.
    
    Args:
        sensor_type: Type of sensor (csi, imu, etc.)
        file_format: File extension
        timestamp: Timestamp (defaults to now)
    
    Returns:
        Filename string
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    ts_str = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
    return f"{sensor_type}_{ts_str}.{file_format}"


def get_metadata_filename(data_filename: str) -> str:
    """Get the metadata filename for a data file.
    
    Args:
        data_filename: Data file name
    
    Returns:
        Metadata filename
    """
    base = Path(data_filename).stem
    return f"{base}.meta.json"
