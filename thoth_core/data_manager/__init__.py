"""Data Management Module for Thoth Device.

This module provides comprehensive data management for sensor data collection,
storage, and preparation for both centralized ML/DL and Federated Learning.

Data Storage Protocol:
======================
All data files are stored in thoth/data/ with corresponding metadata files.

File Naming Convention:
- Data file: {sensor_type}_{timestamp}.{ext}
- Metadata file: {sensor_type}_{timestamp}.meta.json

Example:
- csi_2025-01-23T10-30-00.csv
- csi_2025-01-23T10-30-00.meta.json

Metadata Schema (meta.json):
{
    "file_id": "uuid",
    "filename": "csi_2025-01-23T10-30-00.csv",
    "sensor_type": "csi",
    "data_type": "wifi_csi",
    "created_at": "2025-01-23T10:30:00Z",
    "device_id": "thoth-device-001",
    "collection": {
        "duration_seconds": 60,
        "sample_rate_hz": 100,
        "num_samples": 6000
    },
    "labels": {
        "activity": "walking",
        "subject_id": "subject_01",
        "environment": "indoor_lab",
        "session_id": "session_001"
    },
    "preprocessing": {
        "applied": false,
        "pipeline": null
    },
    "fl_status": {
        "available_for_training": true,
        "used_in_rounds": [],
        "last_used": null
    },
    "quality": {
        "completeness": 1.0,
        "has_gaps": false,
        "validated": true
    },
    "schema_version": "1.0.0"
}
"""

from .protocol import (
    DataFileMetadata,
    CollectionInfo,
    LabelInfo,
    PreprocessingInfo,
    FLStatus,
    QualityInfo,
    SCHEMA_VERSION,
)
from .manager import DataManager
from .scanner import DataScanner

__all__ = [
    "DataFileMetadata",
    "CollectionInfo",
    "LabelInfo",
    "PreprocessingInfo",
    "FLStatus",
    "QualityInfo",
    "DataManager",
    "DataScanner",
    "SCHEMA_VERSION",
]
