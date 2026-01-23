# Data Storage Protocol

This document describes the data storage protocol used by Thoth devices for storing sensor data with metadata that enables categorization, labeling, and training in both centralized ML/DL and Federated Learning settings.

## Overview

All data files are stored in `thoth/data/` with corresponding metadata files. The metadata enables:
- **Categorization**: Organize data by sensor type, activity, subject, environment
- **Labeling**: Store ground truth labels for supervised learning
- **FL Coordination**: Track which data has been used in FL rounds
- **Quality Control**: Validate data completeness and integrity

## File Naming Convention

```
Data file:     {sensor_type}_{timestamp}.{ext}
Metadata file: {sensor_type}_{timestamp}.meta.json
```

### Examples
```
csi_2025-01-23T10-30-00.csv       # WiFi CSI data
csi_2025-01-23T10-30-00.meta.json # Corresponding metadata

imu_2025-01-23T10-35-00.npy       # IMU data (NumPy format)
imu_2025-01-23T10-35-00.meta.json # Corresponding metadata
```

### Supported Sensor Types
| Prefix | Sensor Type | Description |
|--------|-------------|-------------|
| `csi`  | WiFi CSI    | Channel State Information |
| `imu`  | IMU         | Accelerometer + Gyroscope |
| `cam`  | Camera      | Video/Image |
| `mic`  | Microphone  | Audio |
| `radar`| Radar       | mmWave point cloud |
| `gps`  | GPS         | Location |

### Supported File Formats
| Extension | Format | Use Case |
|-----------|--------|----------|
| `.npy`    | NumPy binary | Fast loading, preserves shape |
| `.npz`    | Compressed NumPy | Large datasets |
| `.csv`    | CSV text | Human readable, interoperability |
| `.json`   | JSON | Small datasets, nested structures |
| `.wav`    | WAV audio | Audio data |
| `.mp4`    | MP4 video | Video data |

## Metadata Schema (v1.0.0)

```json
{
    "file_id": "uuid-string",
    "filename": "csi_2025-01-23T10-30-00.csv",
    "sensor_type": "csi",
    "data_type": "wifi_csi",
    "file_format": "csv",
    "created_at": "2025-01-23T10:30:00Z",
    "device_id": "thoth-abc12345",
    
    "collection": {
        "duration_seconds": 60.0,
        "sample_rate_hz": 100.0,
        "num_samples": 6000,
        "start_time": "2025-01-23T10:30:00Z",
        "end_time": "2025-01-23T10:31:00Z",
        "collection_method": "automatic"
    },
    
    "labels": {
        "activity": "walking",
        "subject_id": "subject_01",
        "environment": "indoor_lab",
        "session_id": "session_001",
        "class_id": 0,
        "class_name": "walking",
        "confidence": 1.0,
        "labeler": "manual",
        "verified": true,
        "custom_labels": {}
    },
    
    "preprocessing": {
        "applied": false,
        "pipeline": null,
        "output_shape": null,
        "normalization_params": null
    },
    
    "fl_status": {
        "available_for_training": true,
        "used_in_rounds": [],
        "last_used": null,
        "assigned_partition": null,
        "server_url": null,
        "experiment_id": null
    },
    
    "quality": {
        "completeness": 1.0,
        "has_gaps": false,
        "validated": true,
        "validation_errors": [],
        "checksum": "md5-hash",
        "file_size_bytes": 1024000
    },
    
    "schema_version": "1.0.0",
    "custom_metadata": {}
}
```

## Field Descriptions

### Root Fields
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file_id` | string | Yes | Unique identifier (UUID) |
| `filename` | string | Yes | Data file name |
| `sensor_type` | string | Yes | Sensor type (csi, imu, etc.) |
| `data_type` | string | Yes | Data category for ML |
| `file_format` | string | Yes | File extension |
| `created_at` | string | Yes | ISO 8601 timestamp |
| `device_id` | string | Yes | Device identifier |

### Collection Info
| Field | Type | Description |
|-------|------|-------------|
| `duration_seconds` | float | Collection duration |
| `sample_rate_hz` | float | Sampling rate |
| `num_samples` | int | Number of samples |
| `start_time` | string | Collection start time |
| `end_time` | string | Collection end time |
| `collection_method` | string | "automatic" or "manual" |

### Labels
| Field | Type | Description |
|-------|------|-------------|
| `activity` | string | Activity label (walking, sitting, etc.) |
| `subject_id` | string | Subject identifier |
| `environment` | string | Environment description |
| `session_id` | string | Session identifier |
| `class_id` | int | Numeric class ID |
| `class_name` | string | Class name |
| `confidence` | float | Label confidence (0-1) |
| `labeler` | string | "manual", "automatic", "model" |
| `verified` | bool | Whether label is verified |
| `custom_labels` | object | Additional custom labels |

### FL Status
| Field | Type | Description |
|-------|------|-------------|
| `available_for_training` | bool | Whether data can be used for FL |
| `used_in_rounds` | array | List of FL round numbers |
| `last_used` | string | Timestamp of last use |
| `assigned_partition` | int | FL partition assignment |
| `server_url` | string | FL server URL |
| `experiment_id` | string | FL experiment ID |

### Quality
| Field | Type | Description |
|-------|------|-------------|
| `completeness` | float | Data completeness (0-1) |
| `has_gaps` | bool | Whether data has gaps |
| `validated` | bool | Whether data is validated |
| `validation_errors` | array | List of validation errors |
| `checksum` | string | MD5 checksum |
| `file_size_bytes` | int | File size in bytes |

## Usage Examples

### Python: Save Data with Metadata
```python
from thoth.src.data_manager import DataManager

manager = DataManager()

# Save sensor data
data_path, metadata = manager.save_data(
    data=sensor_data,  # NumPy array
    sensor_type="imu",
    labels={
        "activity": "walking",
        "subject_id": "subject_01",
    },
    collection_info={
        "duration_seconds": 60,
        "sample_rate_hz": 100,
    },
)
```

### Python: Load Data for FL Training
```python
from thoth.src.data_manager import DataManager

manager = DataManager()

# Get all data available for FL
files = manager.list_files(
    sensor_type="imu",
    available_for_fl=True,
)

# Load data
for metadata in files:
    data, meta = manager.load_data(metadata.filename)
    # Use data for training...
```

### Python: Update Labels
```python
manager.update_labels(
    filename="imu_2025-01-23T10-30-00.npy",
    labels={
        "activity": "running",  # Correct the label
        "verified": True,
    },
)
```

### Python: Mark Data Used in FL
```python
manager.mark_used_in_fl(
    filename="imu_2025-01-23T10-30-00.npy",
    round_num=5,
)
```

## Directory Structure

```
thoth/data/
├── device_id.txt              # Device identifier
├── config/                    # Configuration files
│   ├── auth.json
│   └── wifi_credentials.json
│
├── csi_2025-01-23T10-30-00.csv
├── csi_2025-01-23T10-30-00.meta.json
├── imu_2025-01-23T10-35-00.npy
├── imu_2025-01-23T10-35-00.meta.json
└── ...
```

## Extensibility

### Adding New Sensor Types
1. Add the sensor type to `SensorType` enum in `protocol.py`
2. Add data type mapping in `DataManager._infer_data_type()`
3. Create sensor implementation in `sensors/` module

### Adding Custom Metadata Fields
Use the `custom_metadata` field for application-specific data:
```json
{
    "custom_metadata": {
        "experiment_name": "HAR Study 2025",
        "location_id": "building_a_floor_2",
        "weather": "sunny"
    }
}
```

### Schema Versioning
The `schema_version` field tracks metadata format changes. When loading older metadata, the system should handle version differences gracefully.
