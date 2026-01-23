"""Data Loader for FL Client.

This module provides data loading functionality for FL training,
integrating with the data manager to load local data files.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..data_manager import DataManager, DataFileMetadata

logger = logging.getLogger(__name__)


class FLDataLoader:
    """Loads and prepares data for FL training.
    
    This class:
    - Discovers available data files for FL training
    - Loads and combines data from multiple files
    - Applies preprocessing
    - Creates PyTorch DataLoaders
    """
    
    def __init__(
        self,
        data_dir: str,
        sensor_types: Optional[List[str]] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the data loader.
        
        Args:
            data_dir: Path to data directory
            sensor_types: List of sensor types to include
            preprocessing_config: Preprocessing configuration
        """
        self.data_dir = Path(data_dir)
        self.sensor_types = sensor_types or ["imu", "csi"]
        self.preprocessing_config = preprocessing_config or {}
        
        self.data_manager = DataManager(data_dir=data_dir)
        
        self._data: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._metadata: List[DataFileMetadata] = []
        self._class_names: List[str] = []
    
    def discover_data(self) -> Dict[str, Any]:
        """Discover available data for FL training.
        
        Returns:
            Dictionary with discovery results
        """
        results = {
            "total_files": 0,
            "available_for_fl": 0,
            "by_sensor_type": {},
            "by_activity": {},
            "total_samples": 0,
        }
        
        for sensor_type in self.sensor_types:
            files = self.data_manager.list_files(
                sensor_type=sensor_type,
                available_for_fl=True,
            )
            
            results["by_sensor_type"][sensor_type] = len(files)
            results["total_files"] += len(files)
            results["available_for_fl"] += len(files)
            
            for f in files:
                activity = f.labels.activity or "unlabeled"
                if activity not in results["by_activity"]:
                    results["by_activity"][activity] = 0
                results["by_activity"][activity] += 1
                results["total_samples"] += f.collection.num_samples
        
        return results
    
    def load_data(
        self,
        sensor_type: Optional[str] = None,
        activity: Optional[str] = None,
        max_files: Optional[int] = None,
        validation_split: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load data for training.
        
        Args:
            sensor_type: Filter by sensor type
            activity: Filter by activity label
            max_files: Maximum number of files to load
            validation_split: Fraction of data for validation
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        # Get available files
        files = self.data_manager.list_files(
            sensor_type=sensor_type,
            activity=activity,
            available_for_fl=True,
        )
        
        if max_files:
            files = files[:max_files]
        
        if not files:
            logger.warning("No data files found for FL training")
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Load and combine data
        all_data = []
        all_labels = []
        self._metadata = files
        
        # Build class mapping from activities
        activities = sorted(set(f.labels.activity or "unlabeled" for f in files))
        self._class_names = activities
        activity_to_idx = {a: i for i, a in enumerate(activities)}
        
        for metadata in files:
            data, _ = self.data_manager.load_data(metadata.filename)
            if data is None:
                continue
            
            # Get label
            activity = metadata.labels.activity or "unlabeled"
            label_idx = activity_to_idx[activity]
            
            # Handle different data shapes
            if data.ndim == 1:
                # Single sample
                all_data.append(data)
                all_labels.append(label_idx)
            elif data.ndim == 2:
                # Multiple samples (samples, features)
                for sample in data:
                    all_data.append(sample)
                    all_labels.append(label_idx)
            elif data.ndim == 3:
                # Windowed data (windows, seq_len, features)
                for window in data:
                    all_data.append(window)
                    all_labels.append(label_idx)
        
        if not all_data:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Convert to arrays
        self._data = np.array(all_data)
        self._labels = np.array(all_labels)
        
        # Apply preprocessing if configured
        if self.preprocessing_config:
            self._data = self._apply_preprocessing(self._data)
        
        # Split into train/val
        num_samples = len(self._data)
        indices = np.random.permutation(num_samples)
        val_size = int(num_samples * validation_split)
        
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        X_train = self._data[train_indices]
        y_train = self._labels[train_indices]
        X_val = self._data[val_indices]
        y_val = self._labels[val_indices]
        
        logger.info(f"Loaded {num_samples} samples: {len(X_train)} train, {len(X_val)} val")
        
        return X_train, y_train, X_val, y_val
    
    def _apply_preprocessing(self, data: np.ndarray) -> np.ndarray:
        """Apply preprocessing to data.
        
        Args:
            data: Input data
        
        Returns:
            Preprocessed data
        """
        # Normalize if configured
        if self.preprocessing_config.get("normalize", True):
            mean = np.mean(data, axis=0, keepdims=True)
            std = np.std(data, axis=0, keepdims=True) + 1e-8
            data = (data - mean) / std
        
        return data
    
    def get_data_loaders(
        self,
        batch_size: int = 32,
        validation_split: float = 0.2,
        sensor_type: Optional[str] = None,
    ) -> Tuple[Any, Any]:
        """Get PyTorch DataLoaders for training.
        
        Args:
            batch_size: Batch size
            validation_split: Validation split fraction
            sensor_type: Filter by sensor type
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DataLoaders")
        
        X_train, y_train, X_val, y_val = self.load_data(
            sensor_type=sensor_type,
            validation_split=validation_split,
        )
        
        if len(X_train) == 0:
            return None, None
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        return train_loader, val_loader
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return len(self._class_names)
    
    def get_class_names(self) -> List[str]:
        """Get class names."""
        return self._class_names
    
    def get_data_shape(self) -> Tuple[int, ...]:
        """Get shape of a single data sample."""
        if self._data is not None and len(self._data) > 0:
            return self._data[0].shape
        return (0,)
    
    def mark_data_used(self, round_num: int):
        """Mark loaded data as used in FL round.
        
        Args:
            round_num: FL round number
        """
        for metadata in self._metadata:
            self.data_manager.mark_used_in_fl(metadata.filename, round_num)
