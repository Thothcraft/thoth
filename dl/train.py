#!/usr/bin/env python3
"""
Multi-modal training script for radar + CSI activity recognition with reconstruction tasks.

Trains three tasks end-to-end:
1. Classification: Radar + CSI -> embeddings -> classification
2. Radar reconstruction: Radar -> embedding -> reconstructed radar
3. CSI reconstruction: CSI -> embedding -> reconstructed CSI

Models produced (3 total with metadata):
1. radar_spatial_vision_encoder.pth + radar_spatial_vision_encoder.json
2. csi_temporal_signal_encoder.pth + csi_temporal_signal_encoder.json
3. multimodal_fusion_classifier.pth + multimodal_fusion_classifier.json

Each metadata file includes:
- Data type the model works on
- Input/output shapes
- What predictions the model produces
- Label indices for interpreting model outputs (for classifier)

Usage:
    python train.py --train-dataset <dataset_dir> --epochs 50 --batch-size 8
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =============================================================================
# CSI Subcarrier Mask (52 valid subcarriers from 64)
# =============================================================================
CSI_SUBCARRIER_MASK = np.array([
    False, False, False, False, False, False,  # 0-5: guard subcarriers
    True, True, True, True, True, True, True, True, True, True,  # 6-15
    True, True, True, True, True, True, True, True, True, True,  # 16-25
    True, True, True, True, True, True,  # 26-31 (including DC as True for now)
    True, True, True, True, True, True, True, True, True, True,  # 32-41
    True, True, True, True, True, True, True, True, True, True,  # 42-51
    True, True, True, True, True, True,  # 52-57
    False, False, False, False, False, False,  # 58-63: guard subcarriers
], dtype=bool)


# =============================================================================
# Radar Data Processing
# =============================================================================
def process_radar_binary(radar_bin_path: Path, num_frames: int = 100) -> np.ndarray:
    """
    Process radar binary file to 3-channel video frames using MMW-HAT method.
    
    This uses the actual MMW-HAT CubeProcessor approach to generate:
    - Channel 0: Range-Doppler (velocity vs distance)
    - Channel 1: Range-Azimuth (angle vs distance)
    - Channel 2: Range-Elevation (elevation vs distance)
    
    Processing steps:
    1. Load radar configuration from radar_config.json
    2. Initialize RadarProcessor (simplified CubeProcessor using numpy FFT)
    3. Process raw ADC data through processor.process_raw_data()
    4. Generate 3 visualizations using processor.vis_2d()
    5. Apply log-transform: 10*log10(power)
    6. Resize to 64x64 and normalize
    
    Returns:
        np.ndarray: Shape (num_frames, 3, 64, 64) - 3-channel video
                   (T=frames, C=channels, H=height, W=width)
    """
    from radar_processor import RadarProcessor
    
    with open(radar_bin_path, 'rb') as f:
        raw_data = f.read()
    
    print(f"  [RADAR] Raw binary size: {len(raw_data)} bytes")
    
    # Initialize radar processor with auto-detection (no config needed)
    processor = RadarProcessor(
        config_path=None,  # Auto-detect from data
        num_range_bin=128,
        num_azimuth_bin=16,
        num_elevation_bin=16,
        min_range=0.2
    )
    
    # Process raw ADC data
    processor.process_raw_data(raw_data)
    print(f"  [RADAR] Processed data cube shape: {processor.data_cube_fft.shape}")
    
    # Generate single 3-channel image
    image = processor.generate_3channel_image(target_size=(64, 64))
    print(f"  [RADAR] Generated 3-channel image shape: {image.shape} (C=3, H=64, W=64)")
    print(f"  [RADAR] Channel 0: Range-Doppler")
    print(f"  [RADAR] Channel 1: Range-Azimuth")
    print(f"  [RADAR] Channel 2: Range-Elevation")
    
    # For video, replicate the single frame num_frames times
    # (In production, would process multiple frames from the binary)
    video = np.tile(image[np.newaxis, :, :, :], (num_frames, 1, 1, 1))
    print(f"  [RADAR] Replicated to video shape: {video.shape} (T={num_frames}, C=3, H=64, W=64)")
    
    return video


# =============================================================================
# CSI Data Processing
# =============================================================================
def rolling_variance(mag: np.ndarray, var_window: int = 20) -> np.ndarray:
    """Compute rolling variance over a sliding window per subcarrier."""
    if var_window <= 1:
        return np.zeros_like(mag)
    n = mag.shape[0]
    cs = np.cumsum(mag, axis=0)
    cs2 = np.cumsum(mag ** 2, axis=0)
    cs = np.vstack([np.zeros((1, mag.shape[1])), cs])
    cs2 = np.vstack([np.zeros((1, mag.shape[1])), cs2])
    hi = np.arange(1, n + 1)
    lo = np.clip(hi - var_window, 0, None)
    counts = (hi - lo).reshape(-1, 1)
    means = (cs[hi] - cs[lo]) / counts
    mean_sq = (cs2[hi] - cs2[lo]) / counts
    var = np.clip(mean_sq - means ** 2, 0, None)
    return var


def rolling_variance(mag: np.ndarray, var_window: int = 20) -> np.ndarray:
    """Compute rolling variance over a sliding window per subcarrier."""
    if var_window <= 1:
        return np.zeros_like(mag)
    n = mag.shape[0]
    cs = np.cumsum(mag, axis=0)
    cs2 = np.cumsum(mag ** 2, axis=0)
    cs = np.vstack([np.zeros((1, mag.shape[1])), cs])
    cs2 = np.vstack([np.zeros((1, mag.shape[1])), cs2])
    hi = np.arange(1, n + 1)
    lo = np.clip(hi - var_window, 0, None)
    counts = (hi - lo).reshape(-1, 1)
    means = (cs[hi] - cs[lo]) / counts
    mean_sq = (cs2[hi] - cs2[lo]) / counts
    var = np.clip(mean_sq - means ** 2, 0, None)
    return var


def process_csi_csv(csi_csv_path: Path, guaranteed_sr: int = 100, var_window: int = 20, 
                    window_len: int = 100) -> np.ndarray:
    """
    Process CSI CSV file to extract features, following dl.py CSI_Loader implementation.
    
    Processing pipeline (inspired from thoth/WS/train/utils.py CSI_Loader):
    1. Read CSV with error handling for malformed rows
    2. Filter for rows starting with 'CSI_DATA' (skip log messages)
    3. Parse 128-byte CSI payload as 64 I/Q pairs:
       - Even indices [0,2,4,...] = Imaginary component
       - Odd indices [1,3,5,...] = Real component
       (This matches ESP-CSI format where I comes before Q)
    4. Apply subcarrier mask to keep 52 valid LLTF subcarriers
    5. Resample to guaranteed sample rate (100 Hz) using linear interpolation
    6. Compute magnitude: sqrt(real^2 + imag^2)
    7. Apply rolling variance over sliding window
    8. Window to fixed length (pad if necessary)
    9. Normalize to zero mean, unit variance
    
    Returns:
        np.ndarray: Shape (window_len, 52) - windowed CSI features
    
    Raises:
        ValueError: If CSI file cannot be processed
    """
    import pandas as pd
    
    # Read CSV with error handling for malformed rows
    try:
        df = pd.read_csv(csi_csv_path, header=0, on_bad_lines='skip', low_memory=False)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {csi_csv_path}: {e}")
    
    print(f"  [CSI] CSV rows read: {len(df)}")
    
    if df.empty:
        raise ValueError(f"CSV file {csi_csv_path} is empty after reading")
    
    # Filter out rows that don't start with 'CSI_DATA' (these are log messages)
    valid_mask = df['type'].str.startswith('CSI_DATA', na=False)
    df = df[valid_mask]
    print(f"  [CSI] After filtering CSI_DATA rows: {len(df)}")
    
    # Extract CSI data and timestamps
    raw_csi = df['data'].values
    raw_timestamps = df['local_timestamp'].values if 'local_timestamp' in df.columns else df.index.values
    
    # Convert timestamps to numeric, filtering out any non-numeric values
    raw_timestamps = pd.to_numeric(pd.Series(raw_timestamps), errors='coerce')
    raw_timestamps = raw_timestamps.fillna(0).values.astype(np.int64)
    
    # Parse each row into separate real / imag arrays (following dl.py CSI_Loader)
    # The 128-byte payload is 64 I/Q pairs where:
    # - Even indices [0,2,4,...,126] = Imaginary component
    # - Odd indices [1,3,5,...,127] = Real component
    real_list, imag_list, valid_timestamps = [], [], []
    for numline, line in enumerate(raw_csi):
        try:
            if not isinstance(line, str):
                continue
            csi_row = [int(x) for x in line[1:-1].split(",")]
            if len(csi_row) != 128:
                continue
            imag_list.append(csi_row[0::2])  # even indices = imaginary
            real_list.append(csi_row[1::2])  # odd indices = real
            valid_timestamps.append(raw_timestamps[numline])
        except (ValueError, IndexError) as e:
            continue
    
    if not real_list:
        raise ValueError(f"No valid CSI data found in {csi_csv_path}")
    
    # Convert to numpy arrays (N, 64)
    real = np.array(real_list, dtype=np.float64)
    imag = np.array(imag_list, dtype=np.float64)
    timestamps = np.array(valid_timestamps, dtype=np.int64)
    print(f"  [CSI] Parsed {len(real)} CSI arrays, shape: real={real.shape}, imag={imag.shape}, ts={timestamps.shape}")
    
    # Apply subcarrier mask (keep 52 valid subcarriers from 64)
    real = real[:, CSI_SUBCARRIER_MASK]
    imag = imag[:, CSI_SUBCARRIER_MASK]
    print(f"  [CSI] After subcarrier mask: real={real.shape}, imag={imag.shape} (mask sum: {CSI_SUBCARRIER_MASK.sum()})")
    
    # Resample to guaranteed sample rate (following dl.py approach)
    if len(real) > 1:
        real, imag, timestamps = _resample_equal_intervals(real, imag, timestamps, guaranteed_sr)
        print(f"  [CSI] After resample to {guaranteed_sr}Hz: real={real.shape}, imag={imag.shape}")
    
    # Compute magnitude: sqrt(real^2 + imag^2) (dl.py approach, not using complex)
    mag = np.sqrt(real ** 2 + imag ** 2)
    print(f"  [CSI] After magnitude: {mag.shape}")
    
    # Apply rolling variance (following dl.py _rolling_variance)
    mag = rolling_variance(mag, var_window)
    print(f"  [CSI] After rolling variance (window={var_window}): {mag.shape}")
    
    # Window the data
    if mag.shape[0] >= window_len:
        # Take the last window_len samples
        mag = mag[-window_len:]
        print(f"  [CSI] After windowing (last {window_len}): {mag.shape}")
    else:
        # Pad if not enough samples
        pad_width = window_len - mag.shape[0]
        mag = np.pad(mag, ((pad_width, 0), (0, 0)), mode='constant')
        print(f"  [CSI] After padding (pad={pad_width}): {mag.shape}")
    
    # Normalize
    if mag.std() > 1e-8:
        mag = (mag - mag.mean()) / (mag.std())
    else:
        mag = mag - mag.mean()
    print(f"  [CSI] After normalize: mean={mag.mean():.4f}, std={mag.std():.4f}")
    
    return mag.astype(np.float32)


def _resample_equal_intervals(real, imag, timestamps, target_sr):
    """
    Resample CSI data to equal intervals at target_sr Hz (following dl.py).
    
    Uses linear interpolation to resample both real and imaginary components
    to have uniform time spacing at the target sample rate.
    """
    if len(timestamps) < 2:
        return real, imag, timestamps
    
    # Convert to seconds
    timestamps_sec = (timestamps - timestamps[0]) / 1e9
    
    # Create target time points
    duration = timestamps_sec[-1] - timestamps_sec[0]
    target_times = np.arange(0, duration + 1/target_sr, 1/target_sr)
    
    # Interpolate real and imaginary separately
    real_resampled = np.zeros((len(target_times), real.shape[1]))
    imag_resampled = np.zeros((len(target_times), imag.shape[1]))
    
    for i in range(real.shape[1]):
        real_resampled[:, i] = np.interp(target_times, timestamps_sec, real[:, i])
        imag_resampled[:, i] = np.interp(target_times, timestamps_sec, imag[:, i])
    
    return real_resampled, imag_resampled, target_times


# =============================================================================
# Dataset Class
# =============================================================================
class MultiModalDataset(Dataset):
    """Dataset for radar + CSI multi-modal activity recognition."""
    
    def __init__(self, dataset_dir: Path, guaranteed_sr: int = 100, var_window: int = 20,
                 window_len: int = 100, num_radar_frames: int = 100):
        self.dataset_dir = Path(dataset_dir)
        self.guaranteed_sr = guaranteed_sr
        self.var_window = var_window
        self.window_len = window_len
        self.num_radar_frames = num_radar_frames
        
        # Discover classes and samples
        self.samples = []
        self.label_map = {}
        self._discover_samples()
        
        if not self.samples:
            raise ValueError(f"No valid samples found in {dataset_dir}")
        
        print(f"Found {len(self.samples)} samples across {len(self.label_map)} classes")
        print(f"Classes: {list(self.label_map.keys())}")
    
    def _discover_samples(self):
        """Discover all samples in the dataset directory."""
        class_dirs = [d for d in self.dataset_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        for class_idx, class_dir in enumerate(sorted(class_dirs)):
            class_name = class_dir.name
            self.label_map[class_name] = class_idx
            
            # Find timestamped subdirectories
            timestamp_dirs = [d for d in class_dir.iterdir() if d.is_dir()]
            
            for timestamp_dir in timestamp_dirs:
                # Check for required files
                radar_files = list(timestamp_dir.glob("mmw_radar_raw_*.bin"))
                csi_files = list(timestamp_dir.glob("wifi_csi_raw.csv"))
                
                if radar_files and csi_files:
                    self.samples.append({
                        'radar_path': radar_files[0],
                        'csi_path': csi_files[0],
                        'label': class_idx,
                        'class_name': class_name,
                        'timestamp': timestamp_dir.name
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        print(f"\n  [DATASET] Loading sample {idx}: {sample['class_name']}/{sample['timestamp']}")
        
        try:
            # Process radar data
            radar_video = process_radar_binary(sample['radar_path'], self.num_radar_frames)
            
            # Process CSI data
            csi_data = process_csi_csv(sample['csi_path'], self.guaranteed_sr, self.var_window, self.window_len)
            
            # Convert to tensors
            radar_tensor = torch.from_numpy(radar_video).float()  # (T, C, H, W)
            csi_tensor = torch.from_numpy(csi_data).float()  # (T, 52)
            label_tensor = torch.tensor(sample['label'], dtype=torch.long)
            
            print(f"  [DATASET] Radar tensor shape: {radar_tensor.shape}")
            print(f"  [DATASET] CSI tensor shape: {csi_tensor.shape}")
            print(f"  [DATASET] Label: {label_tensor.item()} ({sample['class_name']})")
            
            return {
                'radar': radar_tensor,
                'csi': csi_tensor,
                'label': label_tensor
            }
        except Exception as e:
            raise ValueError(f"Failed to load sample {sample['csi_path'].parent}: {e}")
    
    def split_by_time(self, split_percent: float = 0.8):
        """
        Split dataset by timestamp (more recent = test).
        
        Args:
            split_percent: Percentage for training (default 0.8)
        
        Returns:
            train_dataset, test_dataset
        """
        # Sort samples by timestamp
        sorted_samples = sorted(self.samples, key=lambda x: x['timestamp'])
        
        split_idx = int(len(sorted_samples) * split_percent)
        
        train_samples = sorted_samples[:split_idx]
        test_samples = sorted_samples[split_idx:]
        
        # Create new dataset instances
        train_ds = MultiModalDataset.__new__(MultiModalDataset)
        train_ds.dataset_dir = self.dataset_dir
        train_ds.guaranteed_sr = self.guaranteed_sr
        train_ds.var_window = self.var_window
        train_ds.window_len = self.window_len
        train_ds.num_radar_frames = self.num_radar_frames
        train_ds.samples = train_samples
        train_ds.label_map = self.label_map
        
        test_ds = MultiModalDataset.__new__(MultiModalDataset)
        test_ds.dataset_dir = self.dataset_dir
        test_ds.guaranteed_sr = self.guaranteed_sr
        test_ds.var_window = self.var_window
        test_ds.window_len = self.window_len
        test_ds.num_radar_frames = self.num_radar_frames
        test_ds.samples = test_samples
        test_ds.label_map = self.label_map
        
        print(f"Split dataset: {len(train_samples)} train, {len(test_samples)} test")
        
        return train_ds, test_ds


# =============================================================================
# Model Architectures
# =============================================================================
class RadarEncoder(nn.Module):
    """3D CNN encoder for radar video data."""
    
    def __init__(self, input_channels=3, output_dim=128):
        super().__init__()
        
        # 3D CNN layers
        self.conv3d_1 = nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_1 = nn.BatchNorm3d(32)
        self.pool3d_1 = nn.MaxPool3d((2, 2, 2))
        
        self.conv3d_2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_2 = nn.BatchNorm3d(64)
        self.pool3d_2 = nn.MaxPool3d((2, 2, 2))
        
        self.conv3d_3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_3 = nn.BatchNorm3d(128)
        self.pool3d_3 = nn.MaxPool3d((2, 2, 2))
        
        # Calculate flattened dimension after convolutions
        # Input: (T, 3, 64, 64) -> (T, 32, 32, 32) -> (T, 64, 16, 16) -> (T, 128, 8, 8)
        # After pooling time: (T/8, 128, 8, 8)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 4, 4))
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        # x shape: (B, T, C, H, W) -> need (B, C, T, H, W)
        print(f"    [RADAR_ENCODER] Input shape: {x.shape}")
        x = x.permute(0, 2, 1, 3, 4)
        print(f"    [RADAR_ENCODER] After permute: {x.shape}")
        
        x = F.relu(self.bn3d_1(self.conv3d_1(x)))
        x = self.pool3d_1(x)
        print(f"    [RADAR_ENCODER] After conv3d_1 + pool: {x.shape}")
        
        x = F.relu(self.bn3d_2(self.conv3d_2(x)))
        x = self.pool3d_2(x)
        print(f"    [RADAR_ENCODER] After conv3d_2 + pool: {x.shape}")
        
        x = F.relu(self.bn3d_3(self.conv3d_3(x)))
        x = self.pool3d_3(x)
        print(f"    [RADAR_ENCODER] After conv3d_3 + pool: {x.shape}")
        
        x = self.adaptive_pool(x)
        print(f"    [RADAR_ENCODER] After adaptive_pool: {x.shape}")
        x = x.view(x.size(0), -1)
        print(f"    [RADAR_ENCODER] After view: {x.shape}")
        x = self.fc(x)
        print(f"    [RADAR_ENCODER] Output shape: {x.shape}")
        
        return x


class CSIEncoder(nn.Module):
    """1D CNN encoder for CSI temporal data."""
    
    def __init__(self, input_dim=52, window_len=100, output_dim=128):
        super().__init__()
        self.window_len = window_len
        self.input_dim = input_dim
        
        # 1D CNN layers
        self.conv1d_1 = nn.Conv1d(input_dim, 64, kernel_size=7, padding=3)
        self.bn1d_1 = nn.BatchNorm1d(64)
        self.pool1d_1 = nn.MaxPool1d(2)
        
        self.conv1d_2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn1d_2 = nn.BatchNorm1d(128)
        self.pool1d_2 = nn.MaxPool1d(2)
        
        self.conv1d_3 = nn.Conv1d(128, 256, kernel_size=7, padding=3)
        self.bn1d_3 = nn.BatchNorm1d(256)
        self.pool1d_3 = nn.MaxPool1d(2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        # x shape: (B, T, C) -> need (B, C, T)
        print(f"    [CSI_ENCODER] Input shape: {x.shape}")
        x = x.permute(0, 2, 1)
        print(f"    [CSI_ENCODER] After permute: {x.shape}")
        
        x = F.relu(self.bn1d_1(self.conv1d_1(x)))
        x = self.pool1d_1(x)
        print(f"    [CSI_ENCODER] After conv1d_1 + pool: {x.shape}")
        
        x = F.relu(self.bn1d_2(self.conv1d_2(x)))
        x = self.pool1d_2(x)
        print(f"    [CSI_ENCODER] After conv1d_2 + pool: {x.shape}")
        
        x = F.relu(self.bn1d_3(self.conv1d_3(x)))
        x = self.pool1d_3(x)
        print(f"    [CSI_ENCODER] After conv1d_3 + pool: {x.shape}")
        
        x = self.global_pool(x)
        print(f"    [CSI_ENCODER] After global_pool: {x.shape}")
        x = x.view(x.size(0), -1)
        print(f"    [CSI_ENCODER] After view: {x.shape}")
        x = self.fc(x)
        print(f"    [CSI_ENCODER] Output shape: {x.shape}")
        
        return x


class RadarDecoder(nn.Module):
    """3D CNN decoder for reconstructing radar video from embeddings."""
    
    def __init__(self, input_dim=128, output_channels=3, num_frames=100):
        super().__init__()
        self.num_frames = num_frames
        
        # Fully connected to expand embedding
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * 4 * 4),
            nn.ReLU()
        )
        
        # Reshape to (B, 128, 1, 4, 4) and apply 3D transposed convolutions
        # Target: (B, 3, 100, 64, 64)
        # Path: 1 -> 100 (time), 4 -> 64 (spatial)
        self.conv3d_transpose_1 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1))
        self.bn3d_1 = nn.BatchNorm3d(64)
        
        self.conv3d_transpose_2 = nn.ConvTranspose3d(64, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1))
        self.bn3d_2 = nn.BatchNorm3d(32)
        
        self.conv3d_transpose_3 = nn.ConvTranspose3d(32, output_channels, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1))
        
        # Final upsampling to exact target size
        self.final_upsample = nn.Upsample(size=(num_frames, 64, 64), mode='trilinear', align_corners=False)
    
    def forward(self, x):
        print(f"    [RADAR_DECODER] Input shape: {x.shape}")
        x = self.fc(x)
        print(f"    [RADAR_DECODER] After fc: {x.shape}")
        x = x.view(x.size(0), 128, 1, 4, 4)
        print(f"    [RADAR_DECODER] After view: {x.shape}")
        
        x = F.relu(self.bn3d_1(self.conv3d_transpose_1(x)))
        print(f"    [RADAR_DECODER] After conv3d_transpose_1: {x.shape}")
        
        x = F.relu(self.bn3d_2(self.conv3d_transpose_2(x)))
        print(f"    [RADAR_DECODER] After conv3d_transpose_2: {x.shape}")
        
        x = self.conv3d_transpose_3(x)
        print(f"    [RADAR_DECODER] After conv3d_transpose_3: {x.shape}")
        
        x = self.final_upsample(x)
        print(f"    [RADAR_DECODER] After final_upsample: {x.shape}")
        
        return x


class CSIDecoder(nn.Module):
    """1D CNN decoder for reconstructing CSI temporal data from embeddings."""
    
    def __init__(self, input_dim=128, output_dim=52, window_len=100):
        super().__init__()
        self.window_len = window_len
        self.output_dim = output_dim
        
        # Fully connected to expand embedding
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # Upsample using transposed convolutions
        # Target: (B, 100, 52)
        self.conv1d_transpose_1 = nn.ConvTranspose1d(256, 128, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.bn1d_1 = nn.BatchNorm1d(128)
        
        self.conv1d_transpose_2 = nn.ConvTranspose1d(128, 64, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.bn1d_2 = nn.BatchNorm1d(64)
        
        self.conv1d_transpose_3 = nn.ConvTranspose1d(64, 64, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.bn1d_3 = nn.BatchNorm1d(64)
        
        self.fc_final = nn.Linear(64, output_dim)
    
    def forward(self, x):
        print(f"    [CSI_DECODER] Input shape: {x.shape}")
        x = self.fc(x)
        print(f"    [CSI_DECODER] After fc: {x.shape}")
        x = x.unsqueeze(2)  # Add sequence dimension
        print(f"    [CSI_DECODER] After unsqueeze: {x.shape}")
        
        x = F.relu(self.bn1d_1(self.conv1d_transpose_1(x)))
        print(f"    [CSI_DECODER] After conv1d_transpose_1: {x.shape}")
        
        x = F.relu(self.bn1d_2(self.conv1d_transpose_2(x)))
        print(f"    [CSI_DECODER] After conv1d_transpose_2: {x.shape}")
        
        x = F.relu(self.bn1d_3(self.conv1d_transpose_3(x)))
        print(f"    [CSI_DECODER] After conv1d_transpose_3: {x.shape}")
        
        # Upsample to target window length
        x = F.interpolate(x, size=self.window_len, mode='linear', align_corners=False)
        print(f"    [CSI_DECODER] After interpolate to window_len: {x.shape}")
        
        # Apply final linear layer to get output dimension
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.fc_final(x)
        print(f"    [CSI_DECODER] After fc_final: {x.shape}")
        
        return x  # (B, T, C)


class RadarAutoEncoder(nn.Module):
    """Complete radar autoencoder (encoder + decoder)."""
    
    def __init__(self, input_channels=3, num_frames=100, latent_dim=128):
        super().__init__()
        self.encoder = RadarEncoder(input_channels=input_channels, output_dim=latent_dim)
        self.decoder = RadarDecoder(input_dim=latent_dim, output_channels=input_channels, num_frames=num_frames)
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


class CSIAutoEncoder(nn.Module):
    """Complete CSI autoencoder (encoder + decoder)."""
    
    def __init__(self, input_dim=52, window_len=100, latent_dim=128):
        super().__init__()
        self.encoder = CSIEncoder(input_dim=input_dim, window_len=window_len, output_dim=latent_dim)
        self.decoder = CSIDecoder(input_dim=latent_dim, output_dim=input_dim, window_len=window_len)
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


class MultiModalClassifier(nn.Module):
    """Multi-modal classifier combining radar and CSI encoders with reconstruction heads."""
    
    def __init__(self, num_classes, radar_output_dim=128, csi_output_dim=128, 
                 num_frames=100, csi_window_len=100, csi_input_dim=52):
        super().__init__()
        
        self.radar_encoder = RadarEncoder(output_dim=radar_output_dim)
        self.csi_encoder = CSIEncoder(output_dim=csi_output_dim, input_dim=csi_input_dim, window_len=csi_window_len)
        
        # Reconstruction heads (decoders)
        self.radar_decoder = RadarDecoder(input_dim=radar_output_dim, output_channels=3, num_frames=num_frames)
        self.csi_decoder = CSIDecoder(input_dim=csi_output_dim, output_dim=csi_input_dim, window_len=csi_window_len)
        
        # Classifier takes concatenated features (128 + 128 = 256)
        self.classifier = nn.Sequential(
            nn.Linear(radar_output_dim + csi_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, radar, csi, return_reconstructions=True):
        print(f"  [CLASSIFIER] Forward pass started")
        radar_features = self.radar_encoder(radar)
        csi_features = self.csi_encoder(csi)
        
        # Concatenate features for classification
        combined = torch.cat([radar_features, csi_features], dim=1)
        print(f"  [CLASSIFIER] Combined features shape: {combined.shape}")
        
        logits = self.classifier(combined)
        print(f"  [CLASSIFIER] Logits shape: {logits.shape}")
        
        if return_reconstructions:
            radar_reconstructed = self.radar_decoder(radar_features)
            csi_reconstructed = self.csi_decoder(csi_features)
            print(f"  [CLASSIFIER] Radar reconstruction shape: {radar_reconstructed.shape}")
            print(f"  [CLASSIFIER] CSI reconstruction shape: {csi_reconstructed.shape}")
            return logits, radar_reconstructed, csi_reconstructed
        
        return logits


# =============================================================================
# Training Functions
# =============================================================================
def train_epoch(model, dataloader, optimizer, criterion_class, criterion_recon, 
                recon_weight=0.1, device='cpu'):
    model.train()
    total_loss = 0
    total_class_loss = 0
    total_radar_recon_loss = 0
    total_csi_recon_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        radar = batch['radar'].to(device)
        csi = batch['csi'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits, radar_recon, csi_recon = model(radar, csi, return_reconstructions=True)
        
        # Classification loss
        class_loss = criterion_class(logits, labels)
        
        # Reconstruction losses (MSE)
        # Permute radar to match decoder output shape
        radar_permuted = radar.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) -> (B, C, T, H, W)
        radar_recon_loss = criterion_recon(radar_recon, radar_permuted)
        csi_recon_loss = criterion_recon(csi_recon, csi)
        
        # Combined loss
        total_batch_loss = class_loss + recon_weight * (radar_recon_loss + csi_recon_loss)
        
        total_batch_loss.backward()
        optimizer.step()
        
        total_loss += total_batch_loss.item()
        total_class_loss += class_loss.item()
        total_radar_recon_loss += radar_recon_loss.item()
        total_csi_recon_loss += csi_recon_loss.item()
        
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': total_batch_loss.item(), 
            'cls': class_loss.item(),
            'recon': (radar_recon_loss + csi_recon_loss).item(),
            'acc': 100. * correct / total
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_class_loss = total_class_loss / len(dataloader)
    avg_radar_recon_loss = total_radar_recon_loss / len(dataloader)
    avg_csi_recon_loss = total_csi_recon_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, avg_class_loss, avg_radar_recon_loss, avg_csi_recon_loss, accuracy


def validate(model, dataloader, criterion_class, criterion_recon, 
             recon_weight=0.1, device='cpu'):
    model.eval()
    total_loss = 0
    total_class_loss = 0
    total_radar_recon_loss = 0
    total_csi_recon_loss = 0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            radar = batch['radar'].to(device)
            csi = batch['csi'].to(device)
            labels = batch['label'].to(device)
            
            logits, radar_recon, csi_recon = model(radar, csi, return_reconstructions=True)
            
            # Classification loss
            class_loss = criterion_class(logits, labels)
            
            # Reconstruction losses
            radar_permuted = radar.permute(0, 2, 1, 3, 4)
            radar_recon_loss = criterion_recon(radar_recon, radar_permuted)
            csi_recon_loss = criterion_recon(csi_recon, csi)
            
            # Combined loss
            total_batch_loss = class_loss + recon_weight * (radar_recon_loss + csi_recon_loss)
            
            total_loss += total_batch_loss.item()
            total_class_loss += class_loss.item()
            total_radar_recon_loss += radar_recon_loss.item()
            total_csi_recon_loss += csi_recon_loss.item()
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    avg_class_loss = total_class_loss / len(dataloader)
    avg_radar_recon_loss = total_radar_recon_loss / len(dataloader)
    avg_csi_recon_loss = total_csi_recon_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, avg_class_loss, avg_radar_recon_loss, avg_csi_recon_loss, accuracy, all_predictions, all_labels


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train multi-modal radar + CSI activity recognition model')
    parser.add_argument('--train-dataset', type=str, required=True, help='Path to training dataset directory')
    parser.add_argument('--test-dataset', type=str, default=None, help='Path to test dataset directory (optional)')
    parser.add_argument('--split-percent', type=float, default=0.8, help='Train/test split percent if only train dataset provided (default 0.8)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--guaranteed-sr', type=int, default=100, help='Guaranteed sample rate for CSI')
    parser.add_argument('--var-window', type=int, default=20, help='Rolling variance window')
    parser.add_argument('--window-len', type=int, default=100, help='CSI window length')
    parser.add_argument('--num-radar-frames', type=int, default=100, help='Number of radar frames')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for models and plots (default: <script_dir>/results)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        output_dir = Path(__file__).parent / 'results'
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load datasets
    print(f"\nLoading training dataset from: {args.train_dataset}")
    train_dataset = MultiModalDataset(
        dataset_dir=Path(args.train_dataset),
        guaranteed_sr=args.guaranteed_sr,
        var_window=args.var_window,
        window_len=args.window_len,
        num_radar_frames=args.num_radar_frames
    )
    
    if args.test_dataset:
        print(f"\nLoading test dataset from: {args.test_dataset}")
        test_dataset = MultiModalDataset(
            dataset_dir=Path(args.test_dataset),
            guaranteed_sr=args.guaranteed_sr,
            var_window=args.var_window,
            window_len=args.window_len,
            num_radar_frames=args.num_radar_frames
        )
    else:
        print(f"\nSplitting training dataset (train={args.split_percent*100}%, test={(1-args.split_percent)*100}%)")
        train_dataset, test_dataset = train_dataset.split_by_time(split_percent=args.split_percent)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create models
    num_classes = len(train_dataset.label_map)
    class_names = list(train_dataset.label_map.keys())
    
    # Main multi-modal model
    model = MultiModalClassifier(
        num_classes=num_classes,
        num_frames=args.num_radar_frames,
        csi_window_len=args.window_len,
        csi_input_dim=52
    ).to(device)
    
    # Separate autoencoders for saving
    radar_autoencoder = RadarAutoEncoder(
        input_channels=3,
        num_frames=args.num_radar_frames,
        latent_dim=128
    ).to(device)
    
    csi_autoencoder = CSIAutoEncoder(
        input_dim=52,
        window_len=args.window_len,
        latent_dim=128
    ).to(device)
    
    # Loss functions
    criterion_class = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss()
    recon_weight = 0.1
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training history tracking
    train_radar_recon_losses = []
    val_radar_recon_losses = []
    train_csi_recon_losses = []
    val_csi_recon_losses = []
    train_class_losses = []
    val_class_losses = []
    train_accs = []
    val_accs = []
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        train_loss, train_class_loss, train_radar_recon, train_csi_recon, train_acc = train_epoch(
            model, train_loader, optimizer, criterion_class, criterion_recon, 
            recon_weight=recon_weight, device=device
        )
        val_loss, val_class_loss, val_radar_recon, val_csi_recon, val_acc, val_preds, val_labels = validate(
            model, test_loader, criterion_class, criterion_recon, 
            recon_weight=recon_weight, device=device
        )
        
        print(f"\n  [TRAINING] Train Loss: {train_loss:.4f}, Class: {train_class_loss:.4f}, Radar Recon: {train_radar_recon:.4f}, CSI Recon: {train_csi_recon:.4f}, Acc: {train_acc:.2f}%")
        print(f"  [TRAINING] Val Loss: {val_loss:.4f}, Class: {val_class_loss:.4f}, Radar Recon: {val_radar_recon:.4f}, CSI Recon: {val_csi_recon:.4f}, Acc: {val_acc:.2f}%")
        
        # Track history
        train_radar_recon_losses.append(train_radar_recon)
        val_radar_recon_losses.append(val_radar_recon)
        train_csi_recon_losses.append(train_csi_recon)
        val_csi_recon_losses.append(val_csi_recon)
        train_class_losses.append(train_class_loss)
        val_class_losses.append(val_class_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Update separate autoencoders
        radar_autoencoder.encoder.load_state_dict(model.radar_encoder.state_dict())
        radar_autoencoder.decoder.load_state_dict(model.radar_decoder.state_dict())
        csi_autoencoder.encoder.load_state_dict(model.csi_encoder.state_dict())
        csi_autoencoder.decoder.load_state_dict(model.csi_decoder.state_dict())
    
    # Save models with fancy names and metadata
    model_configs = {
        'radar_spatial_vision_encoder': {
            'model': radar_autoencoder,
            'data_type': 'radar_video',
            'input_shape': [args.num_radar_frames, 3, 64, 64],
            'output_shape': [args.num_radar_frames, 3, 64, 64],
            'predictions': 'reconstructed_radar_video_frames',
            'channels': ['range_doppler', 'range_azimuth', 'range_elevation'],
            'description': '3D CNN encoder-decoder for radar video reconstruction and feature extraction'
        },
        'csi_temporal_signal_encoder': {
            'model': csi_autoencoder,
            'data_type': 'csi_temporal',
            'input_shape': [args.window_len, 52],
            'output_shape': [args.window_len, 52],
            'predictions': 'reconstructed_csi_subcarrier_amplitude',
            'subcarriers': '52 LLTF subcarriers from 64 total',
            'description': '1D CNN encoder-decoder for CSI temporal signal reconstruction and feature extraction'
        },
        'multimodal_fusion_classifier': {
            'model': model,
            'data_type': 'multimodal',
            'input_shapes': {
                'radar': [args.num_radar_frames, 3, 64, 64],
                'csi': [args.window_len, 52]
            },
            'output_shape': [num_classes],
            'predictions': 'activity_class_labels',
            'label_map': train_dataset.label_map,
            'label_index': {i: name for i, name in enumerate(class_names)},
            'description': 'Fusion classifier combining radar and CSI encoders for activity recognition'
        }
    }
    
    for model_name, config in model_configs.items():
        # Save model
        model_path = output_dir / f'{model_name}.pth'
        torch.save(config['model'].state_dict(), model_path)
        
        # Create metadata
        metadata = {
            'model_name': model_name,
            'data_type': config['data_type'],
            'description': config['description'],
            'input': config.get('input_shape') or config.get('input_shapes'),
            'output': config['output_shape'],
            'predictions': config['predictions'],
            'architecture': {
                'radar_encoder_dim': 128,
                'csi_encoder_dim': 128,
                'latent_dim': 128
            }
        }
        
        # Add type-specific metadata
        if 'channels' in config:
            metadata['channels'] = config['channels']
        if 'subcarriers' in config:
            metadata['subcarriers'] = config['subcarriers']
        if 'label_map' in config:
            metadata['label_map'] = config['label_map']
            metadata['label_index'] = config['label_index']
        
        # Save metadata
        metadata_path = output_dir / f'{model_name}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  - {model_name}.pth (model)")
        print(f"  - {model_name}.json (metadata)")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Final validation accuracy: {val_accs[-1]:.2f}%")
    print(f"Models and metadata saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
