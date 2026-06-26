#!/usr/bin/env python3
"""
Radar classifier training script.

Trains a radar-only classifier for activity recognition.
Input: 4D radar tensor (frames, height, width, channels)
Output: Classification labels (folder names from training dataset)

Model produced:
1. radar_classifier.pth + radar_classifier.json

Metadata includes labels field for interpreting model outputs.

Usage:
    python train.py --train-dataset <dataset_dir> --epochs 50 --batch-size 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =============================================================================
# Radar Data Processing
# =============================================================================
def process_radar_binary(radar_bin_path: Path, num_frames: int = 100) -> np.ndarray:
    """Process radar binary file to 3-channel video frames using MMW-HAT method."""
    from radar_processor import RadarProcessor

    with open(radar_bin_path, 'rb') as f:
        raw_data = f.read()

    print(f"  [RADAR] Raw binary size: {len(raw_data)} bytes")

    processor = RadarProcessor(
        config_path=None,
        num_range_bin=128,
        num_azimuth_bin=16,
        num_elevation_bin=16,
        min_range=0.2
    )

    processor.process_raw_data(raw_data)
    print(f"  [RADAR] Processed data cube shape: {processor.data_cube_fft.shape}")

    image = processor.generate_3channel_image(target_size=(64, 64))
    print(f"  [RADAR] Generated 3-channel image shape: {image.shape} (C=3, H=64, W=64)")
    print(f"  [RADAR] Channel 0: RDI (Range-Doppler Intensity)")
    print(f"  [RADAR] Channel 1: Azimuth-Range")
    print(f"  [RADAR] Channel 2: Azimuth-Doppler")

    video = np.tile(image[np.newaxis, :, :, :], (num_frames, 1, 1, 1))
    print(f"  [RADAR] Replicated to video shape: {video.shape} (T={num_frames}, C=3, H=64, W=64)")

    return video


# =============================================================================
# Dataset Class
# =============================================================================
class RadarDataset(Dataset):
    """Dataset for radar-only activity recognition."""

    def __init__(self, dataset_dir: Path, num_radar_frames: int = 100):
        self.dataset_dir = Path(dataset_dir)
        self.num_radar_frames = num_radar_frames

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

            timestamp_dirs = [d for d in class_dir.iterdir() if d.is_dir()]

            for timestamp_dir in timestamp_dirs:
                radar_files = list(timestamp_dir.glob("mmw_radar_raw_*.bin"))

                if radar_files:
                    self.samples.append({
                        'radar_path': radar_files[0],
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
            radar_video = process_radar_binary(sample['radar_path'], self.num_radar_frames)
            radar_tensor = torch.from_numpy(radar_video).float()  # (T, C, H, W)
            label_tensor = torch.tensor(sample['label'], dtype=torch.long)

            print(f"  [DATASET] Radar tensor shape: {radar_tensor.shape}")
            print(f"  [DATASET] Label: {label_tensor.item()} ({sample['class_name']})")

            return {
                'radar': radar_tensor,
                'label': label_tensor
            }
        except Exception as e:
            raise ValueError(f"Failed to load sample {sample['radar_path'].parent}: {e}")

    def split_by_time(self, split_percent: float = 0.8):
        """Split dataset by timestamp (more recent = test)."""
        sorted_samples = sorted(self.samples, key=lambda x: x['timestamp'])
        split_idx = int(len(sorted_samples) * split_percent)

        train_samples = sorted_samples[:split_idx]
        test_samples = sorted_samples[split_idx:]

        train_ds = RadarDataset.__new__(RadarDataset)
        train_ds.dataset_dir = self.dataset_dir
        train_ds.num_radar_frames = self.num_radar_frames
        train_ds.samples = train_samples
        train_ds.label_map = self.label_map

        test_ds = RadarDataset.__new__(RadarDataset)
        test_ds.dataset_dir = self.dataset_dir
        test_ds.num_radar_frames = self.num_radar_frames
        test_ds.samples = test_samples
        test_ds.label_map = self.label_map

        print(f"Split dataset: {len(train_samples)} train, {len(test_samples)} test")

        return train_ds, test_ds


# =============================================================================
# Model Architecture
# =============================================================================
class RadarClassifier(nn.Module):
    """3D CNN classifier for radar video data."""

    def __init__(self, input_channels=3, num_classes=2):
        super().__init__()

        self.conv3d_1 = nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_1 = nn.BatchNorm3d(32)
        self.pool3d_1 = nn.MaxPool3d((2, 2, 2))

        self.conv3d_2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_2 = nn.BatchNorm3d(64)
        self.pool3d_2 = nn.MaxPool3d((2, 2, 2))

        self.conv3d_3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_3 = nn.BatchNorm3d(128)
        self.pool3d_3 = nn.MaxPool3d((2, 2, 2))

        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 4, 4))

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        print(f"    [CLASSIFIER] Input shape: {x.shape}")
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) -> (B, C, T, H, W)
        print(f"    [CLASSIFIER] After permute: {x.shape}")

        x = F.relu(self.bn3d_1(self.conv3d_1(x)))
        x = self.pool3d_1(x)
        print(f"    [CLASSIFIER] After conv3d_1 + pool: {x.shape}")

        x = F.relu(self.bn3d_2(self.conv3d_2(x)))
        x = self.pool3d_2(x)
        print(f"    [CLASSIFIER] After conv3d_2 + pool: {x.shape}")

        x = F.relu(self.bn3d_3(self.conv3d_3(x)))
        x = self.pool3d_3(x)
        print(f"    [CLASSIFIER] After conv3d_3 + pool: {x.shape}")

        x = self.adaptive_pool(x)
        print(f"    [CLASSIFIER] After adaptive_pool: {x.shape}")
        x = x.view(x.size(0), -1)
        print(f"    [CLASSIFIER] After view: {x.shape}")
        x = self.fc(x)
        print(f"    [CLASSIFIER] Output shape: {x.shape}")

        return x


# =============================================================================
# Training Functions
# =============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        radar = batch['radar'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(radar)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            radar = batch['radar'].to(device)
            labels = batch['label'].to(device)

            logits = model(radar)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train radar activity recognition classifier')
    parser.add_argument('--train-dataset', type=str, required=True, help='Path to training dataset directory')
    parser.add_argument('--test-dataset', type=str, default=None, help='Path to test dataset directory (optional)')
    parser.add_argument('--split-percent', type=float, default=0.8, help='Train/test split percent (default 0.8)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num-radar-frames', type=int, default=100, help='Number of radar frames')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: <script_dir>/results)')
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
    train_dataset = RadarDataset(
        dataset_dir=Path(args.train_dataset),
        num_radar_frames=args.num_radar_frames
    )

    if args.test_dataset:
        print(f"\nLoading test dataset from: {args.test_dataset}")
        test_dataset = RadarDataset(
            dataset_dir=Path(args.test_dataset),
            num_radar_frames=args.num_radar_frames
        )
    else:
        print(f"\nSplitting training dataset (train={args.split_percent*100}%, test={(1-args.split_percent)*100}%)")
        train_dataset, test_dataset = train_dataset.split_by_time(split_percent=args.split_percent)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model
    num_classes = len(train_dataset.label_map)
    class_names = list(train_dataset.label_map.keys())

    model = RadarClassifier(input_channels=3, num_classes=num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        print(f"\n  [TRAINING] Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  [TRAINING] Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    # Save model and metadata
    model_path = output_dir / 'radar_classifier.pth'
    torch.save(model.state_dict(), model_path)

    metadata = {
        'model_name': 'radar_classifier',
        'data_type': 'radar_video',
        'description': '3D CNN classifier for radar video activity recognition',
        'input_shape': [args.num_radar_frames, 3, 64, 64],
        'output_shape': [num_classes],
        'predictions': 'activity_class_labels',
        'labels': class_names,
        'label_map': train_dataset.label_map,
        'label_index': {str(i): name for i, name in enumerate(class_names)},
        'channels': ['rdi', 'azimuth_range', 'azimuth_doppler'],
        'architecture': {
            'type': '3D CNN',
            'layers': [32, 64, 128],
            'fc_layers': [256, 128]
        }
    }

    metadata_path = output_dir / 'radar_classifier.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Final validation accuracy: {val_acc:.2f}%")
    print(f"Model and metadata saved to: {output_dir}")
    print(f"  - radar_classifier.pth (model)")
    print(f"  - radar_classifier.json (metadata with labels)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
