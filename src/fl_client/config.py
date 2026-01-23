"""Configuration for FL Client.

This module defines configuration dataclasses for the FL client,
including server connection, model, and training parameters.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class FLAlgorithm(str, Enum):
    """Supported FL algorithms."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDADAM = "fedadam"
    FEDYOGI = "fedyogi"
    FEDADAGRAD = "fedadagrad"
    FEDAVGM = "fedavgm"
    FEDMEDIAN = "fedmedian"
    KRUM = "krum"
    QFEDAVG = "qfedavg"
    FEDDF = "feddf"
    FEDMD = "fedmd"


class ModelArchitecture(str, Enum):
    """Supported model architectures."""
    CNN = "cnn"
    RESNET18 = "resnet18"
    MOBILENET_V2 = "mobilenet_v2"
    LSTM = "lstm"
    GRU = "gru"
    CNN_LSTM = "cnn_lstm"
    TCN = "tcn"
    MLP = "mlp"
    # DL models from dl_models/
    LSTM_NANO = "lstm_nano"
    LSTM_MINI = "lstm_mini"
    LSTM_MAX = "lstm_max"
    GRU_NANO = "gru_nano"
    GRU_MINI = "gru_mini"
    CNN1D_NANO = "cnn1d_nano"
    CNN1D_MINI = "cnn1d_mini"
    TRANSFORMER_NANO = "transformer_nano"
    TRANSFORMER_MINI = "transformer_mini"


@dataclass
class ModelConfig:
    """Model configuration for FL training."""
    architecture: str = "lstm_mini"
    num_classes: int = 4
    input_channels: int = 6
    seq_length: int = 128
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """Local training configuration."""
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    lr_scheduler: Optional[str] = None
    gradient_clipping: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    data_type: str = "imu"
    pipeline: Optional[List[Dict[str, Any]]] = None
    normalize: bool = True
    window_size: int = 128
    step_size: int = 64
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FLClientConfig:
    """Complete FL client configuration."""
    # Server connection
    server_url: str = "http://localhost:8080"
    server_address: str = "localhost:8080"  # For Flower gRPC
    use_grpc: bool = True
    
    # Device identification
    device_id: str = ""
    
    # Data configuration
    data_dir: str = ""
    sensor_types: List[str] = field(default_factory=lambda: ["imu", "csi"])
    min_samples_for_training: int = 100
    
    # Model configuration
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Training configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Preprocessing configuration
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    
    # FL algorithm (set by server)
    algorithm: str = "fedavg"
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    
    # Connection settings
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    heartbeat_interval_seconds: float = 30.0
    
    # Privacy settings
    differential_privacy: bool = False
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    
    def __post_init__(self):
        """Initialize default values."""
        if not self.device_id:
            self.device_id = self._load_device_id()
        if not self.data_dir:
            self.data_dir = str(Path(__file__).parent.parent.parent / "data")
    
    def _load_device_id(self) -> str:
        """Load device ID from file or generate new one."""
        data_dir = Path(__file__).parent.parent.parent / "data"
        device_id_file = data_dir / "device_id.txt"
        
        if device_id_file.exists():
            try:
                return device_id_file.read_text().strip()
            except Exception:
                pass
        
        # Generate new device ID
        import uuid
        device_id = f"thoth-{str(uuid.uuid4())[:8]}"
        
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            device_id_file.write_text(device_id)
        except Exception as e:
            logger.warning(f"Failed to save device_id: {e}")
        
        return device_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "server_url": self.server_url,
            "server_address": self.server_address,
            "use_grpc": self.use_grpc,
            "device_id": self.device_id,
            "data_dir": self.data_dir,
            "sensor_types": self.sensor_types,
            "min_samples_for_training": self.min_samples_for_training,
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
            "algorithm": self.algorithm,
            "algorithm_params": self.algorithm_params,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "differential_privacy": self.differential_privacy,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FLClientConfig":
        """Create from dictionary."""
        config = cls()
        
        for key, value in data.items():
            if key == "model":
                config.model = ModelConfig.from_dict(value)
            elif key == "training":
                config.training = TrainingConfig.from_dict(value)
            elif key == "preprocessing":
                config.preprocessing = PreprocessingConfig(**value)
            elif hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def save(self, path: Path) -> bool:
        """Save configuration to file."""
        try:
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    @classmethod
    def load(cls, path: Path) -> Optional["FLClientConfig"]:
        """Load configuration from file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return None
