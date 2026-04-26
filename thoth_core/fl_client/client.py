"""FL Client Implementation using Flower.

This module provides the main FL client that integrates with Flower (flwr)
to participate in federated learning sessions.
"""

import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

try:
    import flwr as fl
    from flwr.client import NumPyClient
    from flwr.common import (
        Parameters,
        FitRes,
        EvaluateRes,
        GetParametersRes,
        Status,
        Code,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False
    NumPyClient = object  # Placeholder for type hints

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import FLClientConfig, ModelConfig
from .data_loader import FLDataLoader
from .trainer import LocalTrainer

logger = logging.getLogger(__name__)


def create_model(config: ModelConfig) -> nn.Module:
    """Create a model based on configuration.
    
    This function creates models from the dl_models registry or
    falls back to built-in architectures.
    
    Args:
        config: Model configuration
    
    Returns:
        PyTorch model
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    
    arch = config.architecture.lower()
    
    # Try to import from Brain's dl_models if available
    try:
        import sys
        brain_path = Path(__file__).parent.parent.parent.parent / "Brain" / "server"
        if str(brain_path) not in sys.path:
            sys.path.insert(0, str(brain_path))
        
        from dl_models import DLModelRegistry, ModelConfig as DLModelConfig
        
        dl_config = {
            "num_classes": config.num_classes,
            "input_channels": config.input_channels,
            "seq_length": config.seq_length,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
        }
        
        model = DLModelRegistry.create(arch, dl_config)
        if model:
            logger.info(f"Created model '{arch}' from DLModelRegistry")
            return model
    except ImportError:
        logger.debug("Could not import from Brain's dl_models")
    except Exception as e:
        logger.debug(f"DLModelRegistry creation failed: {e}")
    
    # Fall back to built-in models
    if arch in ["lstm", "lstm_mini"]:
        return _create_lstm_model(config)
    elif arch in ["gru", "gru_mini"]:
        return _create_gru_model(config)
    elif arch in ["cnn", "cnn1d", "cnn1d_mini"]:
        return _create_cnn1d_model(config)
    else:
        logger.warning(f"Unknown architecture '{arch}', defaulting to LSTM")
        return _create_lstm_model(config)


def _create_lstm_model(config: ModelConfig) -> nn.Module:
    """Create LSTM model."""
    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=config.input_channels,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0,
                bidirectional=True,
            )
            self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
            self.dropout = nn.Dropout(config.dropout)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            out = lstm_out[:, -1, :]
            out = self.dropout(out)
            return self.fc(out)
    
    return LSTMModel()


def _create_gru_model(config: ModelConfig) -> nn.Module:
    """Create GRU model."""
    class GRUModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                input_size=config.input_channels,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0,
                bidirectional=True,
            )
            self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
            self.dropout = nn.Dropout(config.dropout)
        
        def forward(self, x):
            gru_out, _ = self.gru(x)
            out = gru_out[:, -1, :]
            out = self.dropout(out)
            return self.fc(out)
    
    return GRUModel()


def _create_cnn1d_model(config: ModelConfig) -> nn.Module:
    """Create 1D CNN model."""
    class CNN1DModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(config.input_channels, 64, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
            self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(2)
            self.dropout = nn.Dropout(config.dropout)
            
            # Calculate output size after convolutions
            out_len = config.seq_length // 8  # 3 pooling layers
            self.fc = nn.Linear(128 * out_len, config.num_classes)
        
        def forward(self, x):
            # x: (batch, seq_len, channels) -> (batch, channels, seq_len)
            x = x.permute(0, 2, 1)
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            return self.fc(x)
    
    return CNN1DModel()


class ThothFlowerClient(NumPyClient):
    """Flower NumPyClient implementation for Thoth device.
    
    This client handles communication with the Flower server and
    coordinates local training using the LocalTrainer.
    """
    
    def __init__(
        self,
        config: FLClientConfig,
        model: nn.Module,
        data_loader: FLDataLoader,
    ):
        """Initialize the Flower client.
        
        Args:
            config: FL client configuration
            model: PyTorch model
            data_loader: Data loader for local data
        """
        self.config = config
        self.model = model
        self.data_loader = data_loader
        
        # Create trainer
        self.trainer = LocalTrainer(
            model=model,
            training_config=config.training,
        )
        
        # Load data
        self.train_loader, self.val_loader = data_loader.get_data_loaders(
            batch_size=config.training.batch_size,
        )
        
        self._current_round = 0
        
        logger.info(f"ThothFlowerClient initialized for device {config.device_id}")
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Get model parameters.
        
        Args:
            config: Configuration from server
        
        Returns:
            List of parameter arrays
        """
        return self.trainer.get_parameters()
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters.
        
        Args:
            parameters: List of parameter arrays from server
        """
        self.trainer.set_parameters(parameters)
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Train the model on local data.
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration from server
        
        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        self._current_round = config.get("current_round", self._current_round + 1)
        logger.info(f"Starting local training for round {self._current_round}")
        
        # Set global parameters
        self.set_parameters(parameters)
        
        # Check for FedProx
        use_proximal = config.get("proximal_mu", 0) > 0
        if use_proximal:
            self.trainer.set_global_params(parameters)
            self.trainer.set_proximal_mu(config.get("proximal_mu", 0.01))
        
        # Get training config from server
        local_epochs = config.get("local_epochs", self.config.training.local_epochs)
        
        # Train
        if self.train_loader is None:
            logger.warning("No training data available")
            return self.get_parameters({}), 0, {"error": "no_data"}
        
        results = self.trainer.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=local_epochs,
            use_proximal=use_proximal,
        )
        
        # Mark data as used
        self.data_loader.mark_data_used(self._current_round)
        
        # Get updated parameters
        updated_params = self.get_parameters({})
        num_samples = self.trainer.get_num_samples(self.train_loader)
        
        metrics = {
            "train_loss": results["final_train_loss"],
            "train_accuracy": results["final_train_accuracy"],
            "device_id": self.config.device_id,
            "round": self._current_round,
        }
        
        if results["final_val_accuracy"] is not None:
            metrics["val_accuracy"] = results["final_val_accuracy"]
            metrics["val_loss"] = results["final_val_loss"]
        
        logger.info(f"Round {self._current_round} complete: "
                   f"train_acc={results['final_train_accuracy']:.4f}")
        
        return updated_params, num_samples, metrics
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate the model on local data.
        
        Args:
            parameters: Model parameters from server
            config: Evaluation configuration
        
        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        self.set_parameters(parameters)
        
        if self.val_loader is None:
            return 0.0, 0, {"error": "no_data"}
        
        loss, accuracy = self.trainer._evaluate(self.val_loader)
        num_samples = self.trainer.get_num_samples(self.val_loader)
        
        metrics = {
            "accuracy": accuracy,
            "device_id": self.config.device_id,
        }
        
        return loss, num_samples, metrics


class FLClient:
    """Main FL Client class for Thoth device.
    
    This class manages the FL client lifecycle:
    - Checking server connectivity
    - Downloading configuration and data
    - Starting/stopping FL participation
    """
    
    def __init__(self, config: Optional[FLClientConfig] = None):
        """Initialize the FL client.
        
        Args:
            config: FL client configuration
        """
        if not FLWR_AVAILABLE:
            raise ImportError("Flower (flwr) is required. Install with: pip install flwr")
        
        self.config = config or FLClientConfig()
        self._running = False
        self._client_thread: Optional[threading.Thread] = None
        self._flower_client: Optional[ThothFlowerClient] = None
        
        # Initialize data loader
        self.data_loader = FLDataLoader(
            data_dir=self.config.data_dir,
            sensor_types=self.config.sensor_types,
            preprocessing_config=self.config.preprocessing.to_dict(),
        )
        
        logger.info(f"FLClient initialized: device_id={self.config.device_id}")
    
    def check_server(self) -> Dict[str, Any]:
        """Check if FL server is reachable.
        
        Returns:
            Dictionary with server status
        """
        import urllib.request
        import urllib.error
        
        try:
            url = f"{self.config.server_url}/health"
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=5) as response:
                return {
                    "reachable": True,
                    "status_code": response.status,
                }
        except urllib.error.URLError as e:
            return {
                "reachable": False,
                "error": str(e),
            }
        except Exception as e:
            return {
                "reachable": False,
                "error": str(e),
            }
    
    def check_data_availability(self) -> Dict[str, Any]:
        """Check if local data is available for training.
        
        Returns:
            Data availability information
        """
        discovery = self.data_loader.discover_data()
        
        has_enough_data = discovery["total_samples"] >= self.config.min_samples_for_training
        
        return {
            "available": has_enough_data,
            "total_files": discovery["total_files"],
            "total_samples": discovery["total_samples"],
            "min_required": self.config.min_samples_for_training,
            "by_sensor_type": discovery["by_sensor_type"],
            "by_activity": discovery["by_activity"],
        }
    
    def start(self, blocking: bool = True) -> bool:
        """Start FL client participation.
        
        Args:
            blocking: Whether to block until FL completes
        
        Returns:
            True if started successfully
        """
        if self._running:
            logger.warning("FL client already running")
            return False
        
        # Check data availability
        data_status = self.check_data_availability()
        if not data_status["available"]:
            logger.error(f"Insufficient data for FL training: "
                        f"{data_status['total_samples']} < {data_status['min_required']}")
            return False
        
        # Create model
        model = create_model(self.config.model)
        
        # Create Flower client
        self._flower_client = ThothFlowerClient(
            config=self.config,
            model=model,
            data_loader=self.data_loader,
        )
        
        self._running = True
        
        if blocking:
            self._run_client()
        else:
            self._client_thread = threading.Thread(
                target=self._run_client,
                daemon=True,
            )
            self._client_thread.start()
        
        return True
    
    def _run_client(self):
        """Run the Flower client."""
        try:
            logger.info(f"Connecting to FL server at {self.config.server_address}")
            
            fl.client.start_numpy_client(
                server_address=self.config.server_address,
                client=self._flower_client,
            )
            
            logger.info("FL client finished")
        except Exception as e:
            logger.error(f"FL client error: {e}")
        finally:
            self._running = False
    
    def stop(self):
        """Stop FL client participation."""
        self._running = False
        logger.info("FL client stop requested")
    
    def is_running(self) -> bool:
        """Check if FL client is running."""
        return self._running
    
    def get_status(self) -> Dict[str, Any]:
        """Get FL client status.
        
        Returns:
            Status dictionary
        """
        return {
            "running": self._running,
            "device_id": self.config.device_id,
            "server_address": self.config.server_address,
            "algorithm": self.config.algorithm,
            "data_available": self.check_data_availability(),
        }
