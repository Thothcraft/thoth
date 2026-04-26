"""Federated Learning Client Module for Thoth Device.

This module provides FL client functionality that integrates with Flower (flwr)
to participate in federated learning training sessions. The client can:

- Connect to a Flower server when online
- Train on local data stored in thoth/data/
- Download assigned training data from the server
- Report training metrics back to the server

The client automatically discovers and uses data files with metadata
that are marked as available for FL training.

Usage:
    from thoth.src.fl_client import FLClient, FLClientConfig
    
    # Create client
    config = FLClientConfig(
        server_url="http://brain-server:8080",
        device_id="thoth-001",
    )
    client = FLClient(config)
    
    # Start FL participation
    client.start()
"""

from .config import FLClientConfig, FLAlgorithm, ModelConfig
from .client import FLClient, ThothFlowerClient
from .data_loader import FLDataLoader
from .trainer import LocalTrainer

__all__ = [
    "FLClientConfig",
    "FLAlgorithm",
    "ModelConfig",
    "FLClient",
    "ThothFlowerClient",
    "FLDataLoader",
    "LocalTrainer",
]
