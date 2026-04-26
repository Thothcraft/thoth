"""Local Trainer for FL Client.

This module provides local training functionality for the FL client,
handling model training on local data with support for various
FL algorithms.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Callable

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import TrainingConfig, ModelConfig, FLAlgorithm

logger = logging.getLogger(__name__)


class LocalTrainer:
    """Handles local model training for FL.
    
    This trainer supports:
    - Standard local training (FedAvg)
    - Proximal term training (FedProx)
    - Gradient clipping for DP
    """
    
    def __init__(
        self,
        model: nn.Module,
        training_config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        """Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            training_config: Training configuration
            device: Device to train on
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training")
        
        self.model = model
        self.config = training_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # For FedProx
        self._global_params: Optional[List[torch.Tensor]] = None
        self._proximal_mu: float = 0.01
        
        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        params = self.model.parameters()
        
        if self.config.optimizer.lower() == "adam":
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
    
    def set_global_params(self, params: List[np.ndarray]):
        """Set global model parameters for FedProx.
        
        Args:
            params: List of parameter arrays from server
        """
        self._global_params = [
            torch.tensor(p, device=self.device) for p in params
        ]
    
    def set_proximal_mu(self, mu: float):
        """Set proximal term coefficient for FedProx.
        
        Args:
            mu: Proximal coefficient
        """
        self._proximal_mu = mu
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
        use_proximal: bool = False,
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> Dict[str, Any]:
        """Train the model locally.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (overrides config)
            use_proximal: Whether to use FedProx proximal term
            callback: Optional callback(epoch, metrics)
        
        Returns:
            Training results dictionary
        """
        num_epochs = num_epochs or self.config.local_epochs
        start_time = time.time()
        
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                train_loader,
                use_proximal=use_proximal,
            )
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            
            # Validation phase
            if val_loader:
                val_loss, val_acc = self._evaluate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_acc)
            else:
                val_loss, val_acc = 0.0, 0.0
            
            # Callback
            if callback:
                callback(epoch + 1, {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                })
            
            logger.debug(f"Epoch {epoch+1}/{num_epochs}: "
                        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                        f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        training_time = time.time() - start_time
        
        return {
            "num_epochs": num_epochs,
            "training_time": training_time,
            "final_train_loss": self.history["train_loss"][-1],
            "final_train_accuracy": self.history["train_accuracy"][-1],
            "final_val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else None,
            "final_val_accuracy": self.history["val_accuracy"][-1] if self.history["val_accuracy"] else None,
            "history": self.history,
        }
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        use_proximal: bool = False,
    ) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            use_proximal: Whether to use FedProx proximal term
        
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Add proximal term for FedProx
            if use_proximal and self._global_params is not None:
                proximal_term = 0.0
                for local_param, global_param in zip(
                    self.model.parameters(), self._global_params
                ):
                    proximal_term += torch.sum((local_param - global_param) ** 2)
                loss += (self._proximal_mu / 2) * proximal_term
            
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clipping
                )
            
            self.optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate the model.
        
        Args:
            data_loader: Data loader
        
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as numpy arrays.
        
        Returns:
            List of parameter arrays
        """
        return [
            param.cpu().detach().numpy()
            for param in self.model.parameters()
        ]
    
    def set_parameters(self, params: List[np.ndarray]):
        """Set model parameters from numpy arrays.
        
        Args:
            params: List of parameter arrays
        """
        with torch.no_grad():
            for local_param, new_param in zip(self.model.parameters(), params):
                local_param.copy_(torch.tensor(new_param, device=self.device))
    
    def get_num_samples(self, data_loader: DataLoader) -> int:
        """Get number of samples in data loader.
        
        Args:
            data_loader: Data loader
        
        Returns:
            Number of samples
        """
        return len(data_loader.dataset)
