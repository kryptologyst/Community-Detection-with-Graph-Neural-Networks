"""Utility functions for community detection project."""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        Available torch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Loaded configuration.
    """
    from omegaconf import OmegaConf
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save.
        config_path: Path to save configuration.
    """
    from omegaconf import OmegaConf
    
    OmegaConf.save(config, config_path)


def create_logger(name: str, level: str = "INFO") -> Any:
    """Create a logger instance.
    
    Args:
        name: Logger name.
        level: Logging level.
        
    Returns:
        Logger instance.
    """
    import logging
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """Normalize node features.
    
    Args:
        features: Node features tensor.
        
    Returns:
        Normalized features.
    """
    return torch.nn.functional.normalize(features, p=2, dim=1)


def compute_modularity(
    adj_matrix: torch.Tensor, 
    communities: torch.Tensor
) -> float:
    """Compute modularity of a community partition.
    
    Args:
        adj_matrix: Adjacency matrix.
        communities: Community assignments for each node.
        
    Returns:
        Modularity value.
    """
    n_nodes = adj_matrix.size(0)
    m = adj_matrix.sum().item() / 2  # Number of edges
    
    if m == 0:
        return 0.0
    
    modularity = 0.0
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if communities[i] == communities[j]:
                modularity += adj_matrix[i, j].item() - (
                    adj_matrix[i, :].sum().item() * 
                    adj_matrix[j, :].sum().item()
                ) / (2 * m)
    
    return modularity / (2 * m)


def compute_nmi(communities_true: torch.Tensor, communities_pred: torch.Tensor) -> float:
    """Compute Normalized Mutual Information between true and predicted communities.
    
    Args:
        communities_true: True community assignments.
        communities_pred: Predicted community assignments.
        
    Returns:
        NMI value.
    """
    from sklearn.metrics import normalized_mutual_info_score
    
    return normalized_mutual_info_score(
        communities_true.cpu().numpy(), 
        communities_pred.cpu().numpy()
    )


def compute_ari(communities_true: torch.Tensor, communities_pred: torch.Tensor) -> float:
    """Compute Adjusted Rand Index between true and predicted communities.
    
    Args:
        communities_true: True community assignments.
        communities_pred: Predicted community assignments.
        
    Returns:
        ARI value.
    """
    from sklearn.metrics import adjusted_rand_score
    
    return adjusted_rand_score(
        communities_true.cpu().numpy(), 
        communities_pred.cpu().numpy()
    )


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(
        self, 
        patience: int = 10, 
        min_delta: float = 0.0, 
        restore_best_weights: bool = True
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as an improvement.
            restore_best_weights: Whether to restore best weights.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score.
            model: Model to potentially restore weights.
            
        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
            
        return False
    
    def save_checkpoint(self, model: nn.Module) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save.
        """
        self.best_weights = model.state_dict().copy()
