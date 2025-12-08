"""Training utilities for community detection models."""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

from ..utils import EarlyStopping, get_device, create_logger
from ..data import create_train_val_test_split


class CommunityDetectionTrainer:
    """Trainer for community detection models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4,
        patience: int = 20,
        min_delta: float = 0.001
    ):
        """Initialize trainer.
        
        Args:
            model: Community detection model.
            device: Device to train on.
            learning_rate: Learning rate.
            weight_decay: Weight decay for optimizer.
            patience: Early stopping patience.
            min_delta: Early stopping minimum delta.
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        
        self.logger = create_logger("CommunityDetectionTrainer")
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, data: Data, train_mask: torch.Tensor) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            data: Graph data.
            train_mask: Training node mask.
            
        Returns:
            Training loss and accuracy.
        """
        self.model.train()
        
        # Move data to device
        data = data.to(self.device)
        train_mask = train_mask.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        embeddings, logits = self.model(data.x, data.edge_index)
        
        # Compute loss only on training nodes
        loss = self.criterion(logits[train_mask], data.y[train_mask])
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            pred = logits[train_mask].argmax(dim=1)
            accuracy = (pred == data.y[train_mask]).float().mean().item()
        
        return loss.item(), accuracy
    
    def validate(self, data: Data, val_mask: torch.Tensor) -> Tuple[float, float]:
        """Validate the model.
        
        Args:
            data: Graph data.
            val_mask: Validation node mask.
            
        Returns:
            Validation loss and accuracy.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            data = data.to(self.device)
            val_mask = val_mask.to(self.device)
            
            # Forward pass
            embeddings, logits = self.model(data.x, data.edge_index)
            
            # Compute loss
            loss = self.criterion(logits[val_mask], data.y[val_mask])
            
            # Compute accuracy
            pred = logits[val_mask].argmax(dim=1)
            accuracy = (pred == data.y[val_mask]).float().mean().item()
        
        return loss.item(), accuracy
    
    def train(
        self,
        data: Data,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        n_epochs: int = 200,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            data: Graph data.
            train_mask: Training node mask.
            val_mask: Validation node mask.
            n_epochs: Number of training epochs.
            verbose: Whether to show progress.
            
        Returns:
            Training history.
        """
        self.logger.info(f"Starting training for {n_epochs} epochs")
        
        pbar = tqdm(range(n_epochs), disable=not verbose)
        
        for epoch in pbar:
            # Training
            train_loss, train_acc = self.train_epoch(data, train_mask)
            
            # Validation
            val_loss, val_acc = self.validate(data, val_mask)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Update progress bar
            pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'train_acc': f'{train_acc:.4f}',
                'val_acc': f'{val_acc:.4f}'
            })
            
            # Early stopping
            if self.early_stopping(val_acc, self.model):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def evaluate(
        self,
        data: Data,
        test_mask: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate the model on test set.
        
        Args:
            data: Graph data.
            test_mask: Test node mask.
            
        Returns:
            Evaluation metrics.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            data = data.to(self.device)
            test_mask = test_mask.to(self.device)
            
            # Forward pass
            embeddings, logits = self.model(data.x, data.edge_index)
            
            # Predictions
            pred = logits[test_mask].argmax(dim=1)
            true_labels = data.y[test_mask]
            
            # Compute metrics
            accuracy = (pred == true_labels).float().mean().item()
            
            # Additional metrics
            from sklearn.metrics import (
                normalized_mutual_info_score,
                adjusted_rand_score,
                f1_score
            )
            
            nmi = normalized_mutual_info_score(
                true_labels.cpu().numpy(),
                pred.cpu().numpy()
            )
            
            ari = adjusted_rand_score(
                true_labels.cpu().numpy(),
                pred.cpu().numpy()
            )
            
            f1_macro = f1_score(
                true_labels.cpu().numpy(),
                pred.cpu().numpy(),
                average='macro'
            )
            
            f1_micro = f1_score(
                true_labels.cpu().numpy(),
                pred.cpu().numpy(),
                average='micro'
            )
        
        metrics = {
            'accuracy': accuracy,
            'nmi': nmi,
            'ari': ari,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro
        }
        
        self.logger.info(f"Test metrics: {metrics}")
        
        return metrics
    
    def save_model(self, path: str) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save model.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to load model from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        self.logger.info(f"Model loaded from {path}")


def train_model(
    model: nn.Module,
    data: Data,
    config: Dict,
    save_path: Optional[str] = None
) -> Tuple[CommunityDetectionTrainer, Dict[str, List[float]]]:
    """Train a community detection model.
    
    Args:
        model: Community detection model.
        data: Graph data.
        config: Training configuration.
        save_path: Path to save model.
        
    Returns:
        Trained trainer and training history.
    """
    # Create data splits
    train_mask, val_mask, test_mask = create_train_val_test_split(
        data,
        train_ratio=config.get('train_ratio', 0.6),
        val_ratio=config.get('val_ratio', 0.2),
        test_ratio=config.get('test_ratio', 0.2),
        seed=config.get('seed', 42)
    )
    
    # Initialize trainer
    trainer = CommunityDetectionTrainer(
        model=model,
        learning_rate=config.get('learning_rate', 0.01),
        weight_decay=config.get('weight_decay', 5e-4),
        patience=config.get('patience', 20)
    )
    
    # Train model
    history = trainer.train(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        n_epochs=config.get('n_epochs', 200),
        verbose=config.get('verbose', True)
    )
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(data, test_mask)
    
    # Save model if path provided
    if save_path:
        trainer.save_model(save_path)
    
    return trainer, history
