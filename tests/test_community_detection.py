"""Test suite for community detection project."""

import pytest
import torch
import numpy as np

from src.models import CommunityDetectionModel
from src.data import generate_sbm_graph, generate_karate_club
from src.utils import set_seed, get_device, compute_modularity, compute_nmi, compute_ari


class TestModels:
    """Test GNN model implementations."""
    
    def test_gcn_model(self):
        """Test GCN model creation and forward pass."""
        model = CommunityDetectionModel(
            model_type="gcn",
            input_dim=16,
            hidden_dim=32,
            output_dim=16,
            n_communities=4,
            n_layers=2
        )
        
        # Create dummy data
        x = torch.randn(100, 16)
        edge_index = torch.randint(0, 100, (2, 200))
        
        # Forward pass
        embeddings, logits = model(x, edge_index)
        
        assert embeddings.shape == (100, 16)
        assert logits.shape == (100, 4)
    
    def test_sage_model(self):
        """Test GraphSAGE model creation and forward pass."""
        model = CommunityDetectionModel(
            model_type="sage",
            input_dim=16,
            hidden_dim=32,
            output_dim=16,
            n_communities=4,
            n_layers=2
        )
        
        # Create dummy data
        x = torch.randn(100, 16)
        edge_index = torch.randint(0, 100, (2, 200))
        
        # Forward pass
        embeddings, logits = model(x, edge_index)
        
        assert embeddings.shape == (100, 16)
        assert logits.shape == (100, 4)
    
    def test_gat_model(self):
        """Test GAT model creation and forward pass."""
        model = CommunityDetectionModel(
            model_type="gat",
            input_dim=16,
            hidden_dim=32,
            output_dim=16,
            n_communities=4,
            n_layers=2,
            n_heads=4
        )
        
        # Create dummy data
        x = torch.randn(100, 16)
        edge_index = torch.randint(0, 100, (2, 200))
        
        # Forward pass
        embeddings, logits = model(x, edge_index)
        
        assert embeddings.shape == (100, 16)
        assert logits.shape == (100, 4)
    
    def test_gin_model(self):
        """Test GIN model creation and forward pass."""
        model = CommunityDetectionModel(
            model_type="gin",
            input_dim=16,
            hidden_dim=32,
            output_dim=16,
            n_communities=4,
            n_layers=2
        )
        
        # Create dummy data
        x = torch.randn(100, 16)
        edge_index = torch.randint(0, 100, (2, 200))
        
        # Forward pass
        embeddings, logits = model(x, edge_index)
        
        assert embeddings.shape == (100, 16)
        assert logits.shape == (100, 4)


class TestDataGeneration:
    """Test data generation functions."""
    
    def test_sbm_generation(self):
        """Test SBM graph generation."""
        data, communities = generate_sbm_graph(
            n_nodes=100,
            n_communities=4,
            p_in=0.3,
            p_out=0.05,
            seed=42
        )
        
        assert data.num_nodes == 100
        assert data.x.shape[0] == 100
        assert data.edge_index.shape[1] > 0
        assert len(communities) == 100
        assert len(torch.unique(communities)) == 4
    
    def test_karate_club_generation(self):
        """Test Karate Club graph generation."""
        data, communities = generate_karate_club()
        
        assert data.num_nodes == 34
        assert data.x.shape[0] == 34
        assert data.edge_index.shape[1] > 0
        assert len(communities) == 34
        assert len(torch.unique(communities)) == 2


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        rand1 = torch.randn(10)
        
        set_seed(42)
        rand2 = torch.randn(10)
        
        assert torch.allclose(rand1, rand2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_compute_modularity(self):
        """Test modularity computation."""
        # Create simple adjacency matrix
        adj_matrix = torch.tensor([
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 0]
        ], dtype=torch.float)
        
        communities = torch.tensor([0, 0, 1, 1])
        
        modularity = compute_modularity(adj_matrix, communities)
        assert isinstance(modularity, float)
        assert -1 <= modularity <= 1
    
    def test_compute_nmi(self):
        """Test NMI computation."""
        true_communities = torch.tensor([0, 0, 1, 1])
        pred_communities = torch.tensor([0, 1, 1, 1])
        
        nmi = compute_nmi(true_communities, pred_communities)
        assert isinstance(nmi, float)
        assert 0 <= nmi <= 1
    
    def test_compute_ari(self):
        """Test ARI computation."""
        true_communities = torch.tensor([0, 0, 1, 1])
        pred_communities = torch.tensor([0, 1, 1, 1])
        
        ari = compute_ari(true_communities, pred_communities)
        assert isinstance(ari, float)
        assert -1 <= ari <= 1


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline."""
        # Generate data
        data, true_communities = generate_sbm_graph(
            n_nodes=50,
            n_communities=2,
            p_in=0.5,
            p_out=0.1,
            seed=42
        )
        
        # Create model
        model = CommunityDetectionModel(
            model_type="gcn",
            input_dim=data.x.size(1),
            hidden_dim=16,
            output_dim=8,
            n_communities=2,
            n_layers=2
        )
        
        # Create trainer
        from src.train import CommunityDetectionTrainer
        trainer = CommunityDetectionTrainer(model, learning_rate=0.01)
        
        # Create data splits
        from src.data import create_train_val_test_split
        train_mask, val_mask, test_mask = create_train_val_test_split(
            data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
        )
        
        # Train for a few epochs
        history = trainer.train(
            data=data,
            train_mask=train_mask,
            val_mask=val_mask,
            n_epochs=5,
            verbose=False
        )
        
        # Check training history
        assert len(history['train_losses']) == 5
        assert len(history['val_losses']) == 5
        assert len(history['train_accuracies']) == 5
        assert len(history['val_accuracies']) == 5
        
        # Evaluate
        test_metrics = trainer.evaluate(data, test_mask)
        assert 'accuracy' in test_metrics
        assert 'nmi' in test_metrics
        assert 'ari' in test_metrics


if __name__ == "__main__":
    pytest.main([__file__])
