"""Graph Neural Network models for community detection."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import to_dense_batch


class GCNCommunityDetector(nn.Module):
    """Graph Convolutional Network for community detection."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 16,
        n_communities: int = 4,
        n_layers: int = 2,
        dropout: float = 0.5,
        use_batch_norm: bool = True
    ):
        """Initialize GCN community detector.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output embedding dimension.
            n_communities: Number of communities to detect.
            n_layers: Number of GCN layers.
            dropout: Dropout rate.
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_communities = n_communities
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        if n_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # Community classification head
        self.classifier = nn.Linear(output_dim, n_communities)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Node embeddings and community predictions.
        """
        # GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            if i < len(self.convs) - 1:  # Don't apply activation to last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Community predictions
        community_logits = self.classifier(x)
        
        return x, community_logits


class SAGECommunityDetector(nn.Module):
    """GraphSAGE for community detection."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 16,
        n_communities: int = 4,
        n_layers: int = 2,
        dropout: float = 0.5,
        aggregator: str = 'mean'
    ):
        """Initialize GraphSAGE community detector.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output embedding dimension.
            n_communities: Number of communities to detect.
            n_layers: Number of SAGE layers.
            dropout: Dropout rate.
            aggregator: Aggregation method ('mean', 'max', 'lstm').
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_communities = n_communities
        self.n_layers = n_layers
        self.dropout = dropout
        self.aggregator = aggregator
        
        # SAGE layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=aggregator))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
        
        # Output layer
        if n_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, output_dim, aggr=aggregator))
        
        # Community classification head
        self.classifier = nn.Linear(output_dim, n_communities)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Node embeddings and community predictions.
        """
        # SAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < len(self.convs) - 1:  # Don't apply activation to last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Community predictions
        community_logits = self.classifier(x)
        
        return x, community_logits


class GATCommunityDetector(nn.Module):
    """Graph Attention Network for community detection."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 16,
        n_communities: int = 4,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.5,
        concat: bool = True
    ):
        """Initialize GAT community detector.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output embedding dimension.
            n_communities: Number of communities to detect.
            n_layers: Number of GAT layers.
            n_heads: Number of attention heads.
            dropout: Dropout rate.
            concat: Whether to concatenate attention heads.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_communities = n_communities
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.concat = concat
        
        # Calculate dimensions
        if concat:
            gat_out_dim = hidden_dim * n_heads
        else:
            gat_out_dim = hidden_dim
        
        # GAT layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(
            input_dim, hidden_dim, heads=n_heads, 
            dropout=dropout, concat=concat
        ))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            self.convs.append(GATConv(
                gat_out_dim, hidden_dim, heads=n_heads,
                dropout=dropout, concat=concat
            ))
        
        # Output layer
        if n_layers > 1:
            self.convs.append(GATConv(
                gat_out_dim, output_dim, heads=1,
                dropout=dropout, concat=False
            ))
        
        # Community classification head
        self.classifier = nn.Linear(output_dim, n_communities)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Node embeddings and community predictions.
        """
        # GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < len(self.convs) - 1:  # Don't apply activation to last layer
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Community predictions
        community_logits = self.classifier(x)
        
        return x, community_logits


class GINCommunityDetector(nn.Module):
    """Graph Isomorphism Network for community detection."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 16,
        n_communities: int = 4,
        n_layers: int = 2,
        dropout: float = 0.5,
        eps: float = 0.0
    ):
        """Initialize GIN community detector.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output embedding dimension.
            n_communities: Number of communities to detect.
            n_layers: Number of GIN layers.
            dropout: Dropout rate.
            eps: Epsilon parameter for GIN.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_communities = n_communities
        self.n_layers = n_layers
        self.dropout = dropout
        self.eps = eps
        
        # MLPs for GIN layers
        self.mlps = nn.ModuleList()
        self.convs = nn.ModuleList()
        
        # First layer
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mlps.append(mlp)
        self.convs.append(GINConv(mlp, eps=eps))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.mlps.append(mlp)
            self.convs.append(GINConv(mlp, eps=eps))
        
        # Output layer
        if n_layers > 1:
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            )
            self.mlps.append(mlp)
            self.convs.append(GINConv(mlp, eps=eps))
        
        # Community classification head
        self.classifier = nn.Linear(output_dim, n_communities)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Node embeddings and community predictions.
        """
        # GIN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < len(self.convs) - 1:  # Don't apply activation to last layer
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Community predictions
        community_logits = self.classifier(x)
        
        return x, community_logits


class CommunityDetectionModel(nn.Module):
    """Unified community detection model with multiple GNN backends."""
    
    def __init__(
        self,
        model_type: str,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 16,
        n_communities: int = 4,
        n_layers: int = 2,
        dropout: float = 0.5,
        **kwargs
    ):
        """Initialize community detection model.
        
        Args:
            model_type: Type of GNN ('gcn', 'sage', 'gat', 'gin').
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output embedding dimension.
            n_communities: Number of communities to detect.
            n_layers: Number of layers.
            dropout: Dropout rate.
            **kwargs: Additional model-specific parameters.
        """
        super().__init__()
        
        self.model_type = model_type.lower()
        
        if self.model_type == 'gcn':
            self.model = GCNCommunityDetector(
                input_dim, hidden_dim, output_dim, n_communities,
                n_layers, dropout, **kwargs
            )
        elif self.model_type == 'sage':
            self.model = SAGECommunityDetector(
                input_dim, hidden_dim, output_dim, n_communities,
                n_layers, dropout, **kwargs
            )
        elif self.model_type == 'gat':
            self.model = GATCommunityDetector(
                input_dim, hidden_dim, output_dim, n_communities,
                n_layers, dropout, **kwargs
            )
        elif self.model_type == 'gin':
            self.model = GINCommunityDetector(
                input_dim, hidden_dim, output_dim, n_communities,
                n_layers, dropout, **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Node embeddings and community predictions.
        """
        return self.model(x, edge_index)
    
    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor) -> Optional[torch.Tensor]:
        """Get attention weights for GAT models.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Attention weights if model is GAT, None otherwise.
        """
        if self.model_type == 'gat':
            # This would require modifying GAT to return attention weights
            # For now, return None
            return None
        return None
