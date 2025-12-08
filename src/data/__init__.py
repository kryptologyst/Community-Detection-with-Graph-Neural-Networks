"""Data utilities and synthetic graph generation for community detection."""

import os
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx


def generate_sbm_graph(
    n_nodes: int = 1000,
    n_communities: int = 4,
    p_in: float = 0.3,
    p_out: float = 0.05,
    seed: int = 42
) -> Tuple[Data, torch.Tensor]:
    """Generate a Stochastic Block Model (SBM) graph for community detection.
    
    Args:
        n_nodes: Number of nodes in the graph.
        n_communities: Number of communities.
        p_in: Probability of edge within communities.
        p_out: Probability of edge between communities.
        seed: Random seed.
        
    Returns:
        PyTorch Geometric Data object and true community labels.
    """
    np.random.seed(seed)
    
    # Generate community assignments
    community_sizes = np.random.multinomial(
        n_nodes, 
        np.ones(n_communities) / n_communities
    )
    
    communities = []
    for i, size in enumerate(community_sizes):
        communities.extend([i] * size)
    
    communities = np.array(communities)
    np.random.shuffle(communities)
    
    # Generate adjacency matrix
    adj_matrix = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if communities[i] == communities[j]:
                prob = p_in
            else:
                prob = p_out
            
            if np.random.random() < prob:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
    
    # Convert to PyTorch Geometric format
    edge_index = torch.from_numpy(np.array(np.where(adj_matrix))).long()
    
    # Generate random node features
    node_features = torch.randn(n_nodes, 16)
    
    # Create Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        num_nodes=n_nodes
    )
    
    # Add true community labels
    data.y = torch.from_numpy(communities).long()
    
    return data, torch.from_numpy(communities).long()


def generate_karate_club() -> Tuple[Data, torch.Tensor]:
    """Generate Zachary's Karate Club graph.
    
    Returns:
        PyTorch Geometric Data object and true community labels.
    """
    G = nx.karate_club_graph()
    
    # Convert to PyTorch Geometric format
    data = from_networkx(G)
    
    # Add random node features
    data.x = torch.randn(data.num_nodes, 8)
    
    # True communities (based on the original study)
    communities = torch.zeros(data.num_nodes, dtype=torch.long)
    communities[0:17] = 0  # Mr. Hi's group
    communities[17:] = 1   # John A's group
    
    data.y = communities
    
    return data, communities


def generate_erdos_renyi_graph(
    n_nodes: int = 100,
    p: float = 0.1,
    seed: int = 42
) -> Data:
    """Generate an Erdos-Renyi random graph.
    
    Args:
        n_nodes: Number of nodes.
        p: Probability of edge creation.
        seed: Random seed.
        
    Returns:
        PyTorch Geometric Data object.
    """
    G = nx.erdos_renyi_graph(n_nodes, p, seed=seed)
    data = from_networkx(G)
    data.x = torch.randn(data.num_nodes, 8)
    
    return data


def generate_barabasi_albert_graph(
    n_nodes: int = 100,
    m: int = 3,
    seed: int = 42
) -> Data:
    """Generate a Barabasi-Albert scale-free graph.
    
    Args:
        n_nodes: Number of nodes.
        m: Number of edges to attach from a new node to existing nodes.
        seed: Random seed.
        
    Returns:
        PyTorch Geometric Data object.
    """
    G = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
    data = from_networkx(G)
    data.x = torch.randn(data.num_nodes, 8)
    
    return data


def load_graph_from_csv(
    nodes_path: str,
    edges_path: str,
    node_features: Optional[List[str]] = None
) -> Data:
    """Load graph from CSV files.
    
    Args:
        nodes_path: Path to nodes CSV file.
        edges_path: Path to edges CSV file.
        node_features: List of feature column names.
        
    Returns:
        PyTorch Geometric Data object.
    """
    # Load nodes
    nodes_df = pd.read_csv(nodes_path)
    n_nodes = len(nodes_df)
    
    # Load edges
    edges_df = pd.read_csv(edges_path)
    
    # Create edge index
    edge_index = torch.tensor([
        edges_df['src'].values,
        edges_df['dst'].values
    ], dtype=torch.long)
    
    # Create node features
    if node_features is None:
        node_features = [col for col in nodes_df.columns 
                        if col not in ['node_id', 'label', 'community']]
    
    if node_features:
        x = torch.tensor(nodes_df[node_features].values, dtype=torch.float)
    else:
        x = torch.randn(n_nodes, 8)
    
    # Create labels if available
    y = None
    if 'label' in nodes_df.columns:
        y = torch.tensor(nodes_df['label'].values, dtype=torch.long)
    elif 'community' in nodes_df.columns:
        y = torch.tensor(nodes_df['community'].values, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=n_nodes)


def save_graph_to_csv(
    data: Data,
    nodes_path: str,
    edges_path: str,
    save_features: bool = True
) -> None:
    """Save graph to CSV files.
    
    Args:
        data: PyTorch Geometric Data object.
        nodes_path: Path to save nodes CSV.
        edges_path: Path to save edges CSV.
        save_features: Whether to save node features.
    """
    # Save nodes
    nodes_data = {'node_id': range(data.num_nodes)}
    
    if save_features and data.x is not None:
        for i in range(data.x.size(1)):
            nodes_data[f'feature_{i}'] = data.x[:, i].numpy()
    
    if data.y is not None:
        nodes_data['community'] = data.y.numpy()
    
    nodes_df = pd.DataFrame(nodes_data)
    nodes_df.to_csv(nodes_path, index=False)
    
    # Save edges
    edges_data = {
        'src': data.edge_index[0].numpy(),
        'dst': data.edge_index[1].numpy()
    }
    edges_df = pd.DataFrame(edges_data)
    edges_df.to_csv(edges_path, index=False)


def create_train_val_test_split(
    data: Data,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create train/validation/test splits for node-level tasks.
    
    Args:
        data: PyTorch Geometric Data object.
        train_ratio: Ratio of nodes for training.
        val_ratio: Ratio of nodes for validation.
        test_ratio: Ratio of nodes for testing.
        seed: Random seed.
        
    Returns:
        Train, validation, and test masks.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    np.random.seed(seed)
    n_nodes = data.num_nodes
    
    # Create random permutation
    perm = np.random.permutation(n_nodes)
    
    # Calculate split sizes
    train_size = int(train_ratio * n_nodes)
    val_size = int(val_ratio * n_nodes)
    
    # Create masks
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    
    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size + val_size]] = True
    test_mask[perm[train_size + val_size:]] = True
    
    return train_mask, val_mask, test_mask


def add_self_loops(data: Data) -> Data:
    """Add self-loops to the graph.
    
    Args:
        data: PyTorch Geometric Data object.
        
    Returns:
        Data object with self-loops added.
    """
    from torch_geometric.utils import add_self_loops
    
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    return data


def normalize_adjacency(data: Data) -> Data:
    """Normalize adjacency matrix using symmetric normalization.
    
    Args:
        data: PyTorch Geometric Data object.
        
    Returns:
        Data object with normalized adjacency.
    """
    from torch_geometric.utils import get_laplacian
    
    edge_index, edge_weight = get_laplacian(
        data.edge_index, 
        normalization='sym',
        num_nodes=data.num_nodes
    )
    
    data.edge_index = edge_index
    data.edge_weight = edge_weight
    
    return data
