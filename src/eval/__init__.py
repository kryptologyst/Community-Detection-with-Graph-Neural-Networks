"""Evaluation utilities for community detection models."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    f1_score,
    confusion_matrix
)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import compute_modularity, compute_nmi, compute_ari


class CommunityDetectionEvaluator:
    """Evaluator for community detection models."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics_history = []
    
    def evaluate_model(
        self,
        model: nn.Module,
        data: torch.Tensor,
        true_communities: torch.Tensor,
        test_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, float]:
        """Evaluate a community detection model.
        
        Args:
            model: Trained model.
            data: Graph data.
            true_communities: True community assignments.
            test_mask: Test node mask.
            device: Device to run evaluation on.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model.eval()
        
        with torch.no_grad():
            # Move data to device
            data = data.to(device)
            true_communities = true_communities.to(device)
            
            # Get predictions
            embeddings, logits = model(data.x, data.edge_index)
            pred_communities = logits.argmax(dim=1)
            
            # Use test mask if provided
            if test_mask is not None:
                test_mask = test_mask.to(device)
                true_communities = true_communities[test_mask]
                pred_communities = pred_communities[test_mask]
            
            # Convert to numpy for sklearn metrics
            true_np = true_communities.cpu().numpy()
            pred_np = pred_communities.cpu().numpy()
            
            # Compute metrics
            metrics = {
                'accuracy': accuracy_score(true_np, pred_np),
                'nmi': normalized_mutual_info_score(true_np, pred_np),
                'ari': adjusted_rand_score(true_np, pred_np),
                'f1_macro': f1_score(true_np, pred_np, average='macro'),
                'f1_micro': f1_score(true_np, pred_np, average='micro'),
                'f1_weighted': f1_score(true_np, pred_np, average='weighted')
            }
            
            # Compute modularity if adjacency matrix available
            if hasattr(data, 'edge_index'):
                adj_matrix = self._edge_index_to_adj_matrix(data.edge_index, data.num_nodes)
                metrics['modularity'] = compute_modularity(adj_matrix, pred_communities)
        
        return metrics
    
    def evaluate_baseline_methods(
        self,
        data: torch.Tensor,
        true_communities: torch.Tensor,
        test_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate baseline community detection methods.
        
        Args:
            data: Graph data.
            true_communities: True community assignments.
            test_mask: Test node mask.
            
        Returns:
            Dictionary of baseline method results.
        """
        results = {}
        
        # Convert to numpy
        true_np = true_communities.cpu().numpy()
        if test_mask is not None:
            true_np = true_np[test_mask.cpu().numpy()]
        
        # K-Means clustering on node features
        if hasattr(data, 'x') and data.x is not None:
            features = data.x.cpu().numpy()
            if test_mask is not None:
                features = features[test_mask.cpu().numpy()]
            
            n_communities = len(np.unique(true_np))
            kmeans = KMeans(n_clusters=n_communities, random_state=42)
            kmeans_pred = kmeans.fit_predict(features)
            
            results['kmeans'] = {
                'accuracy': accuracy_score(true_np, kmeans_pred),
                'nmi': normalized_mutual_info_score(true_np, kmeans_pred),
                'ari': adjusted_rand_score(true_np, kmeans_pred),
                'f1_macro': f1_score(true_np, kmeans_pred, average='macro'),
                'f1_micro': f1_score(true_np, kmeans_pred, average='micro')
            }
        
        # Louvain algorithm
        try:
            import networkx as nx
            import community as community_louvain
            
            # Convert to NetworkX graph
            G = self._to_networkx(data)
            
            # Apply Louvain
            louvain_partition = community_louvain.best_partition(G)
            louvain_pred = np.array([louvain_partition[i] for i in range(len(true_np))])
            
            # Align community labels
            louvain_pred = self._align_labels(true_np, louvain_pred)
            
            results['louvain'] = {
                'accuracy': accuracy_score(true_np, louvain_pred),
                'nmi': normalized_mutual_info_score(true_np, louvain_pred),
                'ari': adjusted_rand_score(true_np, louvain_pred),
                'f1_macro': f1_score(true_np, louvain_pred, average='macro'),
                'f1_micro': f1_score(true_np, louvain_pred, average='micro')
            }
            
        except ImportError:
            print("Warning: python-louvain not available, skipping Louvain evaluation")
        
        return results
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        data: torch.Tensor,
        true_communities: torch.Tensor,
        test_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models.
        
        Args:
            models: Dictionary of model names and models.
            data: Graph data.
            true_communities: True community assignments.
            test_mask: Test node mask.
            device: Device to run evaluation on.
            
        Returns:
            Dictionary of model comparison results.
        """
        results = {}
        
        for name, model in models.items():
            metrics = self.evaluate_model(
                model, data, true_communities, test_mask, device
            )
            results[name] = metrics
        
        return results
    
    def plot_confusion_matrix(
        self,
        true_communities: torch.Tensor,
        pred_communities: torch.Tensor,
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            true_communities: True community assignments.
            pred_communities: Predicted community assignments.
            save_path: Path to save plot.
        """
        true_np = true_communities.cpu().numpy()
        pred_np = pred_communities.cpu().numpy()
        
        cm = confusion_matrix(true_np, pred_np)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Community Detection Confusion Matrix')
        plt.xlabel('Predicted Community')
        plt.ylabel('True Community')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot metrics comparison across models.
        
        Args:
            results: Dictionary of model results.
            metrics: List of metrics to plot.
            save_path: Path to save plot.
        """
        if metrics is None:
            metrics = ['accuracy', 'nmi', 'ari', 'f1_macro']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            model_names = list(results.keys())
            metric_values = [results[name].get(metric, 0) for name in model_names]
            
            bars = axes[i].bar(model_names, metric_values)
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _edge_index_to_adj_matrix(
        self, 
        edge_index: torch.Tensor, 
        num_nodes: int
    ) -> torch.Tensor:
        """Convert edge index to adjacency matrix.
        
        Args:
            edge_index: Edge index tensor.
            num_nodes: Number of nodes.
            
        Returns:
            Adjacency matrix.
        """
        adj_matrix = torch.zeros(num_nodes, num_nodes)
        adj_matrix[edge_index[0], edge_index[1]] = 1
        return adj_matrix
    
    def _to_networkx(self, data: torch.Tensor) -> 'nx.Graph':
        """Convert PyTorch Geometric data to NetworkX graph.
        
        Args:
            data: PyTorch Geometric data.
            
        Returns:
            NetworkX graph.
        """
        import networkx as nx
        
        G = nx.Graph()
        
        # Add nodes
        for i in range(data.num_nodes):
            G.add_node(i)
        
        # Add edges
        edge_index = data.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i], edge_index[1, i])
        
        return G
    
    def _align_labels(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
        """Align predicted labels with true labels using Hungarian algorithm.
        
        Args:
            true_labels: True community labels.
            pred_labels: Predicted community labels.
            
        Returns:
            Aligned predicted labels.
        """
        from scipy.optimize import linear_sum_assignment
        
        # Create cost matrix
        n_true = len(np.unique(true_labels))
        n_pred = len(np.unique(pred_labels))
        
        cost_matrix = np.zeros((n_true, n_pred))
        
        for i, true_label in enumerate(np.unique(true_labels)):
            for j, pred_label in enumerate(np.unique(pred_labels)):
                mask_true = (true_labels == true_label)
                mask_pred = (pred_labels == pred_label)
                cost_matrix[i, j] = -np.sum(mask_true & mask_pred)
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Create mapping
        label_mapping = {}
        for i, j in zip(row_indices, col_indices):
            true_label = np.unique(true_labels)[i]
            pred_label = np.unique(pred_labels)[j]
            label_mapping[pred_label] = true_label
        
        # Apply mapping
        aligned_pred = np.array([label_mapping.get(label, label) for label in pred_labels])
        
        return aligned_pred
