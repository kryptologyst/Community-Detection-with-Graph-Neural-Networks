"""Legacy community detection using Louvain algorithm (original implementation)."""

import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
from src.utils import set_seed, create_logger


def louvain_community_detection():
    """Original Louvain community detection implementation."""
    logger = create_logger("Louvain")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    logger.info("Running original Louvain community detection...")
    
    # Create a sample graph (Zachary's Karate Club)
    G = nx.karate_club_graph()
    
    # Apply Louvain community detection
    partition = community_louvain.best_partition(G)
    
    # Visualize the communities
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    
    # Color nodes by community
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
    for community_id in set(partition.values()):
        members = [node for node in partition if partition[node] == community_id]
        nx.draw_networkx_nodes(
            G, pos, nodelist=members, 
            node_color=colors[community_id % len(colors)],
            node_size=300,
            label=f"Community {community_id}"
        )
    
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title("Louvain Community Detection on Karate Club Graph")
    plt.axis("off")
    plt.legend()
    plt.tight_layout()
    plt.savefig('louvain_communities.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print community assignments
    logger.info("Community assignments:")
    for node, comm in partition.items():
        logger.info(f"Node {node} → Community {comm}")
    
    # Compute modularity
    modularity = community_louvain.modularity(partition, G)
    logger.info(f"Modularity: {modularity:.4f}")
    
    return partition, modularity


def compare_with_gnn():
    """Compare Louvain with GNN-based community detection."""
    logger = create_logger("Comparison")
    
    # Run Louvain
    louvain_partition, louvain_modularity = louvain_community_detection()
    
    # Run GNN-based detection
    from src.data import generate_karate_club
    from src.models import CommunityDetectionModel
    from src.train import CommunityDetectionTrainer
    from src.data import create_train_val_test_split
    
    logger.info("Running GNN-based community detection...")
    
    # Generate Karate Club data in PyTorch Geometric format
    data, true_communities = generate_karate_club()
    
    # Create GNN model
    model = CommunityDetectionModel(
        model_type="gcn",
        input_dim=data.x.size(1),
        hidden_dim=16,
        output_dim=8,
        n_communities=2,  # Karate Club has 2 communities
        n_layers=2
    )
    
    # Create trainer
    trainer = CommunityDetectionTrainer(model, learning_rate=0.01)
    
    # Create data splits
    train_mask, val_mask, test_mask = create_train_val_test_split(
        data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
    )
    
    # Train model
    history = trainer.train(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        n_epochs=100,
        verbose=False
    )
    
    # Evaluate
    test_metrics = trainer.evaluate(data, test_mask)
    
    logger.info("Comparison Results:")
    logger.info(f"Louvain Modularity: {louvain_modularity:.4f}")
    logger.info(f"GNN Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"GNN NMI: {test_metrics['nmi']:.4f}")
    logger.info(f"GNN ARI: {test_metrics['ari']:.4f}")
    
    return louvain_partition, test_metrics


if __name__ == "__main__":
    # Run original implementation
    louvain_community_detection()
    
    # Compare with GNN
    compare_with_gnn()
