"""Example script demonstrating community detection with GNNs."""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from src.models import CommunityDetectionModel
from src.data import generate_sbm_graph, generate_karate_club
from src.train import CommunityDetectionTrainer
from src.eval import CommunityDetectionEvaluator
from src.utils import set_seed, create_logger


def main():
    """Main example function."""
    # Set up logging
    logger = create_logger("Example")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    logger.info("Community Detection with Graph Neural Networks - Example")
    
    # Generate synthetic data
    logger.info("Generating SBM graph...")
    data, true_communities = generate_sbm_graph(
        n_nodes=500,
        n_communities=4,
        p_in=0.3,
        p_out=0.05,
        seed=42
    )
    
    logger.info(f"Graph generated: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    logger.info(f"True communities: {len(torch.unique(true_communities))}")
    
    # Create data splits
    from src.data import create_train_val_test_split
    train_mask, val_mask, test_mask = create_train_val_test_split(
        data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
    )
    
    # Test different models
    models_to_test = ["gcn", "sage", "gat", "gin"]
    results = {}
    
    for model_type in models_to_test:
        logger.info(f"Training {model_type.upper()} model...")
        
        # Create model
        model = CommunityDetectionModel(
            model_type=model_type,
            input_dim=data.x.size(1),
            hidden_dim=64,
            output_dim=32,
            n_communities=len(torch.unique(true_communities)),
            n_layers=2,
            dropout=0.5
        )
        
        # Create trainer
        trainer = CommunityDetectionTrainer(
            model=model,
            learning_rate=0.01,
            weight_decay=5e-4,
            patience=20
        )
        
        # Train model
        history = trainer.train(
            data=data,
            train_mask=train_mask,
            val_mask=val_mask,
            n_epochs=100,
            verbose=False
        )
        
        # Evaluate model
        test_metrics = trainer.evaluate(data, test_mask)
        results[model_type] = test_metrics
        
        logger.info(f"{model_type.upper()} Results: {test_metrics}")
    
    # Compare models
    logger.info("Model Comparison:")
    logger.info("-" * 50)
    logger.info(f"{'Model':<10} {'Accuracy':<10} {'NMI':<10} {'ARI':<10} {'F1-Macro':<10}")
    logger.info("-" * 50)
    
    for model_type, metrics in results.items():
        logger.info(
            f"{model_type.upper():<10} "
            f"{metrics['accuracy']:<10.4f} "
            f"{metrics['nmi']:<10.4f} "
            f"{metrics['ari']:<10.4f} "
            f"{metrics['f1_macro']:<10.4f}"
        )
    
    # Evaluate baselines
    logger.info("Evaluating baseline methods...")
    evaluator = CommunityDetectionEvaluator()
    baseline_results = evaluator.evaluate_baseline_methods(
        data, true_communities, test_mask
    )
    
    logger.info("Baseline Results:")
    for method, metrics in baseline_results.items():
        logger.info(f"{method.upper()}: {metrics}")
    
    # Visualize results
    logger.info("Creating visualizations...")
    
    # Plot metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics_to_plot = ['accuracy', 'nmi', 'ari', 'f1_macro']
    
    for i, metric in enumerate(metrics_to_plot):
        row = i // 2
        col = i % 2
        
        model_names = list(results.keys())
        metric_values = [results[name][metric] for name in model_names]
        
        bars = axes[row, col].bar(model_names, metric_values)
        axes[row, col].set_title(f'{metric.upper()} Comparison')
        axes[row, col].set_ylabel(metric.upper())
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[row, col].text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.01,
                f'{value:.3f}', 
                ha='center', 
                va='bottom'
            )
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Test on Karate Club dataset
    logger.info("Testing on Karate Club dataset...")
    karate_data, karate_communities = generate_karate_club()
    
    karate_train_mask, karate_val_mask, karate_test_mask = create_train_val_test_split(
        karate_data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
    )
    
    # Train GCN on Karate Club
    karate_model = CommunityDetectionModel(
        model_type="gcn",
        input_dim=karate_data.x.size(1),
        hidden_dim=16,
        output_dim=8,
        n_communities=len(torch.unique(karate_communities)),
        n_layers=2
    )
    
    karate_trainer = CommunityDetectionTrainer(karate_model, learning_rate=0.01)
    karate_history = karate_trainer.train(
        data=karate_data,
        train_mask=karate_train_mask,
        val_mask=karate_val_mask,
        n_epochs=50,
        verbose=False
    )
    
    karate_metrics = karate_trainer.evaluate(karate_data, karate_test_mask)
    logger.info(f"Karate Club Results: {karate_metrics}")
    
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()
