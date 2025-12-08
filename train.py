"""Main training script for community detection."""

import argparse
import os
from typing import Dict, Any

import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from src.models import CommunityDetectionModel
from src.data import (
    generate_sbm_graph,
    generate_karate_club,
    generate_erdos_renyi_graph,
    generate_barabasi_albert_graph,
    create_train_val_test_split
)
from src.train import CommunityDetectionTrainer
from src.eval import CommunityDetectionEvaluator
from src.utils import set_seed, get_device, create_logger


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Loaded configuration.
    """
    return OmegaConf.load(config_path)


def create_data(config: DictConfig):
    """Create dataset based on configuration.
    
    Args:
        config: Configuration object.
        
    Returns:
        Graph data and true communities.
    """
    dataset_name = config.data.dataset.lower()
    
    if dataset_name == "sbm":
        data, true_communities = generate_sbm_graph(
            n_nodes=config.data.n_nodes,
            n_communities=config.data.n_communities,
            p_in=config.data.p_in,
            p_out=config.data.p_out,
            seed=config.system.seed
        )
    elif dataset_name == "karate":
        data, true_communities = generate_karate_club()
    elif dataset_name == "erdos_renyi":
        data = generate_erdos_renyi_graph(
            n_nodes=config.data.n_nodes,
            p=config.data.get('p', 0.1),
            seed=config.system.seed
        )
        true_communities = torch.zeros(data.num_nodes, dtype=torch.long)
    elif dataset_name == "barabasi_albert":
        data = generate_barabasi_albert_graph(
            n_nodes=config.data.n_nodes,
            m=config.data.get('m', 3),
            seed=config.system.seed
        )
        true_communities = torch.zeros(data.num_nodes, dtype=torch.long)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return data, true_communities


def create_model(config: DictConfig, data) -> CommunityDetectionModel:
    """Create model based on configuration.
    
    Args:
        config: Configuration object.
        data: Graph data.
        
    Returns:
        Community detection model.
    """
    model_config = config.model
    
    # Determine input dimension
    input_dim = data.x.size(1) if data.x is not None else model_config.input_dim
    
    # Create model
    model = CommunityDetectionModel(
        model_type=model_config.type,
        input_dim=input_dim,
        hidden_dim=model_config.hidden_dim,
        output_dim=model_config.output_dim,
        n_communities=model_config.n_communities,
        n_layers=model_config.n_layers,
        dropout=model_config.dropout,
        n_heads=model_config.get('n_heads', 4),
        concat=model_config.get('concat', True),
        eps=model_config.get('eps', 0.0),
        aggregator=model_config.get('aggregator', 'mean')
    )
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train community detection model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["gcn", "sage", "gat", "gin"],
        help="Override model type from config"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["sbm", "karate", "erdos_renyi", "barabasi_albert"],
        help="Override dataset from config"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model_type:
        config.model.type = args.model_type
    if args.dataset:
        config.data.dataset = args.dataset
    
    # Set up logging
    logger = create_logger("CommunityDetection", config.logging.level)
    logger.info(f"Starting training with config: {config}")
    
    # Set random seed
    set_seed(config.system.seed)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data
    logger.info("Creating dataset...")
    data, true_communities = create_data(config)
    logger.info(f"Dataset created: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    
    # Create data splits
    train_mask, val_mask, test_mask = create_train_val_test_split(
        data,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        seed=config.system.seed
    )
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config, data)
    logger.info(f"Model created: {model.model_type}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create trainer
    trainer = CommunityDetectionTrainer(
        model=model,
        device=device,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        patience=config.training.patience,
        min_delta=config.training.min_delta
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        n_epochs=config.training.n_epochs,
        verbose=True
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    test_metrics = trainer.evaluate(data, test_mask)
    
    # Evaluate baselines
    evaluator = CommunityDetectionEvaluator()
    baseline_results = evaluator.evaluate_baseline_methods(
        data, true_communities, test_mask
    )
    
    # Print results
    logger.info("=== RESULTS ===")
    logger.info(f"Test Metrics: {test_metrics}")
    logger.info(f"Baseline Results: {baseline_results}")
    
    # Save results
    results = {
        'test_metrics': test_metrics,
        'baseline_results': baseline_results,
        'config': OmegaConf.to_yaml(config),
        'history': history
    }
    
    results_path = os.path.join(args.output_dir, "results.yaml")
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    # Save model
    if config.logging.save_model:
        model_path = os.path.join(args.output_dir, "model.pth")
        trainer.save_model(model_path)
    
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
