"""Streamlit demo for community detection."""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network

from src.models import CommunityDetectionModel
from src.data import (
    generate_sbm_graph,
    generate_karate_club,
    generate_erdos_renyi_graph,
    generate_barabasi_albert_graph
)
from src.train import CommunityDetectionTrainer
from src.eval import CommunityDetectionEvaluator
from src.utils import set_seed, get_device


def create_interactive_graph(data, communities, title="Community Detection"):
    """Create an interactive graph visualization using PyVis."""
    G = nx.Graph()
    
    # Add nodes
    for i in range(data.num_nodes):
        G.add_node(i, group=communities[i].item())
    
    # Add edges
    edge_index = data.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])
    
    # Create PyVis network
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    
    # Configure physics
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      }
    }
    """)
    
    return net


def plot_community_distribution(communities, title="Community Distribution"):
    """Plot community size distribution."""
    unique, counts = np.unique(communities, return_counts=True)
    
    fig = go.Figure(data=[
        go.Bar(x=[f"Community {i}" for i in unique], y=counts)
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Community",
        yaxis_title="Number of Nodes",
        showlegend=False
    )
    
    return fig


def plot_metrics_comparison(results):
    """Plot metrics comparison across models."""
    metrics = ['accuracy', 'nmi', 'ari', 'f1_macro']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=metrics,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    for i, metric in enumerate(metrics):
        row = i // 2 + 1
        col = i % 2 + 1
        
        model_names = list(results.keys())
        metric_values = [results[name].get(metric, 0) for name in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=metric_values, name=metric),
            row=row, col=col
        )
    
    fig.update_layout(
        title_text="Model Performance Comparison",
        showlegend=False,
        height=600
    )
    
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Community Detection with GNNs",
        page_icon="🕸️",
        layout="wide"
    )
    
    st.title("🕸️ Community Detection with Graph Neural Networks")
    st.markdown("""
    This demo showcases different Graph Neural Network architectures for community detection.
    You can generate synthetic graphs, train various GNN models, and visualize the results.
    """)
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Dataset selection
    dataset = st.sidebar.selectbox(
        "Dataset",
        ["SBM", "Karate Club", "Erdos-Renyi", "Barabasi-Albert"],
        help="Choose the type of graph to generate"
    )
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["GCN", "GraphSAGE", "GAT", "GIN"],
        help="Choose the GNN architecture"
    )
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    n_epochs = st.sidebar.slider("Epochs", 10, 200, 50)
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
    hidden_dim = st.sidebar.slider("Hidden Dimension", 16, 128, 64)
    
    # SBM parameters
    if dataset == "SBM":
        st.sidebar.subheader("SBM Parameters")
        n_nodes = st.sidebar.slider("Number of Nodes", 100, 2000, 500)
        n_communities = st.sidebar.slider("Number of Communities", 2, 8, 4)
        p_in = st.sidebar.slider("Within-community Probability", 0.1, 0.8, 0.3)
        p_out = st.sidebar.slider("Between-community Probability", 0.01, 0.3, 0.05)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Graph Visualization")
        
        # Generate data
        if st.button("Generate Graph", type="primary"):
            with st.spinner("Generating graph..."):
                set_seed(42)
                
                if dataset == "SBM":
                    data, true_communities = generate_sbm_graph(
                        n_nodes=n_nodes,
                        n_communities=n_communities,
                        p_in=p_in,
                        p_out=p_out,
                        seed=42
                    )
                elif dataset == "Karate Club":
                    data, true_communities = generate_karate_club()
                elif dataset == "Erdos-Renyi":
                    data = generate_erdos_renyi_graph(n_nodes=100, p=0.1, seed=42)
                    true_communities = torch.zeros(data.num_nodes, dtype=torch.long)
                elif dataset == "Barabasi-Albert":
                    data = generate_barabasi_albert_graph(n_nodes=100, m=3, seed=42)
                    true_communities = torch.zeros(data.num_nodes, dtype=torch.long)
                
                # Store in session state
                st.session_state.data = data
                st.session_state.true_communities = true_communities
        
        # Display graph if available
        if 'data' in st.session_state:
            data = st.session_state.data
            true_communities = st.session_state.true_communities
            
            st.write(f"**Graph Statistics:**")
            st.write(f"- Nodes: {data.num_nodes}")
            st.write(f"- Edges: {data.edge_index.size(1)}")
            st.write(f"- Communities: {len(torch.unique(true_communities))}")
            
            # Create interactive visualization
            net = create_interactive_graph(data, true_communities, "True Communities")
            net_html = net.generate_html()
            st.components.v1.html(net_html, height=600)
    
    with col2:
        st.header("Model Training")
        
        if 'data' in st.session_state:
            if st.button("Train Model", type="secondary"):
                with st.spinner("Training model..."):
                    data = st.session_state.data
                    true_communities = st.session_state.true_communities
                    
                    # Create model
                    model = CommunityDetectionModel(
                        model_type=model_type.lower(),
                        input_dim=data.x.size(1),
                        hidden_dim=hidden_dim,
                        output_dim=32,
                        n_communities=len(torch.unique(true_communities)),
                        n_layers=2,
                        dropout=0.5
                    )
                    
                    # Create trainer
                    trainer = CommunityDetectionTrainer(
                        model=model,
                        learning_rate=learning_rate,
                        patience=10
                    )
                    
                    # Create data splits
                    from src.data import create_train_val_test_split
                    train_mask, val_mask, test_mask = create_train_val_test_split(
                        data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
                    )
                    
                    # Train model
                    history = trainer.train(
                        data=data,
                        train_mask=train_mask,
                        val_mask=val_mask,
                        n_epochs=n_epochs,
                        verbose=False
                    )
                    
                    # Evaluate model
                    test_metrics = trainer.evaluate(data, test_mask)
                    
                    # Store results
                    st.session_state.model = model
                    st.session_state.trainer = trainer
                    st.session_state.test_metrics = test_metrics
                    st.session_state.history = history
                    
                    st.success("Model trained successfully!")
        
        # Display results if available
        if 'test_metrics' in st.session_state:
            st.subheader("Test Results")
            metrics = st.session_state.test_metrics
            
            for metric, value in metrics.items():
                st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
            
            # Plot training history
            history = st.session_state.history
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history['train_losses'],
                mode='lines',
                name='Train Loss',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                y=history['val_losses'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title="Training History",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Bottom section for detailed analysis
    if 'data' in st.session_state and 'test_metrics' in st.session_state:
        st.header("Detailed Analysis")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Community Distribution")
            true_communities = st.session_state.true_communities
            fig = plot_community_distribution(true_communities.cpu().numpy())
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.subheader("Model Comparison")
            # This would include baseline comparisons
            st.info("Baseline comparison features coming soon!")
        
        # Prediction visualization
        if st.button("Show Predictions"):
            model = st.session_state.model
            data = st.session_state.data
            
            model.eval()
            with torch.no_grad():
                embeddings, logits = model(data.x, data.edge_index)
                pred_communities = logits.argmax(dim=1)
            
            # Create prediction visualization
            net_pred = create_interactive_graph(data, pred_communities, "Predicted Communities")
            net_pred_html = net_pred.generate_html()
            st.components.v1.html(net_pred_html, height=600)


if __name__ == "__main__":
    main()
