import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from src.graph_builder import convert_to_pyg

FEATURES_DIR = "features"
ANOMALY_OUTPUT_FILE = os.path.join(FEATURES_DIR, "anomaly_scores_gnn.csv")

# Define the selected features for training
FEATURE_COLUMNS = [
    "degree", "in_degree", "out_degree",  # Graph structure
    "num_txs_as_sender", "num_txs_as receiver", "total_txs",
    "lifetime_in_blocks", "num_timesteps_appeared_in",
    "btc_transacted_total", "btc_transacted_mean", "btc_transacted_median",
    "btc_sent_total", "btc_sent_mean", "btc_sent_median",
    "btc_received_total", "btc_received_mean", "btc_received_median",
    "fees_total", "fees_mean", "fees_median",
    "blocks_btwn_txs_mean", "blocks_btwn_input_txs_mean", "blocks_btwn_output_txs_mean",
    "num_addr_transacted_multiple", "transacted_w_address_mean"
]

class AnomalyGCN(nn.Module):
    def __init__(self, in_features, hidden_dim=16):
        super(AnomalyGCN, self).__init__()
        print(f"Initializing AnomalyGCN with {in_features} input features")
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)  # Output a single score per node

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.view(-1)  # Return a single value per node

def detect_anomalies(G, node_features, num_anomalies=10, num_epochs=100, learning_rate=0.01):
    """Trains a GNN for anomaly detection and returns top anomalies."""

    # Ensure node_features is indexed by node IDs
    node_features = node_features.copy()
    node_features.index = node_features["node"]

    print("Selecting relevant features...")
    node_features_filtered = node_features[FEATURE_COLUMNS].fillna(0)  # Handle missing values

    print(f"Model was trained on {node_features.shape[1]} node_features.")

    print("Converting graph and features for PyG...")
    pyg_graph, features = convert_to_pyg(G, node_features_filtered)

    print(f"After converting there are {features.shape[1]} node features.")

    # Initialize GNN model
    model = AnomalyGCN(in_features=features.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()  # Minimize reconstruction error

    print("Training GNN for anomaly detection...")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        scores = model(features, pyg_graph.edge_index)
        
        # Use mean as a proxy for "normal" nodes
        loss = loss_fn(scores, torch.ones_like(scores) * scores.mean())
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs} - Loss: {loss.item():.4f}")

    print(f"Training feature count: {node_features_filtered.shape[1]}")

    print("Anomaly scoring...")
    model.eval()
    with torch.no_grad():
        anomaly_scores = model(features, pyg_graph.edge_index).numpy()

    node_features["anomaly_score"] = anomaly_scores
    node_features["anomaly_label"] = (anomaly_scores < anomaly_scores.mean() - 2 * anomaly_scores.std()).astype(int)

    # Sort anomalies by lowest anomaly score
    anomalies = node_features.sort_values(by="anomaly_score")

    # Save results
    os.makedirs(FEATURES_DIR, exist_ok=True)
    node_features.to_csv(ANOMALY_OUTPUT_FILE, index=False)
    print(f"Anomaly detection complete. Results saved to {ANOMALY_OUTPUT_FILE}")

    # Print top N anomalies
    top_anomalies = anomalies.head(num_anomalies)
    print("\nTop 10 Anomalous Nodes:")
    for i, (index, row) in enumerate(top_anomalies.iterrows()):
        print(f"{i+1}. Node: {row['node']}, Score: {row['anomaly_score']:.4f}")
    
    anomalies = anomalies.copy()
    anomalies["node"] = anomalies.index  # preserve node ID from the index
    top_anomalies = anomalies.head(num_anomalies)

    return node_features, top_anomalies, model
