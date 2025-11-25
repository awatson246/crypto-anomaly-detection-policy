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
    "num_txs_as_sender", "num_txs_as_receiver", "total_txs",
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

    # Accept extra args/kwargs used by PyG explainers (node_index, idx, etc.)
    def forward(self, x, edge_index, *args, **kwargs):
        """
        Note: *args and **kwargs allow Explainer to pass node_index (and others)
        during the explanation process without causing an error.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x  # shape will be (num_nodes, 1)


def detect_anomalies(G, node_features, num_epochs=350, learning_rate=0.01):
    """
    Trains a GNN for anomaly detection and returns:
        - full node_features with anomaly_score
        - full sorted anomaly ranking (all nodes)
        - trained model
    """

    # Ensure node_features indexed by node IDs
    node_features = node_features.copy()
    node_features.index = node_features["node"].astype(str)

    print("Selecting relevant features...")
    node_features_filtered = node_features[FEATURE_COLUMNS].fillna(0)

    print("Converting graph and features for PyG...")
    pyg_graph, features, node_id_map = convert_to_pyg(G, node_features_filtered)

    # Initialize GNN model
    model = AnomalyGCN(in_features=features.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    print("Training GNN for anomaly detection...")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        scores = model(features, pyg_graph.edge_index)
        loss = loss_fn(scores, torch.ones_like(scores) * scores.mean())
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs} - Loss: {loss.item():.4f}")

    print("Anomaly scoring...")
    model.eval()
    with torch.no_grad():
        anomaly_scores = model(features, pyg_graph.edge_index).numpy().flatten()

    # Attach anomaly scores
    node_features["anomaly_score"] = anomaly_scores

    # Attach REAL bitcoin node IDs
    node_features["node_id"] = node_id_map

    # Save full list of scores
    os.makedirs(FEATURES_DIR, exist_ok=True)
    node_features.to_csv(ANOMALY_OUTPUT_FILE, index=False)

    # Sort ALL nodes by anomaly_score (ascending = more anomalous)
    full_rank = node_features.sort_values("anomaly_score")

    print(f"Anomaly detection complete. Scores saved to {ANOMALY_OUTPUT_FILE}")

    # Return all scores — NOT truncated at 10
    return node_features, full_rank, model

