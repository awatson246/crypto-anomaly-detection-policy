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
        - full node_features (ALL nodes in G) with anomaly_score
        - full sorted anomaly ranking (all nodes)
        - wallet-only anomaly ranking (safe for explainer)
        - trained model
    """

    print("\nNormalizing node_features input...")

    # Normalize node ID column
    node_features['node'] = node_features['node'].astype(str)

    # Only keep features whose nodes exist in the graph
    valid_nodes = set(map(str, G.nodes()))
    node_features = node_features[node_features['node'].isin(valid_nodes)].copy()

    # Set index to node ID
    node_features.index = node_features['node']

    print(f" - Features received: {len(node_features)}")
    print(f" - Graph nodes:        {len(G.nodes())}")

    print("\nBuilding full feature matrix aligned to graph nodes...")

    # Build feature matrix aligned EXACTLY to G.nodes()
    all_nodes = list(map(str, G.nodes()))
    node_features_full = node_features.reindex(all_nodes)

    # Missing nodes get 0s
    node_features_full = node_features_full.fillna(0)

    # Ensure all feature columns exist
    for feat in FEATURE_COLUMNS:
        if feat not in node_features_full.columns:
            node_features_full[feat] = 0.0

    # Ensure node_type is present
    if "node_type" not in node_features_full.columns:
        node_features_full["node_type"] = "unknown"

    print(" - Final aligned feature rows:", len(node_features_full))
    print(" - Should match graph nodes:", len(G.nodes()))

    # Convert to torch tensor
    features = torch.tensor(node_features_full[FEATURE_COLUMNS].values, dtype=torch.float)

    print("\nBuilding PyG graph...")
    pyg_graph, _ = convert_to_pyg(G, node_features_full)

    # Sanity check
    assert features.shape[0] == pyg_graph.num_nodes, \
        f"FEATURE MISMATCH! features={features.shape[0]}  pyg={pyg_graph.num_nodes}"

    print("\nTraining GNN anomaly detector...")
    model = AnomalyGCN(in_features=features.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        scores = model(features, pyg_graph.edge_index)

        # Self-supervised: minimize distance from global average
        loss = loss_fn(scores, torch.full_like(scores, scores.mean().item()))
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs} - Loss: {loss.item():.4f}")

    print("\nScoring nodes...")
    model.eval()
    with torch.no_grad():
        anomaly_scores = model(features, pyg_graph.edge_index).cpu().numpy().flatten()

    # Attach anomaly scores
    node_features_full["anomaly_score"] = anomaly_scores

    print(" - anomaly_scores len:", len(anomaly_scores))
    print(" - node_features_full rows:", len(node_features_full))

    # Save
    os.makedirs(FEATURES_DIR, exist_ok=True)
    node_features_full.to_csv(ANOMALY_OUTPUT_FILE, index=False)

    print("\nRanking anomalies...")

    # Full ranking
    full_rank_all_nodes = node_features_full.sort_values("anomaly_score")

    # Wallet-only safe subset
    wallet_rank = full_rank_all_nodes[
        full_rank_all_nodes["type"].astype(str) == "wallet"
    ]

    print(" - wallet_rank count:", len(wallet_rank))
    print(f"Anomaly detection complete. Scores saved to {ANOMALY_OUTPUT_FILE}")

    return node_features_full, full_rank_all_nodes, wallet_rank, model
