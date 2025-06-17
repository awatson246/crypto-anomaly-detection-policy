import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx, k_hop_subgraph
from graphlime import GraphLIME
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

# Define which features are used
FEATURE_COLUMNS = [
    "degree", "in_degree", "out_degree",  
    "num_txs_as_sender", "num_txs_as_receiver", "total_txs",
    "lifetime_in_blocks", "num_timesteps_appeared_in",
    "btc_transacted_total", "btc_transacted_mean", "btc_transacted_median",
    "btc_sent_total", "btc_sent_mean", "btc_sent_median",
    "btc_received_total", "btc_received_mean", "btc_received_median",
    "fees_total", "fees_mean", "fees_median",
    "blocks_btwn_txs_mean", "blocks_btwn_input_txs_mean", "blocks_btwn_output_txs_mean",
    "num_addr_transacted_multiple", "transacted_w_address_mean"
]

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Monkey-patch GraphLIME to avoid deprecated 'normalize' arg
def patched_explain_node(self, node_idx, x, edge_index):
    eps = getattr(self, 'eps', 1e-5)
    self.model.eval()
    with torch.no_grad():
        y = self.model(x, edge_index)

    if y.dim() == 2 and y.size(1) == 1:
        y = y.view(-1)

    n, d = x.size()
    dist = x.reshape(1, n, d) - x.reshape(n, 1, d)
    dist = dist ** 2
    K = torch.exp(-dist.sum(dim=-1) / eps).cpu().numpy()
    L = (y.reshape(-1, 1) == y.reshape(1, -1)).float().cpu().numpy()

    K_bar = K[np.arange(n) != node_idx][:, np.arange(n) != node_idx]
    L_bar = L[np.arange(n) != node_idx][:, np.arange(n) != node_idx]

    solver = Lasso(alpha=self.rho, fit_intercept=False, positive=True)
    solver.fit(K_bar * n, L_bar * n)
    return torch.tensor(solver.coef_, dtype=torch.float)

GraphLIME.explain_node = patched_explain_node
print("Patched GraphLIME to avoid deprecated arguments.")

def format_insight(anomaly_node_id, top_features, node_attributes):
    """
    Builds a readable summary and structured data for LLM input.
    """
    insight = f"""Node ID: {anomaly_node_id}
Type: {node_attributes.get('type', 'unknown')}
Class Label: {node_attributes.get('class', 'N/A')}
Time Step: {node_attributes.get('Time step', 'N/A')}
Lifetime (blocks): {node_attributes.get('lifetime_in_blocks', 'N/A')}

Top contributing features from GraphLIME:
""" + "\n".join([f" - {feat}: {imp:.3e}" for feat, imp in top_features]) + "\n\n"

    insight += "Additional node statistics:\n"
    node_data_dict = {}
    for key in [
        "total_txs", "btc_received_total", "btc_sent_total",
        "transacted_w_address_total", "num_txs_as_sender", "num_txs_as_receiver",
        "btc_transacted_total", "fees_total", "degree"
    ]:
        if key in node_attributes:
            val = node_attributes[key]
            node_data_dict[key] = val
            insight += f" - {key}: {val}\n"

    insight += "\nThis node was flagged as an anomaly due to the above feature importance patterns."
    return insight, top_features, node_data_dict

def explain_anomaly(G, model, anomaly_node, save_path=None, k_hops=2):
    """
    Run GraphLIME explanation for a specific node in the graph.
    """
    MAX_SUBGRAPH_NODES = 2000

    # Ensure anomaly node ID is str if graph uses strings
    anomaly_node = str(anomaly_node) if isinstance(list(G.nodes())[0], str) else int(anomaly_node)

    if anomaly_node not in G:
        raise ValueError(f"Node {anomaly_node} not found in graph!")

    for node in G.nodes():
        G.nodes[node]['degree'] = G.degree(node)

    # Extract numeric features
    sample_attrs = G.nodes[next(iter(G.nodes()))]
    numeric_keys = [k for k, v in sample_attrs.items() if isinstance(v, (int, float, np.integer, np.floating))]

    # Convert to PyG format
    pyg_graph = from_networkx(G, group_node_attrs=numeric_keys)
    if not hasattr(pyg_graph, 'x') or pyg_graph.x is None:
        raise ValueError("PyG conversion failed: No node features found.")

    node_idx = list(G.nodes()).index(anomaly_node)
    node_idx_tensor = torch.tensor([node_idx], dtype=torch.long)

    # Extract subgraph
    subset, edge_index, mapping, _ = k_hop_subgraph(
        node_idx=node_idx_tensor, num_hops=k_hops, edge_index=pyg_graph.edge_index, relabel_nodes=True
    )

    if len(subset) > MAX_SUBGRAPH_NODES:
        center_node_global = subset[mapping].item()
        trimmed_node_ids = subset[:MAX_SUBGRAPH_NODES].tolist()
        if center_node_global not in trimmed_node_ids:
            trimmed_node_ids[-1] = center_node_global
        trimmed_node_ids = list(set(trimmed_node_ids))
        id_map = {old_id: i for i, old_id in enumerate(trimmed_node_ids)}

        trimmed_node_tensor = torch.tensor(trimmed_node_ids, dtype=torch.long)
        mask = torch.isin(edge_index[0], trimmed_node_tensor) & torch.isin(edge_index[1], trimmed_node_tensor)
        trimmed_edge_index = edge_index[:, mask]
        sub_edge_index = torch.tensor([
            [id_map[int(s)] for s in trimmed_edge_index[0]],
            [id_map[int(d)] for d in trimmed_edge_index[1]]
        ], dtype=torch.long)
        sub_x = pyg_graph.x[trimmed_node_tensor]
        center_idx = id_map[center_node_global]
    else:
        sub_x = pyg_graph.x[subset]
        sub_edge_index = edge_index
        center_idx = mapping.item()

    if sub_x.ndim == 1:
        sub_x = sub_x.unsqueeze(0)

    # Standardize features
    sub_x_np = sub_x.detach().cpu().numpy()
    sub_x_np = StandardScaler().fit_transform(sub_x_np)
    sub_x = torch.tensor(sub_x_np, dtype=torch.float)

    if sub_x.shape[1] != model.conv1.in_channels:
        raise ValueError(f"Subgraph feature mismatch: {sub_x.shape[1]} vs expected {model.conv1.in_channels}")

    # Run explanation
    graphlime = GraphLIME(model)
    explanation = graphlime.explain_node(center_idx, sub_x, sub_edge_index)
    feature_importance = explanation.detach().numpy().flatten()

    # Safely pair feature names and importance scores
    num_features = min(len(FEATURE_COLUMNS), len(feature_importance))

    # Only keep non-zero importance scores (with index tracking)
    nonzero_importances = [
        (i, feature_importance[i])
        for i in range(num_features)
        if feature_importance[i] > 1e-8  # small epsilon to ignore near-zero noise
    ]

    # Sort by importance (descending) and take top 3
    top_indices = sorted(nonzero_importances, key=lambda x: x[1], reverse=True)[:3]
    top_features = [(FEATURE_COLUMNS[i], imp) for i, imp in top_indices]

    row = G.nodes[anomaly_node]
    insight, top_features, node_data_dict = format_insight(anomaly_node, top_features, row)

    return feature_importance, top_features, node_data_dict, insight
