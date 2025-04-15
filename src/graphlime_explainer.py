import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx, k_hop_subgraph
from graphlime import GraphLIME
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as ticker


# Patch GraphLIME to use Lasso and remove deprecated args
def patched_explain_node(self, node_idx, x, edge_index):
    eps = getattr(self, 'eps', 1e-5)  # fallback if not set
    self.model.eval()
    with torch.no_grad():
        y = self.model(x, edge_index)

    if y.dim() == 2 and y.size(1) == 1:
        y = y.view(-1)

    n, d = x.size()

    # Compute kernel matrix
    dist = x.reshape(1, n, d) - x.reshape(n, 1, d)
    dist = dist ** 2
    eps = 1e-5  # default GraphLIME value
    K = torch.exp(-dist.sum(dim=-1) / eps).cpu().numpy()


    #Label similarity
    L = (y.reshape(-1, 1) == y.reshape(1, -1)).float().cpu().numpy()

    # Remove i-th row/col
    K_bar = K[np.arange(n) != node_idx][:, np.arange(n) != node_idx]
    L_bar = L[np.arange(n) != node_idx][:, np.arange(n) != node_idx]

    # Lasso without deprecated args
    solver = Lasso(alpha=self.rho, fit_intercept=False, positive=True)
    solver.fit(K_bar * n, L_bar * n)

    return torch.tensor(solver.coef_, dtype=torch.float)

# Monkey-patch GraphLIME globally
GraphLIME.explain_node = patched_explain_node
print("Patched GraphLIME to use Lasso without deprecated normalize arg.")


# Ensure we use the same feature columns as in training
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

def get_subgraph(G, node, k_hops=2):
    """
    Extract a k-hop subgraph around a given node.
    
    Parameters:
        G (networkx.Graph): Full graph.
        node (int): Node ID to center subgraph around.
        k_hops (int): Number of hops to include in subgraph.
    
    Returns:
        subgraph (networkx.Graph): Extracted subgraph.
    """
    nodes = set([node])
    for _ in range(k_hops):
        neighbors = set()
        for n in nodes:
            neighbors.update(G.neighbors(n))
        nodes.update(neighbors)

    return G.subgraph(nodes)

def generate_llm_summary(anomaly_node, top_features):
    top_names = [f[0] for f in top_features]
    insight = (
        f"This node ({anomaly_node}) was flagged as an anomaly due to high influence from: "
        f"{', '.join(top_names)}."
    )
    return insight


def explain_anomaly(G, model, anomaly_node, save_path=None, k_hops=2):
    """
    Generate a GraphLIME explanation for a single anomaly node using a subgraph.
    """

    MAX_SUBGRAPH_NODES = 2000

    if isinstance(list(G.nodes())[0], str):
        anomaly_node = str(anomaly_node)
    else:
        anomaly_node = int(anomaly_node)

    if anomaly_node not in G:
        raise ValueError(f"Node {anomaly_node} not found in graph!")

    for node in G.nodes():
        G.nodes[node]['degree'] = G.degree(node)

    sample_attrs = G.nodes[next(iter(G.nodes()))]
    numeric_keys = [k for k, v in sample_attrs.items() if isinstance(v, (int, float, np.integer, np.floating))]

    pyg_graph = from_networkx(G, group_node_attrs=numeric_keys)
    if not hasattr(pyg_graph, 'x') or pyg_graph.x is None:
        raise ValueError("PyG graph conversion failed: No node features found!")

    node_idx = list(G.nodes()).index(anomaly_node)
    node_idx_tensor = torch.tensor([node_idx], dtype=torch.long)

    subset, edge_index, mapping, _ = k_hop_subgraph(
        node_idx=node_idx_tensor, num_hops=k_hops, edge_index=pyg_graph.edge_index, relabel_nodes=True
    )

    if len(subset) > MAX_SUBGRAPH_NODES:
        print(f"Subgraph exceeds {MAX_SUBGRAPH_NODES} nodes. Trimming for memory safety.")
        center_node_global = subset[mapping].item()

        trimmed_node_ids = subset[:MAX_SUBGRAPH_NODES].tolist()
        if center_node_global not in trimmed_node_ids:
            trimmed_node_ids[-1] = center_node_global

        trimmed_node_ids = list(set(trimmed_node_ids))
        id_map = {old_id: i for i, old_id in enumerate(trimmed_node_ids)}

        trimmed_node_tensor = torch.tensor(trimmed_node_ids, dtype=torch.long)

        mask_0 = torch.isin(edge_index[0], trimmed_node_tensor)
        mask_1 = torch.isin(edge_index[1], trimmed_node_tensor)
        mask = mask_0 & mask_1
        trimmed_edge_index = edge_index[:, mask]

        src = [id_map[int(s)] for s in trimmed_edge_index[0]]
        dst = [id_map[int(d)] for d in trimmed_edge_index[1]]
        sub_edge_index = torch.tensor([src, dst], dtype=torch.long)

        # Safely extract features
        sub_x = pyg_graph.x[trimmed_node_tensor]
        if sub_x.ndim == 1:
            sub_x = sub_x.unsqueeze(0)

        center_idx = id_map[center_node_global]

    else:
        if not isinstance(subset, torch.Tensor):
            subset = torch.tensor(subset, dtype=torch.long)
        elif subset.ndim == 0:
            subset = subset.unsqueeze(0)

        sub_x = pyg_graph.x[subset]
        if sub_x.ndim == 1:
            sub_x = sub_x.unsqueeze(0)

        sub_edge_index = edge_index
        center_idx = mapping.item()

    in_features = model.conv1.in_channels
    if sub_x.shape[1] != in_features:
        raise ValueError(f"Feature mismatch: sub_x has {sub_x.shape[1]} features, but model expects {in_features}.")

    graphlime = GraphLIME(model)
    # Normalize feature values to avoid overflow
    sub_x_np = sub_x.detach().cpu().numpy()
    sub_x_np = StandardScaler().fit_transform(sub_x_np)
    sub_x = torch.tensor(sub_x_np, dtype=torch.float)

    explanation = graphlime.explain_node(center_idx, sub_x, sub_edge_index)
    feature_importance = explanation.detach().numpy()

    # Plot explanation
    # plt.figure(figsize=(8, 6))
    # feature_importance = np.array(feature_importance).flatten()
    # if np.isnan(feature_importance).any():
    #     raise ValueError("Feature importance contains NaNs — check model or inputs.")
    # plt.bar(range(len(feature_importance)), feature_importance)
    # plt.xticks(ticks=range(len(feature_importance)), labels=FEATURE_COLUMNS, rotation=90)
    # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))

    # plt.xlabel("Feature Index")
    # plt.ylabel("Importance Score")
    # plt.title(f"GraphLIME Explanation for Node {anomaly_node}")

    # Get top 3 most important features
    feature_importance = np.array(feature_importance).flatten()
    if np.isnan(feature_importance).any():
        raise ValueError("Feature importance contains NaNs — check model or inputs.")
    num_features = min(len(FEATURE_COLUMNS), len(feature_importance))
    top_indices = feature_importance[:num_features].argsort()[::-1][:3]
    top_features = [(FEATURE_COLUMNS[i], feature_importance[i]) for i in top_indices]


    print(f"\nTop contributing features for node {anomaly_node}:")
    for name, score in top_features:
        print(f" - {name}: {score:.4f}")

    summary = (
    f"This node was flagged as an anomaly due to high influence from: "
    f"{top_features[0][0]}, {top_features[1][0]}, and {top_features[2][0]}. "
    )

    # if save_path:
    #     plt.savefig(save_path)
    #     print(f"Explanation saved to {save_path}")
    # else:
    #     plt.show()

    insight = generate_llm_summary(anomaly_node, top_features)
    return feature_importance, insight
