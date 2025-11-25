import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx, k_hop_subgraph
from torch_geometric.explain import Explainer, GNNExplainer as GNNExplainerAlgo
from graphlime import GraphLIME
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings


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

# ---- Patched GraphLIME ----
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

# ---- Explanation runner ----
def run_explainers(model, sub_x, sub_edge_index, center_idx, explainers=["graphlime", "gnnexplainer"]):
    results = {}

    if "graphlime" in explainers:
        graphlime = GraphLIME(model)
        explanation = graphlime.explain_node(center_idx, sub_x, sub_edge_index)
        results['graphlime'] = explanation.detach().numpy().flatten()

    if "gnnexplainer" in explainers:
        try:
            expl = Explainer(
                model=model,
                algorithm=GNNExplainerAlgo(epochs=100),
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config=dict(
                    mode='regression',      # anomaly GNN outputs scalar
                    task_level='node',
                    return_type='raw'
                )
            )

            explanation = expl(
                node_index=center_idx,
                x=sub_x,
                edge_index=sub_edge_index
            )

            results["gnnexplainer"] = (
                explanation.node_feat_mask.detach().cpu().numpy()
            )

        except Exception as e:
            print(f"[WARN] GNNExplainer failed: {e}")
            results["gnnexplainer"] = None

    return results

# ---- Main wrapper ----
def explain_anomaly_multi(G, model, anomaly_node, save_path=None, k_hops=2, explainers=["graphlime", "gnnexplainer"]):
    MAX_SUBGRAPH_NODES = 2000
    anomaly_node = str(anomaly_node) if isinstance(list(G.nodes())[0], str) else int(anomaly_node)

    if anomaly_node not in G:
        raise ValueError(f"Node {anomaly_node} not found in graph!")

    for node in G.nodes():
        G.nodes[node]['degree'] = G.degree(node)

    numeric_keys = [k for k, v in G.nodes[next(iter(G.nodes()))].items() if isinstance(v, (int, float, np.integer, np.floating))]
    pyg_graph = from_networkx(G, group_node_attrs=numeric_keys)
    if not hasattr(pyg_graph, 'x') or pyg_graph.x is None:
        raise ValueError("PyG conversion failed: No node features found.")

    node_idx = list(G.nodes()).index(anomaly_node)
    node_idx_tensor = torch.tensor([node_idx], dtype=torch.long)

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
        sub_edge_index = edge_index[:, mask]
        sub_edge_index = torch.tensor([
            [id_map[int(s)] for s in sub_edge_index[0]],
            [id_map[int(d)] for d in sub_edge_index[1]]
        ], dtype=torch.long)
        sub_x = pyg_graph.x[trimmed_node_tensor]
        center_idx = id_map[center_node_global]
    else:
        sub_x = pyg_graph.x[subset]
        sub_edge_index = edge_index
        center_idx = mapping.item()

    if sub_x.ndim == 1:
        sub_x = sub_x.unsqueeze(0)

    sub_x_np = sub_x.detach().cpu().numpy()
    sub_x_np = StandardScaler().fit_transform(sub_x_np)
    sub_x = torch.tensor(sub_x_np, dtype=torch.float)

    if sub_x.shape[1] != model.conv1.in_channels:
        raise ValueError(f"Subgraph feature mismatch: {sub_x.shape[1]} vs expected {model.conv1.in_channels}")

    # Run all explainers
    feature_importances = run_explainers(model, sub_x, sub_edge_index, center_idx, explainers)
    return feature_importances
