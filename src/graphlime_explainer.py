import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx, k_hop_subgraph
from graphlime import GraphLIME

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

def explain_anomaly(G, model, anomaly_node, save_path=None, k_hops=2):
    """
    Generate a GraphLIME explanation for a single anomaly node using a subgraph.
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from torch_geometric.utils import from_networkx, k_hop_subgraph
    from graphlime import GraphLIME

    MAX_SUBGRAPH_NODES = 2000  # 👈 Set your memory-safe max here

    # Ensure node ID types match (NetworkX might use string keys)
    if isinstance(list(G.nodes())[0], str):
        anomaly_node = str(anomaly_node)
    else:
        anomaly_node = int(anomaly_node)

    if anomaly_node not in G:
        print(f"Warning: Node {anomaly_node} not found in G!")
        print(f"Available node IDs (first 10): {list(G.nodes())[:10]}")
        raise ValueError(f"Node {anomaly_node} not found in graph!")

    # Add degree-based features (you can skip if already added earlier)
    for node in G.nodes():
        G.nodes[node]['degree'] = G.degree(node)

    # Filter numeric features
    sample_node = next(iter(G.nodes()))
    sample_attrs = G.nodes[sample_node]
    numeric_keys = [k for k, v in sample_attrs.items() if isinstance(v, (int, float, np.float64, np.int64))]

    # Convert to PyG
    pyg_graph = from_networkx(G, group_node_attrs=numeric_keys)
    if not hasattr(pyg_graph, 'x') or pyg_graph.x is None:
        raise ValueError("PyG graph conversion failed: No node features found!")

    print(f"Converted to PyG. PyG has {pyg_graph.x.shape[0]} nodes.")

    # Get the node's index
    node_idx = list(G.nodes()).index(anomaly_node)
    node_idx_tensor = torch.tensor([node_idx], dtype=torch.long)

    # Get k-hop subgraph
    subset, edge_index, mapping, _ = k_hop_subgraph(
        node_idx=node_idx_tensor, num_hops=k_hops, edge_index=pyg_graph.edge_index, relabel_nodes=True
    )

    print(f"Subgraph extracted: {len(subset)} nodes.")

    # 🔻 Trim the subgraph if it's too big
    if len(subset) > MAX_SUBGRAPH_NODES:
        print(f"Subgraph exceeds {MAX_SUBGRAPH_NODES} nodes. Trimming for memory safety.")
        # Always include the center node
        center_idx = mapping.item()
        trimmed_subset = subset[:MAX_SUBGRAPH_NODES]
        trimmed_edge_index = edge_index[:, (edge_index[0].isin(trimmed_subset) & edge_index[1].isin(trimmed_subset))]

        # Reindex to ensure center node is still present
        if center_idx not in trimmed_subset:
            trimmed_subset[0] = subset[center_idx]  # force-inject center node
            center_idx = 0  # it's now at position 0
        else:
            center_idx = (trimmed_subset == subset[center_idx]).nonzero(as_tuple=True)[0].item()

        subset = trimmed_subset
        edge_index = trimmed_edge_index
    else:
        center_idx = mapping.item()

    sub_x = pyg_graph.x[subset]
    sub_edge_index = edge_index

    in_features = model.conv1.in_channels
    if sub_x.shape[1] != in_features:
        raise ValueError(f"Feature mismatch: Subgraph has {sub_x.shape[1]} features, but model expects {in_features}.")

    # Run GraphLIME
    graphlime = GraphLIME(model)
    explanation = graphlime.explain_node(center_idx, sub_x, sub_edge_index)
    feature_importance = explanation[center_idx].detach().numpy()

    # Plot explanation
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance Score")
    plt.title(f"GraphLIME Explanation for Node {anomaly_node}")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    return feature_importance
