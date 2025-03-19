import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx
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

def explain_anomaly(G, model, anomaly_node, save_path=None):
    """
    Generate a GraphLIME explanation for a single anomaly node in a GNN.

    Parameters:
        G (networkx.Graph): The graph.
        model (torch.nn.Module): Trained GNN anomaly detection model.
        anomaly_node (int): The node to explain.
        save_path (str, optional): Path to save visualization.

    Returns:
        explanation (torch.Tensor): Feature importance scores.
    """
    # Ensure all nodes have the same attributes (only the ones from FEATURE_COLUMNS)
    for n in G.nodes:
        for key in FEATURE_COLUMNS:
            if key not in G.nodes[n]:  
                G.nodes[n][key] = 0  # Assign default value

    # Convert non-numeric attributes to numeric values
    def convert_value(val):
        if isinstance(val, (int, float)):  
            return val
        elif isinstance(val, str):  
            return hash(val) % 100000  # Convert strings to numbers
        return 0  # Default fallback

    # Prepare feature matrix (using only FEATURE_COLUMNS)
    node_features_list = [[convert_value(G.nodes[n][key]) for key in FEATURE_COLUMNS] for n in G.nodes]
    
    # Convert NetworkX graph to PyG format with the correct features
    pyg_graph = from_networkx(G)
    pyg_graph.x = torch.tensor(node_features_list, dtype=torch.float)  # Attach features to PyG graph

    # Debug prints
    print(f"GraphLIME input feature shape: {pyg_graph.x.shape}")
    print(f"Model expects input features: {model.conv1.in_channels}")

    # Ensure feature dimensions match
    if pyg_graph.x.shape[1] != model.conv1.in_channels:
        raise ValueError(f"Feature mismatch: Graph has {pyg_graph.x.shape[1]} features, but model expects {model.conv1.in_channels}.")

    # Ensure the model is in evaluation mode
    model.eval()

    # Locate node index
    node_idx = list(G.nodes()).index(anomaly_node)
    
    # Run GraphLIME
    graphlime = GraphLIME(model)
    explanation = graphlime.explain_node(node_idx, pyg_graph.x, pyg_graph.edge_index)
    
    # Extract feature importance scores
    feature_importance = explanation[node_idx].detach().numpy()

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
