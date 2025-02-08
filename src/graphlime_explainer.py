import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx
from graphlime import GraphLIME

def explain_anomaly(G, model, anomaly_node, num_features, save_path=None):
    """
    Generate a GraphLIME explanation for a single anomaly node.

    Parameters:
        G (networkx.Graph): The graph.
        model (torch.nn.Module): Trained anomaly detection model.
        anomaly_node (str): The node to explain.
        num_features (int): Number of node features.
        save_path (str, optional): Path to save visualization.

    Returns:
        explanation (dict): Feature importance scores.
    """
    # Convert NetworkX graph to PyG format
    pyg_graph = from_networkx(G)
    
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Extract node features and run through the model
    node_idx = list(G.nodes()).index(anomaly_node)
    features = torch.tensor([list(G.nodes[n].values()) for n in G.nodes()], dtype=torch.float)
    
    # Apply GraphLIME for explanation
    graphlime = GraphLIME(num_hops=2)
    explanation = graphlime.explain_node(node_idx, features, pyg_graph.edge_index, model)
    
    # Plot the explanation
    plt.figure(figsize=(8, 6))
    plt.bar(range(num_features), explanation[node_idx])
    plt.xlabel("Feature Index")
    plt.ylabel("Importance Score")
    plt.title(f"GraphLIME Explanation for Node {anomaly_node}")
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    return explanation[node_idx]
