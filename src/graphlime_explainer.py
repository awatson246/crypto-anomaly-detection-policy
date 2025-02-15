import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx
from graphlime import GraphLIME

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
    # Ensure all nodes have the same attributes
    all_keys = set().union(*(G.nodes[n].keys() for n in G.nodes))  
    for n in G.nodes:
        for key in all_keys:
            if key not in G.nodes[n]:  
                G.nodes[n][key] = 0  # Assign default value

    # Convert non-numeric attributes to numeric values
    def convert_value(val):
        if isinstance(val, (int, float)):  
            return val
        elif isinstance(val, str):  
            return hash(val) % 100000  # Convert strings to numbers
        return 0  # Default fallback

    # Prepare feature matrix
    node_features_list = [[convert_value(G.nodes[n][key]) for key in all_keys] for n in G.nodes]
    
    # Convert NetworkX graph to PyG format
    pyg_graph = from_networkx(G)
    pyg_graph.x = torch.tensor(node_features_list, dtype=torch.float)  # Attach features to PyG graph
    
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
