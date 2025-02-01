import os
import pandas as pd
import networkx as nx

FEATURES_DIR = "features"
NODE_FEATURES_FILE = os.path.join(FEATURES_DIR, "node_features.csv")
EDGE_FEATURES_FILE = os.path.join(FEATURES_DIR, "edge_features.csv")

def extract_node_features(G):
    """Extracts node-level features and returns a DataFrame."""
    node_data = []

    for node, attrs in G.nodes(data=True):
        degree = G.degree(node)
        in_degree = sum(1 for _ in G.predecessors(node)) if isinstance(G, nx.DiGraph) else None
        out_degree = sum(1 for _ in G.successors(node)) if isinstance(G, nx.DiGraph) else None
        node_type = attrs.get("type", "unknown")

        node_data.append({
            "node": node,
            "degree": degree,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "node_type": node_type,
            **{k: v for k, v in attrs.items() if k not in ["type"]}
        })

    return pd.DataFrame(node_data)

def extract_edge_features(G):
    """Extracts edge-level features and returns a DataFrame."""
    edge_data = []

    for source, target, attrs in G.edges(data=True):
        edge_type = attrs.get("type", "unknown")

        edge_data.append({
            "source": source,
            "target": target,
            "edge_type": edge_type,
            **{k: v for k, v in attrs.items() if k not in ["type"]}
        })

    return pd.DataFrame(edge_data)

def save_features(df, file_path):
    """Saves a DataFrame to a CSV file."""
    os.makedirs(FEATURES_DIR, exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Saved features to {file_path}")

def load_features():
    """Checks for saved features and loads them if available."""
    if os.path.exists(NODE_FEATURES_FILE) and os.path.exists(EDGE_FEATURES_FILE):
        user_input = input("Existing features found. Reuse them? (y/n): ").strip().lower()
        if user_input == "y":
            print("Loading existing features...")
            return pd.read_csv(NODE_FEATURES_FILE), pd.read_csv(EDGE_FEATURES_FILE)
    
    print("Extracting new features...")
    return None, None

def process_features(G):
    """Handles feature extraction or loading based on user input."""
    node_features, edge_features = load_features()
    
    if node_features is None or edge_features is None:
        node_features = extract_node_features(G)
        edge_features = extract_edge_features(G)
        save_features(node_features, NODE_FEATURES_FILE)
        save_features(edge_features, EDGE_FEATURES_FILE)

    return node_features, edge_features
