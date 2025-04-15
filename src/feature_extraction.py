import os
import pandas as pd
import networkx as nx

FEATURES_DIR = "features"
NODE_FEATURES_FILE = os.path.join(FEATURES_DIR, "node_features.csv")
EDGE_FEATURES_FILE = os.path.join(FEATURES_DIR, "edge_features.csv")

def extract_node_features(G):
    """Extracts node-level features and ensures consistent attributes for all nodes."""
    node_data = []
    
    # Define expected feature keys
    default_features = {
        "degree": 0,
        "in_degree": 0,
        "out_degree": 0,
        "node_type": "unknown",
    }

    for node, attrs in G.nodes(data=True):
        # Compute graph-based features
        degree = G.degree(node)
        in_degree = sum(1 for _ in G.predecessors(node)) if isinstance(G, nx.DiGraph) else 0
        out_degree = sum(1 for _ in G.successors(node)) if isinstance(G, nx.DiGraph) else 0
        
        # Merge node attributes with defaults
        node_features = {**default_features, **attrs}  # Ensures missing fields get defaults
        node_features.update({
            "node": node,
            "degree": degree,
            "in_degree": in_degree,
            "out_degree": out_degree,
        })

        node_data.append(node_features)

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
    """Loads saved features with consistent data types."""
    if os.path.exists(NODE_FEATURES_FILE) and os.path.exists(EDGE_FEATURES_FILE):
        user_input = input("Existing features found. Reuse them? (y/n): ").strip().lower()
        if user_input == "y":
            print("Loading existing features...")

            node_features = pd.read_csv(NODE_FEATURES_FILE, low_memory=False, dtype={
                "node": str,  # Ensure nodes are treated as strings
                "degree": int,
                "in_degree": int,
                "out_degree": int,
                "node_type": str,
            })
            edge_features = pd.read_csv(EDGE_FEATURES_FILE, low_memory=False, dtype={
                "source": str,  # Ensure edge source/target are strings
                "target": str,
                "edge_type": str,
            })

            return node_features, edge_features

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
