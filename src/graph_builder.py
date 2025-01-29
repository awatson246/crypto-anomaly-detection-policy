import networkx as nx
import pandas as pd
import os

file_paths = {
    "wallets": "raw/wallets_features_classes_combined.csv",
    "addr_addr": "raw/AddrAddr_edgelist.csv",
    "addr_tx": "raw/AddrTx_edgelist.csv",
    "tx_addr": "raw/TxAddr_edgelist.csv",
    "txs_classes": "raw/txs_classes.csv",
    "txs_edgelist": "raw/txs_edgelist.csv",
    "txs_features": "raw/txs_features.csv",
}

def load_data(base_dir=""):
    """Loads CSV files into DataFrames, ensuring correct path formatting."""
    data = {}
    for key, path in file_paths.items():
        full_path = os.path.normpath(os.path.join(base_dir, path))  # Normalize path to handle slashes
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
        data[key] = pd.read_csv(full_path)
    return data


def build_graph(data):
    """Constructs a NetworkX graph from the dataset."""
    G = nx.Graph()

    # Add wallet nodes with features
    wallets_df = data["wallets"]
    for _, row in wallets_df.iterrows():
        G.add_node(row["wallet_id"], type="wallet", **row.to_dict())

    # Add transaction nodes with features
    txs_features_df = data["txs_features"]
    for _, row in txs_features_df.iterrows():
        G.add_node(row["tx_id"], type="transaction", **row.to_dict())

    # Add edges from different relationships
    edge_types = ["addr_addr", "addr_tx", "tx_addr", "txs_edgelist"]
    for edge_type in edge_types:
        edges_df = data[edge_type]
        for _, row in edges_df.iterrows():
            G.add_edge(row["source"], row["target"], type=edge_type, **row.to_dict())

    return G

if __name__ == "__main__":
    data = load_data()
    graph = build_graph(data)
    print(f"Graph created with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
