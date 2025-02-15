import networkx as nx
from torch_geometric.utils import from_networkx
import pandas as pd
import os
import torch

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
    """Loads CSV files into structured DataFrames for users, wallets, and transactions."""
    data = {}
    for key, path in file_paths.items():
        full_path = os.path.normpath(os.path.join(base_dir, path))  # Normalize path format
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
        data[key] = pd.read_csv(full_path)

    # Structuring output: Users, Wallets, Transactions
    wallets_df = data["wallets"]  # Wallet feature data
    transactions_df = data["txs_edgelist"]  # Transactions between entities
    edges_df = {
        "addr_addr": data["addr_addr"],
        "addr_tx": data["addr_tx"],
        "tx_addr": data["tx_addr"]
    }

    return wallets_df, transactions_df, edges_df 

def build_graph(wallets_df, transactions_df, edges_df):
    """Constructs a NetworkX graph efficiently."""
    G = nx.Graph()

    wallet_nodes = [(row["address"], {"type": "wallet", **row.to_dict()}) for _, row in wallets_df.iterrows()]
    G.add_nodes_from(wallet_nodes)

    transaction_nodes = set(transactions_df["txId1"]).union(set(transactions_df["txId2"]))
    G.add_nodes_from((tx_id, {"type": "transaction"}) for tx_id in transaction_nodes)

    column_mappings = {
        "addr_addr": ("input_address", "output_address"),
        "addr_tx": ("input_address", "txId"),
        "tx_addr": ("txId", "output_address"),
    }

    for edge_type, df in edges_df.items():
        df.columns = df.columns.str.strip()  # Clean column names
        
        if edge_type not in column_mappings:
            print(f"Skipping unknown edge type: {edge_type}")
            continue
        
        src_col, tgt_col = column_mappings[edge_type]
        
        if src_col not in df.columns or tgt_col not in df.columns:
            print(f"Skipping {edge_type} due to missing columns: {df.columns}")
            continue
        
        edge_list = [(row[src_col], row[tgt_col], {"type": edge_type}) for _, row in df.iterrows()]
        G.add_edges_from(edge_list) 

    return G

def standardize_node_attributes(G):
    """Ensures all nodes have the same attributes by filling missing values with defaults."""
    all_attributes = set()
    
    # Collect all possible attributes across nodes
    for _, attrs in G.nodes(data=True):
        all_attributes.update(attrs.keys())

    # Assign default values (0 for numerical, empty string otherwise)
    for node in G.nodes():
        for attr in all_attributes:
            if attr not in G.nodes[node]:
                G.nodes[node][attr] = 0  # Default value for missing attributes
    
    return G

def convert_to_pyg(G, node_features):
    """Converts NetworkX graph and node features to PyTorch Geometric format."""
    # Standardize node attributes
    G = standardize_node_attributes(G)

    # Convert graph structure
    pyg_graph = from_networkx(G)
    
    # Convert node features to a PyTorch tensor
    feature_columns = ["degree", "in_degree", "out_degree"]  # Adjust based on actual features
    features = torch.tensor(node_features[feature_columns].values, dtype=torch.float)

    return pyg_graph, features
