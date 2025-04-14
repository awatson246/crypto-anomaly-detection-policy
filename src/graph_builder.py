import networkx as nx
from torch_geometric.utils import from_networkx
import pandas as pd
import os
import torch
from src.graphlime_explainer import FEATURE_COLUMNS

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

def standardize_graph_and_features(G, node_features):
    # Clean node attribute keys
    for node, attrs in G.nodes(data=True):
        cleaned_attrs = {k.replace(" ", "_"): v for k, v in attrs.items()}
        G.nodes[node].update(cleaned_attrs)

    # Clean DataFrame column names
    node_features.columns = [col.replace(" ", "_") for col in node_features.columns]

    # Ensure all nodes have all features
    for node in G.nodes():
        for feature in FEATURE_COLUMNS:
            if feature not in G.nodes[node]:
                G.nodes[node][feature] = 0.0  # or np.nan if preferred

    return G, node_features

def convert_to_pyg(G, node_features):
    from src.graphlime_explainer import FEATURE_COLUMNS  # or define locally if needed

    # Clean node_features column names
    node_features.columns = [col.replace(" ", "_") for col in node_features.columns]

    # Force every node to have the same clean attributes
    for node in G.nodes():
        fixed_attrs = {}
        for feature in FEATURE_COLUMNS:
            value = G.nodes[node].get(feature, 0.0)
            if not isinstance(value, (int, float)):
                value = 0.0
            fixed_attrs[feature] = float(value)
        
        G.nodes[node].clear()
        G.nodes[node].update(fixed_attrs)

    # Convert to PyG graph using only these attributes
    pyg_graph = from_networkx(G, group_node_attrs=FEATURE_COLUMNS)

    # Convert features DataFrame to tensor
    features = torch.tensor(node_features[FEATURE_COLUMNS].values, dtype=torch.float)

    return pyg_graph, features