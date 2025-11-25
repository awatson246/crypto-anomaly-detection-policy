import networkx as nx
from torch_geometric.utils import from_networkx
import pandas as pd
import os
import torch
from src.graphlime_explainer import FEATURE_COLUMNS

file_paths = {
    "wallets": "raw/wallets_features.csv",
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
    """
    Construct a *faithful* address–transaction heterogeneous graph.
    Uses MultiDiGraph to preserve multi-edges and directions.
    """

    # ----------------------------------------------------
    # 1. Multi-edge, directed heterogeneous graph
    # ----------------------------------------------------
    G = nx.MultiDiGraph()

    # ----------------------------------------------------
    # 2. Add Wallet/Address Nodes
    # ----------------------------------------------------
    # Wallet nodes (wallet-level features)
    for _, row in wallets_df.iterrows():
        addr = row["address"]
        G.add_node(addr, type="wallet", **row.to_dict())

    # Transaction nodes
    tx_nodes = set(transactions_df["txId1"]).union(set(transactions_df["txId2"]))
    for tx_id in tx_nodes:
        G.add_node(tx_id, type="transaction")

    # ----------------------------------------------------
    # 3. Add Edges for Three Edge Lists
    # ----------------------------------------------------
    column_mappings = {
        "addr_addr": ("input_address", "output_address"),
        "addr_tx": ("input_address", "txId"),
        "tx_addr": ("txId", "output_address"),
    }

    for edge_type, df in edges_df.items():
        df.columns = df.columns.str.strip()

        if edge_type not in column_mappings:
            print(f"Skipping unknown edge type: {edge_type}")
            continue

        src_col, tgt_col = column_mappings[edge_type]

        if src_col not in df.columns or tgt_col not in df.columns:
            print(f"[WARN] {edge_type} missing columns: {df.columns}")
            continue

        # Preserve all edges (multi-edge!)
        for _, row in df.iterrows():
            src = row[src_col]
            tgt = row[tgt_col]
            G.add_edge(src, tgt, type=edge_type)

    # ----------------------------------------------------
    # 4. Compute NEW interpretable structural features
    # ----------------------------------------------------
    for node in G.nodes():
        neighbors = set(G.neighbors(node))

        G.nodes[node]["degree_multi"] = G.degree(node)  # preserves multi-edge count
        G.nodes[node]["unique_neighbors"] = len(neighbors)

        # Transaction-role features
        G.nodes[node]["unique_senders"] = len([n for n in neighbors if G.has_edge(n, node)])
        G.nodes[node]["unique_receivers"] = len([n for n in neighbors if G.has_edge(node, n)])

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

    return pyg_graph, features, list(G.nodes())