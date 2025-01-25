import pandas as pd
import os

def load_ellipticplusplus_actors(data_dir="data/raw/ellipticplusplus/actors/"):
    features_file = os.path.join(data_dir, "wallets_features.csv")
    classes_file = os.path.join(data_dir, "wallets_classes.csv")

    addr_addr_file = os.path.join(data_dir, "AddrAddr_edgelist.csv")
    addr_tx_file = os.path.join(data_dir, "AddrTx_edgelist.csv")
    tx_addr_file = os.path.join(data_dir, "TxAddr_edgelist.csv")

    # Load files
    features = pd.read_csv(features_file)
    classes = pd.read_csv(classes_file)
    addr_addr_edges = pd.read_csv(addr_addr_file)
    addr_tx_edges = pd.read_csv(addr_tx_file)
    tx_addr_edges = pd.read_csv(tx_addr_file)

    return features, classes, addr_addr_edges, addr_tx_edges, tx_addr_edges

if __name__ == "__main__":
    data = load_ellipticplusplus_actors()
    for d in data:
        print(d.head())
