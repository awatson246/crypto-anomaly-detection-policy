import pandas as pd

# Define file paths (update as needed)
file_paths = {
    "wallets_features_classes_combines": "data/raw/wallets_features_classes_combined.csv",
    "addr_addr": "data/raw\AddrAddr_edgelist.csv",
    "addr_tx": "data/raw\AddrTx_edgelist.csv",
    "tx_addr": "data/raw\TxAddr_edgelist.csv",
    "txs_classes": "data/raw/txs_classes.csv",
    "txs_edgelist": "data/raw/txs_edgelist.csv",
    "txs_features": "data/raw/txs_features.csv"
}

# Load datasets
datasets = {name: pd.read_csv(path) for name, path in file_paths.items()}

# Preview each dataset
for name, df in datasets.items():
    print(f"\n--- {name} ---")
    print(df.head())
    print(df.info())
    print(f"Missing values:\n{df.isnull().sum()}")
