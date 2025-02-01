from src.graph_builder import load_data, build_graph
from src.feature_extraction import process_features
from src.anomaly_detection import detect_anomalies
from src.visualize import visualize_graph

data_dir = "data"

# Load raw data
wallets_df, transactions_df, edges_df = load_data(data_dir)
print("Data acquired!")

# Build graph
print("Building graph...")
G = build_graph(wallets_df, transactions_df, edges_df)
print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Extract or load features
node_features, edge_features = process_features(G)

# Detect anomalies
df, anomalies = detect_anomalies()

# Optional: Save a visualization
visualize_graph(G, folder="results", filename="crypto_graph.png")
