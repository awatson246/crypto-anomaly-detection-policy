from src.graph_builder import load_data, build_graph
from src.feature_extraction import process_features
from src.anomaly_detection import detect_anomalies
from src.visualize import visualize_graph
from src.anomaly_detection import detect_anomalies
from src.graphlime_explainer import explain_anomaly

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

# Run anomaly detection
df, top_anomalies, model = detect_anomalies(G, node_features)

# Save a visualization
visualize_graph(G, folder="results", filename="crypto_graph.png")

# Ask user to select an anomaly to explain
selected_idx = int(input("\nSelect an anomaly to explain (1-10): ")) - 1
selected_node = top_anomalies.iloc[selected_idx]["node"]

print(f"\nExplaining anomaly: Node {selected_node}")

# Run GraphLIME explanation
num_features = len(["degree", "in_degree", "out_degree"])  # Adjust based on actual features
explanation = explain_anomaly(G, model, selected_node, num_features, save_path="results/explanation.png")

print("Explanation saved to results/explanation.png")