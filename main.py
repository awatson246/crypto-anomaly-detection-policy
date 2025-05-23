from src.graph_builder import load_data, build_graph
from src.feature_extraction import process_features
from src.anomaly_detection import detect_anomalies
from src.visualize import visualize_graph
from src.anomaly_detection import detect_anomalies
from src.graphlime_explainer import explain_anomaly
from src.llm_explainer import interpret_with_openai

data_dir = "data"

# Load raw data
wallets_df, transactions_df, edges_df = load_data(data_dir)
print("Data acquired!")

# Build graph
print("Building graph...")
G = build_graph(wallets_df, transactions_df, edges_df)
print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Extract or load features
node_features, edge_features = process_features(G)

# Run anomaly detection
df, top_anomalies, model = detect_anomalies(G, node_features)

# Save a visualization
visualize_graph(G, folder="results", filename="crypto_graph.png")

# Ask user to select an anomaly to explain
selected_idx = int(input("\nSelect an anomaly to explain (1-10): ")) - 1
selected_node = top_anomalies.index[selected_idx] 

print(f"\nExplaining anomaly: Node {selected_node}")

# Run GraphLIME explanation
num_features = len(["degree", "in_degree", "out_degree"]) 

explanation, top_features, node_data_dict, insight = explain_anomaly(G, model, selected_node, save_path="results/explanation.png")
print("\nLLM input:\n", insight)

# GPT explanation
llm_reasoning = interpret_with_openai(
    node_id=selected_node,
    top_features=top_features,
    node_data=node_data_dict  # dictionary of real feature values
)

print("\nLLM says:\n", llm_reasoning)


print("Explanation saved to results/explanation.png")