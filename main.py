from src.graph_builder import load_data, build_graph
from src.visualize import visualize_graph

data_dir = "data"
wallets_df, transactions_df, edges_df = load_data(data_dir)
print("Data aquired!")

print("Building graph...")
G = build_graph(wallets_df, transactions_df, edges_df)

visualize_graph(G, folder="results", filename="crypto_graph.png")
