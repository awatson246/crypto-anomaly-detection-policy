from src.graph_builder import load_data, build_graph
from src.visualize import visualize_graph

data_dir = "data"
users_df, wallets_df, transactions_df = load_data(data_dir)
G = build_graph(users_df, wallets_df, transactions_df)

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

visualize_graph(G)
