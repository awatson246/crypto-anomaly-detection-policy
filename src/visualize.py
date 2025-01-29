import networkx as nx
import matplotlib.pyplot as plt
from src.graph_builder import build_graph, load_data

def visualize_graph(G, save_path=None):
    """Simple visualization of the graph."""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)  # Layout for visualization
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=8)
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    data_dir = "../data"
    users_df, wallets_df, transactions_df = load_data(data_dir)
    G = build_graph(users_df, wallets_df, transactions_df)
    visualize_graph(G)
