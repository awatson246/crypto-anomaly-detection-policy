import os
import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(G, folder="results", filename="graph.png", max_nodes=500):
    """Visualizes and saves the NetworkX graph in a specified folder."""
    
    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, filename)

    if len(G) > max_nodes:
        sampled_nodes = list(G.nodes)[:max_nodes]  # Take a subset of nodes
        H = G.subgraph(sampled_nodes)
    else:
        H = G  # Use the full graph if it's small enough

    print(f"Using {len(H)} nodes for visualization.")

    try:
        pos = nx.spring_layout(H, seed=42, k=0.1, iterations=50)
    except Exception as e:
        print(f"Error computing layout: {e}")
        return

    print("Plotting the graph...")
    plt.figure(figsize=(12, 12))
    nx.draw(H, pos, node_size=10, edge_color="gray", alpha=0.6, with_labels=False)

    plt.title("Graph Visualization")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Graph saved at: {file_path}")
