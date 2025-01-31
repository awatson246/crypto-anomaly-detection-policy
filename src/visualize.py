import os
import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(G, folder="visualizations", filename="graph.png"):
    """Visualizes and saves the NetworkX graph in a specified folder."""
    
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    
    # Construct full file path
    file_path = os.path.join(folder, filename)
    
    # Plot the graph
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42, k=0.1)  # Force-directed layout
    nx.draw(G, pos, node_size=10, edge_color="gray", alpha=0.6, with_labels=False)
    
    # Save the image
    plt.title("Graph Visualization")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to prevent memory issues

    print(f"Graph saved at: {file_path}")
