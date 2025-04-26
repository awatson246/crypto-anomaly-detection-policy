import json
import os
import networkx as nx

def export_mini_dashboard_graph(G, node_features, insights_dict, out_dir="dashboard/dashboard_data", k_hops=1):
    """
    Extracts a smaller graph centered around nodes with LLM insights and exports:
    - graph.json
    - node_data.json
    - llm_insights.json
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Get seed nodes (anomalies with LLM insights) ---
    center_nodes = [n for n in insights_dict if "No LLM interpretation available" not in insights_dict[n]["llm_reasoning"]]

    # --- Collect k-hop neighbors ---
    subgraph_nodes = set(center_nodes)
    for node in center_nodes:
        if node in G:
            neighbors = nx.single_source_shortest_path_length(G, node, cutoff=k_hops).keys()
            subgraph_nodes.update(neighbors)

    G_sub = G.subgraph(subgraph_nodes)

    # --- Export edge list ---
    edge_data = [
        {"source": str(u), "target": str(v)}
        for u, v in G_sub.edges()
    ]
    with open(os.path.join(out_dir, "graph.json"), "w") as f:
        json.dump(edge_data, f, indent=2)

    # --- Export node metadata ---
    def to_serializable(val):
        return val.item() if hasattr(val, "item") else val

    mini_node_data = {}
    for node in G_sub.nodes():
        node_str = str(node)
        if node_str not in node_features.index:
            continue

        row = node_features.loc[node_str]
        mini_node_data[node_str] = {
            "anomaly_score": to_serializable(row.get("anomaly_score", None)),
            "is_anomalous": bool(row.get("anomaly_label", 0)),
            "feature_values": {
                k: to_serializable(v)
                for k, v in row.items()
                if k not in ["anomaly_score", "anomaly_label"]
            },
            "has_llm": (
                node_str in insights_dict
                and "No LLM interpretation available" not in insights_dict[node_str].get("llm_reasoning", "")
            )
        }

    with open(os.path.join(out_dir, "node_data.json"), "w") as f:
        json.dump(mini_node_data, f, indent=2)

    # --- Export insights ---
    mini_insights = {
        k: v for k, v in insights_dict.items() if k in G_sub.nodes
    }

    with open(os.path.join(out_dir, "llm_insights.json"), "w") as f:
        json.dump(mini_insights, f, indent=2)

    print(f"Exported mini graph with {len(G_sub.nodes)} nodes and {len(G_sub.edges)} edges.")
