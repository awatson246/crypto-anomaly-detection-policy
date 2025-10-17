import json
import os
import networkx as nx
import math
import pandas as pd
import numpy as np

def sanitize_for_json(obj):
    """Recursively convert NaN/inf to None in nested structures."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj

def export_mini_dashboard_graph(G, node_features, insights_dict, out_dir="dashboard/dashboard_data", k_hops=2, max_central_nodes=35):
    """
    Extracts a smaller graph centered around nodes with LLM insights and exports:
    - graph.json
    - node_data.json
    - llm_insights.json
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Select top LLM-explained anomalies ---
    center_nodes = [
        n for n in insights_dict
        if "No LLM interpretation available" not in insights_dict[n]["llm_reasoning"]
    ][:max_central_nodes]

    print(f"Selected {len(center_nodes)} central anomaly nodes for mini graph.")

    # Collect k-hop neighbors
    subgraph_nodes = set()
    for node in center_nodes:
        if node in G:
            neighbors = nx.single_source_shortest_path_length(G, node, cutoff=k_hops).keys()
            subgraph_nodes.update(neighbors)

    G_sub = G.subgraph(subgraph_nodes).copy()
    print(f"Mini graph has {len(G_sub.nodes)} nodes and {len(G_sub.edges)} edges.")

    # Save edge list
    edge_data = [
        {"source": str(u), "target": str(v)}
        for u, v in G_sub.edges()
    ]
    with open(os.path.join(out_dir, "graph.json"), "w") as f:
        json.dump(edge_data, f, indent=2)

    # Safe conversion for JSON
    def to_serializable(val):
        # Safely convert values for JSON
        if pd.isna(val):
            return None
        if hasattr(val, "item"):
            return val.item()
        if isinstance(val, (np.int64, np.float64, np.bool_)):
            return val.tolist()
        return val

    # Node metadata
    mini_node_data = {}
    for node in G_sub.nodes():
        node_str = str(node)
        if node_str not in node_features.index:
            print(f"[WARN] Node {node_str} not in node_features index")
            continue

        row = node_features.loc[node_str]
        score = row.get("anomaly_score", None)
        label = row.get("anomaly_label", 0)

        mini_node_data[node_str] = {
            "anomaly_score": to_serializable(score),
            "is_anomalous": bool(int(label)) if not pd.isna(label) else False,
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

    # Filtered LLM insights
    mini_insights = {
        k: v for k, v in insights_dict.items() if k in G_sub.nodes
    }
    with open(os.path.join(out_dir, "llm_insights.json"), "w") as f:
        json.dump(mini_insights, f, indent=2)

    print(f"Export complete. Files saved to '{out_dir}'")