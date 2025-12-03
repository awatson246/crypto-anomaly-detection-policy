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

def export_mini_dashboard_graph(
    G,
    node_features,
    insights_dict,
    out_dir="dashboard/dashboard_data",
    k_hops=2,
    max_central_nodes=35
):
    """
    Extracts a smaller graph centered around nodes with valid LLM insights and exports:
    - graph.json
    - node_data.json
    - llm_insights.json
    """

    os.makedirs(out_dir, exist_ok=True)

    # 1. Identify nodes with valid LLM consensus
    def get_consensus(ins):
        if not isinstance(ins, dict):
            return None
        block = ins.get("llm_output_structured", {})
        consensus = block.get("consensus", None)
        if isinstance(consensus, dict):
            return consensus
        return None

    center_nodes = [
        n for n, ins in insights_dict.items() if get_consensus(ins) is not None
    ]
    center_nodes = center_nodes[:max_central_nodes]

    print(f"Selected {len(center_nodes)} LLM-explained anomaly nodes for mini graph.")

    # 2. Build k-hop ego graph
    subgraph_nodes = set()
    for node in center_nodes:
        if node in G:
            neighbors = nx.single_source_shortest_path_length(G, node, cutoff=k_hops).keys()
            subgraph_nodes.update(neighbors)
        else:
            print(f"[WARN] Insight node {node} not in graph.")

    G_sub = G.subgraph(subgraph_nodes).copy()
    print(f"Mini graph has {len(G_sub.nodes)} nodes and {len(G_sub.edges)} edges.")

    # 3. Save graph edges
    edges_out = [
        {"source": str(u), "target": str(v)}
        for u, v in G_sub.edges()
    ]
    with open(os.path.join(out_dir, "graph.json"), "w") as f:
        json.dump(edges_out, f, indent=2)

    # 4. Node metadata including LLM insight fields
    def to_serializable(val):
        if pd.isna(val):
            return None
        if hasattr(val, "item"):
            return val.item()
        if isinstance(val, (np.int64, np.float64, np.bool_)):
            return val.tolist()
        return val

    mini_node_data = {}

    for node in G_sub.nodes():
        node_str = str(node)

        if node_str not in node_features.index:
            print(f"[WARN] Node {node_str} missing from node_features — skipping.")
            continue

        row = node_features.loc[node_str]
        score = row.get("anomaly_score", None)
        label = row.get("anomaly_label", 0)

        entry = {
            "anomaly_score": to_serializable(score),
            "is_anomalous": bool(int(label)) if not pd.isna(label) else False,
            "has_llm": node_str in center_nodes,
            "feature_values": {
                k: to_serializable(v)
                for k, v in row.items()
                if k not in ["anomaly_score", "anomaly_label"]
            }
        }

        # Add LLM structured consensus info
        if node_str in insights_dict:
            ins = insights_dict[node_str]
            consensus = get_consensus(ins)

            if consensus is not None:
                entry["llm"] = {
                    "is_fraud": consensus.get("is_fraud"),
                    "fraud_type": consensus.get("fraud_type"),
                    "agreement_rate": consensus.get("agreement_rate"),
                    "fraud_type_agreement": consensus.get("fraud_type_agreement"),
                    "avg_confidence": consensus.get("avg_confidence")
                }

                # GraphLIME top features (for bar charts)
                entry["graphlime_top_features"] = [
                    {"feature": ft, "importance": float(val)}
                    for ft, val in ins.get("top_features", [])
                ]

                # Attach full raw samples if needed for UI popup
                samples = ins["llm_output_structured"].get("samples", [])
                entry["llm_samples"] = samples

                # Include node_data so UI panels can show raw stats
                entry["node_data"] = ins.get("node_data", {})

        mini_node_data[node_str] = entry

    with open(os.path.join(out_dir, "node_data.json"), "w") as f:
        json.dump(mini_node_data, f, indent=2)

    # 5. Only LLM insights belonging to the subgraph
    mini_insights = {k: v for k, v in insights_dict.items() if k in G_sub.nodes()}
    with open(os.path.join(out_dir, "llm_insights.json"), "w") as f:
        json.dump(mini_insights, f, indent=2)

    print(f"Export complete. Files saved to '{out_dir}'")
