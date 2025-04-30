import json
import pandas as pd
import os
import sys
from dashboard_export import export_mini_dashboard_graph

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.graph_builder import load_data, build_graph

# Load graph data
data_dir = "data"
wallets_df, transactions_df, edges_df = load_data(data_dir)
G = build_graph(wallets_df, transactions_df, edges_df)

# Load node features
node_features = pd.read_csv("features/node_features.csv", low_memory=False).set_index("node")
node_features = node_features.copy()
node_features.index = node_features.index.astype(str)

# Load previously generated LLM insights
with open("results/llm_insights.json", "r") as f:
    all_insights_dict = json.load(f)

# Export dashboard-ready files

export_mini_dashboard_graph(G, node_features, all_insights_dict, k_hops=1, max_central_nodes=15)
