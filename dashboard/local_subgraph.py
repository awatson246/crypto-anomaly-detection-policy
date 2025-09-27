import networkx as nx
import json

class LocalSubgraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.node_data = {}
        self.insights = {}

    def initialize_from_query(self, G, node_features, insights_dict, filter_label="anomaly"):
        """Start with only nodes matching filter (e.g. anomalies)."""
        nodes = [
            n for n, row in node_features.iterrows()
            if str(row.get("anomaly_label", 0)) == "1"
        ] if filter_label == "anomaly" else list(G.nodes())

        self.graph = G.subgraph(nodes).copy()

        # Store node features
        for n in self.graph.nodes():
            self.node_data[n] = node_features.loc[str(n)].to_dict()

        # Store insights (only for selected nodes)
        self.insights = {k: v for k, v in insights_dict.items() if k in self.graph.nodes()}

    def to_json(self, out_dir="dashboard/dashboard_data"):
        edge_data = [{"source": str(u), "target": str(v)} for u, v in self.graph.edges()]
        node_data = {str(n): self.node_data[n] for n in self.graph.nodes()}
        with open(f"{out_dir}/graph.json", "w") as f:
            json.dump(edge_data, f, indent=2)
        with open(f"{out_dir}/node_data.json", "w") as f:
            json.dump(node_data, f, indent=2)
        with open(f"{out_dir}/llm_insights.json", "w") as f:
            json.dump(self.insights, f, indent=2)
