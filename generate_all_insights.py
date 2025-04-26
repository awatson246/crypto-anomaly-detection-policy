import os
import json
from tqdm import tqdm
from src.graphlime_explainer import explain_anomaly
from src.llm_explainer import interpret_with_openai
from src.graph_builder import load_data, build_graph
from src.feature_extraction import process_features
from src.anomaly_detection import detect_anomalies

def generate_llm_insights_for_top_anomalies(G, model, anomalies_df, out_path="results/llm_insights.json", top_n=100, dry_run=False):
    """
    Run GraphLIME + LLM insight generation for top N anomalies.
    Skips nodes with valid LLM insights already in the output file.
    """

    # Load existing file if available
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            insights = json.load(f)
        print(f"Loaded {len(insights)} existing insights from {out_path}")
    else:
        insights = {}

    print(f"\nGenerating LLM insights for top {top_n} anomalous nodes...\n")

    for i, (node_id, row) in tqdm(enumerate(anomalies_df.head(top_n).iterrows()), total=top_n):
        node_id_str = str(node_id)

        if node_id_str in insights and "No LLM interpretation available" not in insights[node_id_str]["llm_reasoning"]:
            print(f"Skipping node {node_id_str} (LLM already computed)")
            continue

        try:
            explanation, top_features, node_data_dict, insight = explain_anomaly(G, model, node_id)
            if dry_run:
                llm_output = "[DRY RUN] LLM interpretation would go here."
            else:
                llm_output = interpret_with_openai(node_id=node_id, top_features=top_features, node_data=node_data_dict)
        except Exception as e:
            print(f"Skipping node {node_id_str} due to error: {e}")
            llm_output = "No LLM interpretation available due to processing error."
            top_features = []
            insight = ""

        insights[node_id_str] = {
            "top_features": [(name, float(score)) for name, score in top_features],
            "insight_text": insight,
            "llm_reasoning": llm_output
        }

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(insights, f, indent=2)

    print(f"\nUpdated insights saved to: {out_path}")

    return insights

# Load and build
wallets_df, transactions_df, edges_df = load_data("data")
G = build_graph(wallets_df, transactions_df, edges_df)
node_features, edge_features = process_features(G, confirm_path="n")
_, anomalies_df, model = detect_anomalies(G, node_features)

# Generate JSON of insights
generate_llm_insights_for_top_anomalies(
    G, model, anomalies_df,
    out_path="results/llm_insights.json",
    top_n=10,
    dry_run=False  # Use True for testing without GPT
)

