import os
import json
import time
from tqdm import tqdm
from src.graphlime_explainer import explain_anomaly
from src.llm_explainer import interpret_with_openai
from src.graph_builder import load_data, build_graph
from src.feature_extraction import process_features
from src.anomaly_detection import detect_anomalies
import torch

def generate_llm_insights_for_top_anomalies(
    G,
    model,
    anomalies_df,
    out_path="results/llm_insights.json",
    metrics_path="results/runtime_cost_log.json",
    top_n=100,
    dry_run=False
):
    # Load cached LLM insights
    if os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                text = f.read().strip()
                insights = json.loads(text) if text else {}
        except json.JSONDecodeError:
            insights = {}
    else:
        insights = {}

    # Load metrics log
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                text = f.read().strip()
                metrics_log = json.loads(text) if text else []
        except:
            metrics_log = []
    else:
        metrics_log = []

    os.makedirs("results", exist_ok=True)

    print(f"\nGenerating insights for top {top_n} anomalies...\n")

    for i, (node_id, row) in tqdm(enumerate(anomalies_df.head(top_n).iterrows()), total=top_n):
        node_id_str = str(node_id)

        if node_id_str in insights:
            continue

        # ---- GNN time ----
        gnn_start = time.time()
        score = model(torch.tensor([node_id])) if hasattr(model, "__call__") else None
        gnn_time = time.time() - gnn_start

        # ---- GraphLIME time ----
        gl_start = time.time()
        try:
            explanation, top_features, node_data_dict, insight = explain_anomaly(G, model, node_id)
        except Exception as e:
            top_features, insight, node_data_dict = [], "", {}
        gl_time = time.time() - gl_start

        # ---- LLM time ----
        if dry_run:
            llm_output = "[DRY RUN]"
            llm_latency = 0
            in_tok = out_tok = 0
            cost = 0
        else:
            llm_output, llm_latency, in_tok, out_tok, cost = interpret_with_openai(
                node_id=node_id,
                top_features=top_features,
                node_data=node_data_dict
            )

        # Save output
        insights[node_id_str] = {
            "top_features": [(name, float(v)) for name, v in top_features],
            "insight_text": insight,
            "llm_reasoning": llm_output
        }

        with open(out_path, "w") as f:
            json.dump(insights, f, indent=2)

        # Save metrics
        metrics_log.append({
            "node_id": node_id_str,
            "gnn_time_s": gnn_time,
            "graphlime_time_s": gl_time,
            "llm_latency_s": llm_latency,
            "llm_input_tokens": in_tok,
            "llm_output_tokens": out_tok,
            "llm_cost_usd": cost,
            "total_time_s": gnn_time + gl_time + llm_latency,
        })

        with open(metrics_path, "w") as f:
            json.dump(metrics_log, f, indent=2)

    print(f"\nDONE. Logged {len(metrics_log)} runtime entries to {metrics_path}.\n")
    return insights, metrics_log

# Load and build
wallets_df, transactions_df, edges_df = load_data("data")
G = build_graph(wallets_df, transactions_df, edges_df)
node_features, edge_features = process_features(G, confirm_path="n")
_, anomalies_df, model = detect_anomalies(G, node_features)

# Generate JSON of insights
generate_llm_insights_for_top_anomalies(
    G, model, anomalies_df,
    out_path="results/llm_insights.json",
    top_n=100,
    dry_run=False  # Use True for testing without GPT
)

