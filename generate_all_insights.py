import os
import json
import time
from tqdm import tqdm

from src.graphlime_explainer import explain_anomaly
from src.llm_explainer import interpret_with_openai
from src.graph_builder import load_data, build_graph
from src.feature_extraction import process_features
from src.anomaly_detection import detect_anomalies


def generate_llm_insights_for_top_anomalies(
    G,
    model,
    anomalies_df,
    out_path="results/llm_insights.json",
    metrics_path="results/runtime_cost_log.json",
    gnn_time=0,
    top_n=100,
    dry_run=False,
):
    """
    Generates GraphLIME + LLM insights with cost/latency logging.

    Stores:
      - Structured LLM JSON from interpret_with_openai
      - GraphLIME feature importance
      - Raw insight_text (from GraphLIME)
      - Token + cost + per-node latency breakdown

    Output files:
      - out_path:     JSON dictionary keyed by node ID
      - metrics_path: list of per-node timing/cost dictionaries
    """

    # ------------------------------------------------------
    # Load cached LLM insights (resume without overwriting)
    # ------------------------------------------------------
    if os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                raw = f.read().strip()
                insights = json.loads(raw) if raw else {}
        except:
            print("[WARN] insights file corrupted → starting fresh.")
            insights = {}
    else:
        insights = {}

    # ------------------------------------------------------
    # Load metrics log (for cumulative latency/cost tracking)
    # ------------------------------------------------------
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                raw = f.read().strip()
                metrics_log = json.loads(raw) if raw else []
        except:
            print("[WARN] metrics file corrupted → starting fresh.")
            metrics_log = []
    else:
        metrics_log = []

    os.makedirs("results", exist_ok=True)

    print(f"\n=== Generating insights for top {top_n} anomalies ===\n")

    # Iterate through anomalies
    for _, (node_id, row) in enumerate(anomalies_df.head(top_n).iterrows()):
        node_id_str = str(node_id)

        # Skip if we already have this one
        if node_id_str in insights:
            continue

        # --------------------------
        # GraphLIME attribution time
        # --------------------------
        gl_start = time.time()
        try:
            _, top_features, node_data_dict, insight_txt = explain_anomaly(G, model, node_id)
        except Exception as e:
            print(f"[WARN] GraphLIME failed on node {node_id}: {e}")
            top_features = []
            node_data_dict = {}
            insight_txt = ""
        gl_time = time.time() - gl_start

        # --------------------------
        # LLM structured explanation
        # --------------------------
        if dry_run:
            llm_struct = {
                "explanation": "[DRY RUN]",
                "is_fraud": None,
                "fraud_type": None,
                "confidence": 0.0,
            }
            llm_latency = 0
            in_tok = out_tok = 0
            cost = 0.0
        else:
            llm_struct, llm_latency, in_tok, out_tok, cost = interpret_with_openai(
                node_id=node_id,
                top_features=top_features,
                node_data=node_data_dict,
            )

        # --------------------------
        # Save insight to JSON
        # --------------------------
        insights[node_id_str] = {
            "top_features": [(name, float(val)) for name, val in top_features],
            "graphlime_raw_insight": insight_txt,
            "llm_output_structured": llm_struct,
        }

        with open(out_path, "w") as f:
            json.dump(insights, f, indent=2)

        # --------------------------
        # Save metrics
        # --------------------------
        metrics_log.append({
            "node_id": node_id_str,
            "gnn_time_s": float(gnn_time),
            "graphlime_time_s": float(gl_time),
            "llm_latency_s": float(llm_latency),
            "llm_input_tokens": int(in_tok),
            "llm_output_tokens": int(out_tok),
            "llm_cost_usd": float(cost),
            "total_time_s": float(gnn_time + gl_time + llm_latency),
        })

        with open(metrics_path, "w") as f:
            json.dump(metrics_log, f, indent=2)

    print(f"\nDONE — {len(insights)} insights stored.")
    print(f"Metrics written to {metrics_path}\n")

    return insights, metrics_log


# ----------------------------------------------------------
# Pipeline to call the generator (unchanged)
# ----------------------------------------------------------
wallets_df, transactions_df, edges_df = load_data("data")
G = build_graph(wallets_df, transactions_df, edges_df)
node_features, edge_features = process_features(G, confirm_path="n")

gnn_start = time.time()
_, anomalies_df, model = detect_anomalies(G, node_features)
gnn_time = time.time() - gnn_start

# Generate JSON of insights
generate_llm_insights_for_top_anomalies(
    G,
    model,
    anomalies_df,
    out_path="results/llm_insights.json",
    metrics_path="results/runtime_cost_log.json",
    gnn_time=gnn_time,
    top_n=100,
    dry_run=False,
)
