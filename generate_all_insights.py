import os
import json
import time
import math
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import torch

# Robust import for spearman
try:
    from scipy.stats import spearmanr
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# metrics (safe precision)
try:
    from sklearn.metrics import precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Local modules (your existing implementations)
from src.graphlime_explainer import explain_anomaly_multi
from src.llm_explainer import interpret_with_openai_multi
from src.graph_builder import load_data, build_graph
from src.feature_extraction import process_features
from src.anomaly_detection import detect_anomalies
from src.decision_tree_builder import train_and_save_decision_trees

# ---------- Config ----------
RESULTS_DIR = "results"
INSIGHTS_FILE = os.path.join(RESULTS_DIR, "llm_insights.json")
METRICS_FILE = os.path.join(RESULTS_DIR, "runtime_cost_log.json")
EXPLAINER_SCORES_FILE = os.path.join(RESULTS_DIR, "explainer_scores.json")
GNN_PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "gnn_predictions.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Keep same feature column order as your explainer
FEATURE_COLUMNS = [
    "degree", "in_degree", "out_degree",
    "num_txs_as_sender", "num_txs_as_receiver", "total_txs",
    "lifetime_in_blocks", "num_timesteps_appeared_in",
    "btc_transacted_total", "btc_transacted_mean", "btc_transacted_median",
    "btc_sent_total", "btc_sent_mean", "btc_sent_median",
    "btc_received_total", "btc_received_mean", "btc_received_median",
    "fees_total", "fees_mean", "fees_median",
    "blocks_btwn_txs_mean", "blocks_btwn_input_txs_mean", "blocks_btwn_output_txs_mean",
    "num_addr_transacted_multiple", "transacted_w_address_mean"
]

# ---------- Helpers ----------
def extract_json_from_text(text: str):
    """
    Try to extract JSON object embedded in a string that may include fenced code blocks
    or other wrapper text. Returns Python object or None.
    """
    if text is None:
        return None
    s = str(text)
    # Remove code fences like ```json ... ``` or ``` ... ```
    s_clean = s
    # Remove ```json and ``` fences
    s_clean = s_clean.replace("```json", "").replace("```", "").strip()
    # Some LLM outputs prefix with "JSON:" or "```json\n{...}\n```", handle common prefixes
    # Try to find the first '{' and last '}' and parse substring
    first = s_clean.find('{')
    last = s_clean.rfind('}')
    if first != -1 and last != -1 and last > first:
        candidate = s_clean[first:last+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # If bracket method fails, try direct json.loads
    try:
        return json.loads(s_clean)
    except Exception:
        return None

def spearman_fallback(a, b):
    """Fallback spearman if scipy missing"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size:
        return None
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None
    def rankdata(x):
        sorter = np.argsort(x)
        ranks = np.empty_like(sorter, dtype=float)
        ranks[sorter] = np.arange(len(x), dtype=float)
        # tie handling: average ranks
        unique_vals, inv_idx, counts = np.unique(x, return_inverse=True, return_counts=True)
        if np.any(counts > 1):
            for val_idx, cnt in enumerate(counts):
                if cnt > 1:
                    idxs = np.where(inv_idx == val_idx)[0]
                    ranks[idxs] = ranks[idxs].mean()
        return ranks + 1.0
    ra, rb = rankdata(a), rankdata(b)
    ra_mean, rb_mean = ra.mean(), rb.mean()
    num = np.sum((ra - ra_mean) * (rb - rb_mean))
    den = math.sqrt(np.sum((ra - ra_mean)**2) * np.sum((rb - rb_mean)**2))
    if den == 0:
        return None
    return float(num / den)

def compute_spearman(a, b):
    if a is None or b is None:
        return None
    if SCIPY_AVAILABLE:
        corr, p = spearmanr(a, b)
        if math.isnan(corr):
            return None
        return float(corr)
    else:
        return spearman_fallback(a, b)

# ---------- Per-node worker (thread-safe) ----------
def process_single_node(
    node_id,
    row_dict,
    G,
    model,
    gnn_time,
    dry_run,
    num_samples,
    llm_temperature,
    llm_model,
    top_features_count,
):
    """
    Returns: insight_entry, explainer_scores_entry, gnn_pred_entry (dict node->pred), metrics_entry
    """
    start_time = time.time()
    # Defaults
    explainer_scores_entry = {}
    insight_entry = {
        "top_features": [],
        "explainer_scores": {},
        "llm_output_structured": None,
        "faithfulness": {},
        "metadata": {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "node_data": row_dict
    }
    gnn_pred_entry = {str(node_id): None}
    metrics_entry = {
        "node_id": str(node_id),
        "gnn_time_s": float(gnn_time),
        "explainer_time_s": 0.0,
        "llm_latency_s": 0.0,
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
        "llm_cost_usd": 0.0,
        "total_time_s": 0.0,
        "agreement_rate": 0.0,
        "fraud_type_agreement": 0.0,
        "avg_confidence": 0.0
    }

    # 1) Explainers
    try:
        t0 = time.time()
        expl_results = explain_anomaly_multi(G, model, node_id, explainers=["graphlime", "gnnexplainer"])
        metrics_entry["explainer_time_s"] = time.time() - t0
    except Exception as e:
        expl_results = {}
        metrics_entry["explainer_time_s"] = 0.0
        print(f"[WARN] explain_anomaly_multi failed for {node_id}: {e}")

    # Choose top features based on GraphLIME preferentially
    top_feature_names = []
    try:
        if "graphlime" in expl_results:
            arr = np.asarray(expl_results["graphlime"], dtype=float)
            idxs = np.argsort(arr)[::-1][:top_features_count]
            top_feature_names = [FEATURE_COLUMNS[i] for i in idxs]
        else:
            # fallback union
            chosen = []
            for expl, arr in expl_results.items():
                arr = np.asarray(arr, dtype=float)
                idxs = np.argsort(arr)[::-1][:top_features_count]
                for i in idxs:
                    chosen.append(FEATURE_COLUMNS[i])
                if len(chosen) >= top_features_count:
                    break
            top_feature_names = list(dict.fromkeys(chosen))[:top_features_count]
    except Exception as e:
        print(f"[WARN] selecting top features failed for {node_id}: {e}")
        top_feature_names = []

    # Collect explainer scores for those features
    for feat in top_feature_names:
        i = FEATURE_COLUMNS.index(feat)
        scores = {}
        for expl, vals in expl_results.items():
            try:
                scores[expl] = float(vals[i])
            except Exception:
                scores[expl] = None
        explainer_scores_entry[feat] = scores

    insight_entry["top_features"] = [(feat, explainer_scores_entry[feat].get("graphlime", 0.0)) for feat in top_feature_names]
    insight_entry["explainer_scores"] = explainer_scores_entry

    # Faithfulness (Spearman between graphlime and others)
    faith = {}
    if "graphlime" in expl_results:
        for expl, vals in expl_results.items():
            if expl == "graphlime":
                continue
            corr = compute_spearman(expl_results["graphlime"], vals)
            faith[f"graphlime_vs_{expl}"] = corr
    insight_entry["faithfulness"] = faith

    # 2) GNN prediction (best-effort)
    try:
        with torch.no_grad():
            if hasattr(model, "x") and hasattr(model, "edge_index"):
                logits = model(model.x, model.edge_index)
                node_idx = list(G.nodes()).index(node_id)
                node_logits = logits[node_idx]
                if node_logits.dim() == 0 or (node_logits.dim() == 1 and node_logits.numel() == 1):
                    pred = int((node_logits > 0).item())
                else:
                    pred = int(torch.argmax(node_logits).item())
            else:
                featvec = [row_dict.get(c, 0.0) for c in FEATURE_COLUMNS]
                x_in = torch.tensor([featvec], dtype=torch.float)
                try:
                    logits = model(x_in)
                    arr = logits.squeeze(0)
                    if arr.numel() == 1:
                        pred = int((arr > 0).item())
                    else:
                        pred = int(torch.argmax(arr).item())
                except Exception:
                    pred = None
    except Exception as e:
        print(f"[WARN] GNN prediction failed for {node_id}: {e}")
        pred = None
    gnn_pred_entry = {str(node_id): pred}
    insight_entry.setdefault("metadata", {})["gnn_prediction"] = pred

    # 3) LLM multi-sample (structured) — robust parsing
    try:
        t_llm0 = time.time()
        if dry_run:
            llm_result = {"samples": [], "consensus": {"node_id": str(node_id), "is_fraud": None},
                          "avg_latency": 0.0, "total_input_tokens": 0, "total_output_tokens": 0, "total_cost_usd": 0.0}
            llm_latency = 0.0
            in_tok = out_tok = cost = 0.0
        else:
            top_features_for_llm = insight_entry["top_features"]
            llm_result = interpret_with_openai_multi(
                node_id=str(node_id),
                top_features=top_features_for_llm,
                node_data=row_dict,
                model=llm_model,
                num_samples=num_samples,
                temperature=llm_temperature,
            )
            # parse/clean samples inside llm_result to ensure structured fields are populated
            parsed_samples = []
            for s in llm_result.get("samples", []):
                # Some samples may include 'explanation' containing a JSON code fence. try to parse it.
                explanation_text = s.get("explanation") or s.get("text") or ""
                parsed = extract_json_from_text(explanation_text)
                if isinstance(parsed, dict):
                    # copy recognized fields if present
                    s_parsed = {
                        "raw_explanation": explanation_text,
                        "parsed": parsed,
                        "is_fraud": parsed.get("is_fraud", s.get("is_fraud")),
                        "fraud_type": parsed.get("fraud_type", s.get("fraud_type")),
                        "confidence": parsed.get("confidence", s.get("confidence", 0.0)),
                        "evidence": parsed.get("evidence", s.get("evidence", {}))
                    }
                else:
                    s_parsed = {
                        "raw_explanation": explanation_text,
                        "parsed": None,
                        "is_fraud": s.get("is_fraud"),
                        "fraud_type": s.get("fraud_type"),
                        "confidence": s.get("confidence", 0.0),
                        "evidence": s.get("evidence", {})
                    }
                parsed_samples.append(s_parsed)
            llm_result["samples_parsed"] = parsed_samples

            llm_latency = llm_result.get("avg_latency", 0.0)
            in_tok = llm_result.get("total_input_tokens", 0)
            out_tok = llm_result.get("total_output_tokens", 0)
            cost = llm_result.get("total_cost_usd", 0.0)
        metrics_entry["llm_latency_s"] = float(llm_latency)
        metrics_entry["llm_input_tokens"] = int(in_tok)
        metrics_entry["llm_output_tokens"] = int(out_tok)
        metrics_entry["llm_cost_usd"] = float(cost)
    except Exception as e:
        print(f"[WARN] LLM call failed for {node_id}: {e}")
        llm_result = {"samples": [], "consensus": {"node_id": str(node_id), "is_fraud": None}}
        metrics_entry["llm_latency_s"] = 0.0
        metrics_entry["llm_input_tokens"] = 0
        metrics_entry["llm_output_tokens"] = 0
        metrics_entry["llm_cost_usd"] = 0.0

    # sanitize consensus: ensure booleans/numbers not None where possible
    consensus = llm_result.get("consensus", {})
    # If consensus is not giving a boolean for is_fraud, try majority from parsed samples
    if consensus.get("is_fraud") is None:
        parsed = llm_result.get("samples_parsed", [])
        votes = [1 if (s.get("is_fraud") in [True, 1, "true", "True"]) else 0 for s in parsed if s.get("is_fraud") is not None]
        if votes:
            is_fraud_vote = 1 if sum(votes) / len(votes) >= 0.5 else 0
            consensus["is_fraud"] = bool(is_fraud_vote)
        else:
            consensus["is_fraud"] = None
    insight_entry["llm_output_structured"] = llm_result
    metrics_entry["agreement_rate"] = float(llm_result.get("agreement_rate", 0.0))
    metrics_entry["avg_confidence"] = float(llm_result.get("avg_confidence", 0.0))
    metrics_entry["fraud_type_agreement"] = float(llm_result.get("fraud_type_agreement") or 0.0)

    metrics_entry["total_time_s"] = float(time.time() - start_time)

    return insight_entry, explainer_scores_entry, gnn_pred_entry, metrics_entry

# ---------- Orchestrator ----------
def generate_llm_insights_for_top_anomalies(
    G,
    model,
    anomalies_df,
    out_path=INSIGHTS_FILE,
    metrics_path=METRICS_FILE,
    gnn_time=0,
    sample_count=50,
    dry_run=False,
    num_samples=5,
    llm_temperature=0.2,
    llm_model="gpt-4o-mini",
    top_features_count=5,
    max_workers=8,
    batch_save_every=20,
):
    """
    Threaded orchestrator: selects top sample_count anomalies, runs them in threads,
    and saves JSONs periodically. Resume-safe.
    """
    # resume loads
    if os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                raw = f.read().strip()
                insights = json.loads(raw) if raw else {}
        except Exception:
            print("[WARN] Corrupted insights file — starting fresh.")
            insights = {}
    else:
        insights = {}

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                raw = f.read().strip()
                metrics_log = json.loads(raw) if raw else []
        except Exception:
            print("[WARN] Corrupted metrics file — starting fresh.")
            metrics_log = []
    else:
        metrics_log = []

    if os.path.exists(EXPLAINER_SCORES_FILE) and os.path.getsize(EXPLAINER_SCORES_FILE) > 0:
        with open(EXPLAINER_SCORES_FILE, "r") as f:
            explainer_scores = json.load(f)
    else:
        explainer_scores = {}

    if os.path.exists(GNN_PREDICTIONS_FILE):
        with open(GNN_PREDICTIONS_FILE, "r") as f:
            gnn_predictions = json.load(f)
    else:
        gnn_predictions = {}

    # ensure anomalies_df has anomaly_score column
    if "anomaly_score" not in anomalies_df.columns:
        raise ValueError("anomalies_df must include 'anomaly_score' column for ranking.")

    df_sorted = anomalies_df.sort_values(by="anomaly_score", ascending=False)
    sample_df = df_sorted.head(sample_count)
    print(f"[INFO] Selected top {len(sample_df)} anomalies for evaluation (sample_count={sample_count}).")

    # Build work list skipping processed
    work_items = []
    for _, row in sample_df.iterrows():
        node_id = row.name
        if str(node_id) in insights:
            continue
        work_items.append((node_id, row.to_dict()))
    total_to_process = len(work_items)
    print(f"[INFO] {total_to_process} nodes to process after skipping already-done.")

    if total_to_process == 0:
        print("[INFO] Nothing to process.")
        return insights, metrics_log, explainer_scores, gnn_predictions

    workers = min(max_workers, os.cpu_count() or 4)
    print(f"[INFO] Using {workers} worker threads.")

    # thread worker wrapper
    worker_fn = partial(
        process_single_node,
        G=G,
        model=model,
        gnn_time=gnn_time,
        dry_run=dry_run,
        num_samples=num_samples,
        llm_temperature=llm_temperature,
        llm_model=llm_model,
        top_features_count=top_features_count
    )

    processed = 0
    save_counter = 0
    start_all = time.time()

    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(worker_fn, node_id, row_dict): (node_id, row_dict) for (node_id, row_dict) in work_items}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Nodes"):
            node_id, _ = futures[fut]
            try:
                insight_entry, expl_scores_entry, gnn_pred_entry, metrics_entry = fut.result()
            except Exception as e:
                print(f"[WARN] worker exception for node {node_id}: {e}")
                continue

            nid = str(node_id)
            # write into main dicts
            insights[nid] = insight_entry
            explainer_scores[nid] = expl_scores_entry
            gnn_predictions.update(gnn_pred_entry)
            metrics_log.append(metrics_entry)

            processed += 1
            save_counter += 1

            if save_counter >= batch_save_every:
                # periodic flush
                with open(out_path, "w") as f:
                    json.dump(insights, f, indent=2)
                with open(EXPLAINER_SCORES_FILE, "w") as f:
                    json.dump(explainer_scores, f, indent=2)
                with open(GNN_PREDICTIONS_FILE, "w") as f:
                    json.dump(gnn_predictions, f, indent=2)
                with open(metrics_path, "w") as f:
                    json.dump(metrics_log, f, indent=2)
                save_counter = 0

    # final save
    with open(out_path, "w") as f:
        json.dump(insights, f, indent=2)
    with open(EXPLAINER_SCORES_FILE, "w") as f:
        json.dump(explainer_scores, f, indent=2)
    with open(GNN_PREDICTIONS_FILE, "w") as f:
        json.dump(gnn_predictions, f, indent=2)
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)

    elapsed = time.time() - start_all
    print(f"\nDONE — processed {processed} nodes in {elapsed/60:.2f} minutes. Saved to {RESULTS_DIR}.")

    # After processing, train decision trees as before
    print("Training decision tree on LLM consensus labels...")
    stats = train_and_save_decision_trees()
    print("Decision tree training finished. Stats saved to results/decision_tree_stats.json")

    return insights, metrics_log, explainer_scores, gnn_predictions

# ---------- Main ----------
if __name__ == "__main__":
    wallets_df, transactions_df, edges_df = load_data("data")
    G = build_graph(wallets_df, transactions_df, edges_df)
    node_features, edge_features = process_features(G, confirm_path="n")

    gnn_start = time.time()
    _, _, anomalies_df, model = detect_anomalies(G, node_features)
    gnn_time = time.time() - gnn_start

    # Run the orchestrator for 220 nodes (top anomalies)
    insights, metrics, explainer_scores, gnn_preds = generate_llm_insights_for_top_anomalies(
        G=G,
        model=model,
        anomalies_df=anomalies_df,
        out_path=INSIGHTS_FILE,
        metrics_path=METRICS_FILE,
        gnn_time=gnn_time,
        sample_count=50,
        dry_run=False,
        num_samples=5,
        llm_temperature=0.2,
        llm_model="gpt-4o-mini",
        top_features_count=5,
        max_workers=8,
        batch_save_every=25
    )
    print("All done :)")