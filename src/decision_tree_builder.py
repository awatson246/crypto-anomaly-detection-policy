import os
import re
import json
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

RESULTS_DIR = "results"
INSIGHTS_FILE = os.path.join(RESULTS_DIR, "llm_insights.json")


# If you have FEATURE_COLUMNS defined in src.graphlime_explainer, import it.
try:
    from src.graphlime_explainer import FEATURE_COLUMNS
except Exception:
    # fallback: minimal sensible list (should be replaced by your canonical FEATURE_COLUMNS)
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


# Numeric stats we will always try to extract from the GraphLIME insight text
CORE_STATS = [
    "total_txs",
    "btc_received_total",
    "btc_sent_total",
    "num_txs_as_sender",
    "num_txs_as_receiver",
    "btc_transacted_total",
    "fees_total",
    "degree"
]

NUMERIC_RE = re.compile(r"[- ]+([a-zA-Z0-9_\- ]+):\s*([0-9.+eE\-]+)")

def load_insights(path=INSIGHTS_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No insights file found at {path}")
    with open(path, "r") as f:
        return json.load(f)


def parse_numeric_stats_from_insight(insight_text):
    """
    Parse the 'Additional node statistics' block produced by format_insight().
    Returns a dict of numeric fields present.
    """
    stats = {}
    if not insight_text:
        return stats

    # Find lines like " - key: value"
    for m in NUMERIC_RE.finditer(insight_text):
        key = m.group(1).strip().replace(" ", "_")
        val_str = m.group(2)
        try:
            val = float(val_str)
        except:
            continue
        stats[key] = val
    return stats


def build_feature_vector(entry):
    """
    Given a single insights entry, return a fixed-length numeric vector and the feature names order.
    Vector layout:
      - CORE_STATS in given order (zero if missing)
      - FEATURE_COLUMNS importance values (float) in given order (0 if absent)
    """
    insight_text = entry.get("graphlime_raw_insight", "") or ""
    parsed_stats = parse_numeric_stats_from_insight(insight_text)

    # Core stats vector
    core_vals = [float(parsed_stats.get(k, 0.0)) for k in CORE_STATS]

    # Map top_features (list of [name, importance]) into a dict for easy lookup
    top_feats = {}
    for item in entry.get("top_features", []):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            name, val = item
            try:
                top_feats[str(name)] = float(val)
            except:
                top_feats[str(name)] = 0.0

    # For each FEATURE_COLUMNS entry, use the GraphLIME importance if present, else 0
    importance_vals = [float(top_feats.get(feat, 0.0)) for feat in FEATURE_COLUMNS]

    feature_vector = core_vals + importance_vals
    feature_names = CORE_STATS + FEATURE_COLUMNS
    return feature_vector, feature_names


def extract_dataset(insights):
    """
    Convert insights json into X (n x d), y_binary (n,), y_types (n,) and feature_names.
    Skips entries where consensus is missing.
    """
    X = []
    y_binary = []
    y_types = []

    for node_id, entry in insights.items():
        llm_struct = entry.get("llm_output_structured", {})
        consensus = llm_struct.get("consensus") if isinstance(llm_struct, dict) else None
        # if consensus not present, try older structure where llm_output_structured itself is consensus
        if consensus is None and isinstance(llm_struct, dict) and "is_fraud" in llm_struct:
            consensus = llm_struct

        if not consensus:
            continue

        is_fraud = consensus.get("is_fraud")
        fraud_type = consensus.get("fraud_type") or None

        # skip entries with missing is_fraud
        if is_fraud is None:
            continue

        # Build feature vector
        vec, feature_names = build_feature_vector(entry)

        # Defensive check: ensure vector is flat numeric list
        if not isinstance(vec, (list, tuple)) or any(not isinstance(v, (int, float)) for v in vec):
            continue

        X.append(vec)
        y_binary.append(1 if is_fraud else 0)
        y_types.append(fraud_type if fraud_type is not None else "normal")

    if len(X) == 0:
        raise ValueError("No usable examples found in insights file (X is empty).")

    X_arr = np.array(X, dtype=float)
    y_binary_arr = np.array(y_binary)
    y_types_arr = np.array(y_types)

    return X_arr, y_binary_arr, y_types_arr, feature_names


def train_and_save_decision_trees():
    insights = load_insights()

    X, y_binary, y_types, feature_names = extract_dataset(insights)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Encode multiclass fraud_type labels
    le = LabelEncoder()
    y_types_enc = le.fit_transform(y_types)

    # Train/test split for binary and multiclass (same split for fairness)
    X_train, X_test, yb_train, yb_test, yt_train, yt_test = train_test_split(
        X, y_binary, y_types_enc, test_size=0.2, random_state=42
    )

    # Decision Tree: fraud vs not fraud
    dt_binary = DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, class_weight="balanced", random_state=42)
    dt_binary.fit(X_train, yb_train)
    yb_pred = dt_binary.predict(X_test)
    rules_binary = export_text(dt_binary, feature_names=feature_names)
    with open(os.path.join(RESULTS_DIR, "decision_tree_rules_binary.txt"), "w") as f:
        f.write(rules_binary)
    joblib.dump(dt_binary, os.path.join(RESULTS_DIR, "decision_tree_binary.pkl"))

    # Decision Tree: fraud-type multiclass
    dt_types = DecisionTreeClassifier(max_depth=5, min_samples_leaf=8, class_weight="balanced", random_state=42)
    dt_types.fit(X_train, yt_train)
    yt_pred = dt_types.predict(X_test)
    rules_types = export_text(dt_types, feature_names=feature_names)
    with open(os.path.join(RESULTS_DIR, "decision_tree_rules_multiclass.txt"), "w") as f:
        f.write(rules_types)
    joblib.dump(dt_types, os.path.join(RESULTS_DIR, "decision_tree_multiclass.pkl"))

    # Stats & reports
    stats = {
        "n_examples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "binary_accuracy": float(accuracy_score(yb_test, yb_pred)),
        "binary_report": classification_report(yb_test, yb_pred, output_dict=True, zero_division=0),
        "multiclass_accuracy": float(accuracy_score(yt_test, yt_pred)),
        "multiclass_report": classification_report(yt_test, yt_pred, output_dict=True, zero_division=0),
        "fraud_type_classes": le.classes_.tolist()
    }

    with open(os.path.join(RESULTS_DIR, "decision_tree_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== Decision Tree Training Complete ===")
    print(f"Examples: {stats['n_examples']}, Features: {stats['n_features']}")
    print("Saved files to results/: decision_tree_rules_*.txt, decision_tree_*.pkl, decision_tree_stats.json")

    return stats


if __name__ == "__main__":
    train_and_save_decision_trees()
