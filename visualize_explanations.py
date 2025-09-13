import json
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go

INPUT_JSON = "results/llm_insights.json"
OUTPUT_DIR = "results/summary_visuals"

FRAUD_TYPES = [
    "Ponzi schemes", "phishing attacks", "pump-and-dump schemes",
    "ransomware", "SIM swapping", "mining malware",
    "giveaway scams", "impersonation scams", "securities fraud",
    "money laundering"
]

def extract_fraud_type(reasoning):
    """Match fraud type keywords from LLM reasoning."""
    for ftype in FRAUD_TYPES:
        if re.search(ftype.lower().replace("-", "[- ]"), reasoning.lower()):
            return ftype
    return "Unclassified"

def main():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build co-occurrence counts
    records = []
    for node_id, entry in data.items():
        features = [f for f, _ in entry.get("top_features", [])]
        fraud_type = extract_fraud_type(entry.get("llm_reasoning", ""))
        for feat in features:
            records.append((feat, fraud_type))

    df = pd.DataFrame(records, columns=["Feature", "FraudType"])
    counts = df.value_counts().reset_index(name="Count")

    # --- Sankey diagram ---
    features = counts["Feature"].unique().tolist()
    fraud_types = counts["FraudType"].unique().tolist()

    labels = features + fraud_types
    source = [features.index(f) for f in counts["Feature"]]
    target = [len(features) + fraud_types.index(ft) for ft in counts["FraudType"]]
    value = counts["Count"].tolist()

    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=20, thickness=20),
        link=dict(source=source, target=target, value=value)
    )])

    # Save interactive HTML
    fig_sankey.write_html(os.path.join(OUTPUT_DIR, "feature_fraud_sankey.html"))

    # --- Heatmap ---
    pivot = counts.pivot_table(
        index="FraudType", columns="Feature", values="Count", fill_value=0
    )

    # Convert all values to integers to avoid fmt errors
    pivot = pivot.astype(int)

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, cmap="Blues", cbar=True, fmt="d")
    plt.title("Feature–Fraud Type Co-occurrence")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_fraud_heatmap.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()