import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.feature_extraction import process_features

FEATURES_DIR = "features"
ANOMALY_OUTPUT_FILE = os.path.join(FEATURES_DIR, "anomaly_scores.csv")

def detect_anomalies(G, node_features, num_anomalies=10):
    """Runs Isolation Forest on node features to detect anomalies and returns the top N anomalies."""
    print("Loading or extracting features for anomaly detection...")
    
    if node_features is None:
        raise ValueError("Node features could not be loaded or extracted.")

    # Selecting numerical columns for anomaly detection
    feature_columns = ["degree", "in_degree", "out_degree"]  # Adjust based on actual features
    df_filtered = node_features[feature_columns].fillna(0)  # Replace NaNs with 0
    
    print("Training Isolation Forest...")
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(df_filtered)  # Make sure the model is fitted first

    # Now we can use decision_function() since the model is fitted
    node_features["anomaly_score"] = model.decision_function(df_filtered)  # Higher means more normal
    node_features["anomaly_label"] = model.predict(df_filtered)  # -1 for anomalies

    # Sort anomalies by lowest anomaly score (most anomalous)
    anomalies = node_features[node_features["anomaly_label"] == -1].sort_values(by="anomaly_score")

    # Save results
    os.makedirs(FEATURES_DIR, exist_ok=True)
    node_features.to_csv(ANOMALY_OUTPUT_FILE, index=False)
    print(f"Anomaly detection complete. Results saved to {ANOMALY_OUTPUT_FILE}")

    # Print top N anomalies
    top_anomalies = anomalies.head(num_anomalies)
    print("\nTop 10 Anomalous Nodes:")
    for i, (index, row) in enumerate(top_anomalies.iterrows()):
        print(f"{i+1}. Node: {row['node']}, Score: {row['anomaly_score']:.4f}")

    return node_features, top_anomalies, model
