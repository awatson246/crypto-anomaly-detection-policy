import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

FEATURES_DIR = "features"
ANOMALY_OUTPUT_FILE = os.path.join(FEATURES_DIR, "anomalies.csv")

def load_features():
    """Loads precomputed node features for anomaly detection."""
    node_features_path = os.path.join(FEATURES_DIR, "node_features.csv")
    if not os.path.exists(node_features_path):
        raise FileNotFoundError("Feature file not found. Run feature extraction first.")
    
    return pd.read_csv(node_features_path)

def detect_anomalies():
    """Runs Isolation Forest on node features to detect anomalies."""
    print("Loading features for anomaly detection...")
    df = load_features()

    # Selecting numerical columns for anomaly detection
    feature_columns = ["degree", "in_degree", "out_degree"]  # Adjust based on data availability
    df_filtered = df[feature_columns].fillna(0)  # Replace NaNs with 0
    
    print("Training Isolation Forest...")
    model = IsolationForest(contamination=0.01, random_state=42)
    df["anomaly_score"] = model.fit_predict(df_filtered)

    # Save results
    os.makedirs(FEATURES_DIR, exist_ok=True)
    df.to_csv(ANOMALY_OUTPUT_FILE, index=False)
    print(f"Anomaly detection complete. Results saved to {ANOMALY_OUTPUT_FILE}")

    # Print anomaly summary
    anomalies = df[df["anomaly_score"] == -1]
    print(f"Detected {len(anomalies)} anomalies.")
    
    return df, anomalies

