# dashboard/backend/app.py
from flask import Flask, jsonify, abort
from flask_cors import CORS
from pathlib import Path
import json
import logging

app = Flask(__name__)
CORS(app)  # allow cross-origin requests (useful when frontend served separately)

# Resolve the dashboard_data directory robustly (works on Windows/Unix)
BASE_DIR = Path(__file__).resolve().parent.parent  # .../dashboard
DATA_DIR = BASE_DIR / "dashboard_data"

# Fallback: if not found, try repo-root/dashboard_data (in case structure differs)
if not DATA_DIR.exists():
    alt = BASE_DIR.parent / "dashboard_data"
    if alt.exists():
        DATA_DIR = alt

# If still not found, raise a helpful error
if not DATA_DIR.exists():
    raise RuntimeError(
        f"dashboard_data directory not found. Checked:\n - {str(BASE_DIR / 'dashboard_data')}\n - {str(BASE_DIR.parent / 'dashboard_data')}\n"
    )

app.logger.setLevel(logging.INFO)
app.logger.info(f"Using DATA_DIR = {DATA_DIR}")

def load_json_file(fname):
    p = DATA_DIR / fname
    if not p.exists():
        app.logger.error(f"File not found: {p}")
        abort(404, description=f"File not found: {fname}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        app.logger.error(f"JSON decode error for {p}: {e}")
        abort(500, description=f"Invalid JSON in {fname}")

@app.route("/subgraph", methods=["GET"])
def get_subgraph():
    """
    Returns an object with { nodes: [...], links: [...] } expected by the front-end D3 renderer.
    Uses dashboard_data/graph.json (edge list) and node_data.json for node attributes.
    """
    edge_list = load_json_file("graph.json")            # expected: list of {"source":"x","target":"y"}
    raw_node_data = {}
    try:
        raw_node_data = load_json_file("node_data.json")  # expected: mapping node_id -> metadata
    except:
        # Not fatal: node_data may be missing; we still serve nodes/links
        app.logger.warning("node_data.json missing or unreadable; proceeding without node metadata")

    # Normalize edges -> links with source/target properties (strings)
    links = []
    node_ids = set()
    for e in edge_list:
        src = str(e.get("source"))
        tgt = str(e.get("target"))
        links.append({"source": src, "target": tgt})
        node_ids.add(src)
        node_ids.add(tgt)

    # Build node objects (include anomaly / label if available)
    nodes = []
    for nid in sorted(node_ids):
        meta = raw_node_data.get(nid, {})
        node_obj = {
            "id": nid,
            # try common fields from your node_data.json; fall back to sensible defaults
            "label": meta.get("feature_values", {}).get("label", meta.get("label", nid)),
            "anomaly": bool(meta.get("is_anomalous", False)),
            "anomaly_score": meta.get("anomaly_score", None),
            "raw": meta  # include full metadata if frontend needs it
        }
        nodes.append(node_obj)

    return jsonify({"nodes": nodes, "links": links})

@app.route("/node-data", methods=["GET"])
def get_node_data():
    return jsonify(load_json_file("node_data.json"))

@app.route("/llm-insights", methods=["GET"])
def get_llm_insights():
    return jsonify(load_json_file("llm_insights.json"))

if __name__ == "__main__":
    # Run on localhost:5000
    app.run(debug=True, port=5000)
