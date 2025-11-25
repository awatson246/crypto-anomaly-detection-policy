import os
import time
import json
import openai
from dotenv import load_dotenv
from statistics import mode
from collections import Counter

load_dotenv()
client = openai.OpenAI()

# -----------------------------
#  Model pricing (per 1M tokens)
# -----------------------------
MODEL_PRICES = {
    "gpt-4o-mini": {"input": 0.15/1_000_000, "output": 0.60/1_000_000},
    "gpt-4o": {"input": 0.80/1_000_000, "output": 3.20/1_000_000},
    "gpt-3.5-turbo": {"input": 0.50/1_000_000, "output": 1.50/1_000_000},
}

def estimate_cost(model, in_tokens, out_tokens):
    if model not in MODEL_PRICES:
        return None
    rates = MODEL_PRICES[model]
    return in_tokens * rates["input"] + out_tokens * rates["output"]


# -------------------------------------------------------------
#   Helper: Safe retry wrapper (strong exponential backoff)
# -------------------------------------------------------------
def retry_api_call(fn, max_retries=12, tag="LLM"):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            wait = (2 ** attempt) + (0.2 * attempt)

            if "rate" in msg or "limit" in msg or "overloaded" in msg:
                print(f"[{tag}] Rate limit, retrying in {wait:.2f}s...")
                time.sleep(wait)
                continue

            if "timeout" in msg or "503" in msg or "connection" in msg:
                print(f"[{tag}] Network hiccup, retrying in {wait:.2f}s...")
                time.sleep(wait)
                continue

            print(f"[{tag}] Fatal error: {e}")
            return None

    print(f"[{tag}] Failed after maximum retries.")
    return None


# -------------------------------------------------------------
#      NEW: MULTI-SAMPLE LLM INTERPRETATION FUNCTION
# -------------------------------------------------------------
def interpret_with_openai_multi(
    node_id,
    top_features,
    node_data,
    model="gpt-4o-mini",
    num_samples=5,
    temperature=0.2,
):
    """
    Returns:
      {
        "samples": [...all raw LLM JSON outputs...],
        "consensus": {...single consensus JSON...},
        "agreement_rate": float,
        "fraud_type_agreement": float,
        "avg_confidence": float,
        "avg_latency": float,
        "total_cost_usd": float,
        "total_input_tokens": int,
        "total_output_tokens": int
      }
    """

    fraud_types = (
        "Ponzi schemes, phishing attacks, pump-and-dump schemes, ransomware, "
        "SIM swapping, mining malware, giveaway scams, impersonation scams, "
        "securities fraud, money laundering"
    )

    formatted_weights = "\n".join(
        [f"- {name}: {value:.3e}" for name, value in top_features]
    )

    formatted_data = "\n".join(
        [f"- {k}: {v}" for k, v in node_data.items()]
    )

    base_prompt = f"""
You are a cryptocurrency forensics analyst.

You will receive:
1. GraphLIME feature importances
2. Raw node statistics
3. Fraud-type ontology

Return STRICT JSON:

{{
  "explanation": "... (2-5 sentences)",
  "is_fraud": true/false,
  "fraud_type": "... or null",
  "confidence": float,
  "evidence": {{
      "features": [list of most relevant feature names],
      "behaviors": [list of notable behavioral patterns]
  }}
}}

Rules:
- fraud_type MUST be from: {fraud_types}
- If not fraudulent → fraud_type = null.
- Be concise and data-grounded.
- Return ONLY JSON.

### Feature Importances
{formatted_weights}

### Node Statistics
{formatted_data}
"""

    samples = []
    all_is_fraud = []
    all_fraud_types = []
    all_confidences = []

    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_usd = 0.0
    total_latency = 0.0

    # --------------------------------------------
    # Run N LLM calls
    # --------------------------------------------
    for i in range(num_samples):
        def call_api():
            return client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": base_prompt
                }],
                temperature=temperature
            )

        start = time.time()
        response = retry_api_call(
            call_api, tag=f"LLM:{model}:sample{i}"
        )
        latency = time.time() - start
        total_latency += latency

        if response is None:
            samples.append({
                "explanation": "LLM failed",
                "is_fraud": None,
                "fraud_type": None,
                "confidence": 0.0,
                "evidence": {}
            })
            continue

        msg = response.choices[0].message.content.strip()

        # Parse JSON robustly
        try:
            js = json.loads(msg)
        except:
            try:
                js = json.loads(msg.strip("` \n"))
            except:
                js = {
                    "explanation": msg,
                    "is_fraud": None,
                    "fraud_type": None,
                    "confidence": 0.0,
                    "evidence": {}
                }

        samples.append(js)

        if js["is_fraud"] is not None:
            all_is_fraud.append(js["is_fraud"])
        if js["fraud_type"] is not None:
            all_fraud_types.append(js["fraud_type"])
        all_confidences.append(js.get("confidence", 0))

        # Token stats
        total_input_tokens += response.usage.prompt_tokens
        total_output_tokens += response.usage.completion_tokens
        total_cost_usd += estimate_cost(
            model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )

    # -----------------------------------------------------
    # Consensus: fraud = majority vote
    # -----------------------------------------------------
    if all_is_fraud:
        try:
            consensus_fraud = mode(all_is_fraud)
        except:
            consensus_fraud = Counter(all_is_fraud).most_common(1)[0][0]
    else:
        consensus_fraud = None

    # -----------------------------------------------------
    # Consensus: fraud type (conditional)
    # -----------------------------------------------------
    if consensus_fraud:
        if all_fraud_types:
            try:
                consensus_type = mode(all_fraud_types)
            except:
                consensus_type = Counter(all_fraud_types).most_common(1)[0][0]
        else:
            consensus_type = None
    else:
        consensus_type = None

    # -----------------------------------------------------
    # Agreement statistics
    # -----------------------------------------------------
    agreement_rate = (
        sum(1 for s in samples if s["is_fraud"] == consensus_fraud)
        / len(samples)
    )

    if consensus_fraud:
        if all_fraud_types:
            fraud_type_agreement = (
                sum(1 for t in all_fraud_types if t == consensus_type)
                / len(all_fraud_types)
            )
        else:
            fraud_type_agreement = 0.0
    else:
        fraud_type_agreement = None

    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    consensus_output = {
        "node_id": node_id,
        "is_fraud": consensus_fraud,
        "fraud_type": consensus_type,
        "agreement_rate": agreement_rate,
        "fraud_type_agreement": fraud_type_agreement,
        "avg_confidence": avg_confidence,
    }

    return {
        "samples": samples,
        "consensus": consensus_output,
        "agreement_rate": agreement_rate,
        "fraud_type_agreement": fraud_type_agreement,
        "avg_confidence": avg_confidence,
        "avg_latency": total_latency / num_samples,
        "total_cost_usd": total_cost_usd,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens
    }
