import os
import time
import json
import openai
from dotenv import load_dotenv

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

            # Rate limit / overloaded system
            if "rate" in msg or "limit" in msg or "overloaded" in msg:
                print(f"[{tag}] Rate limit, retrying in {wait:.2f}s...")
                time.sleep(wait)
                continue

            # Connection hiccups
            if "timeout" in msg or "503" in msg or "connection" in msg:
                print(f"[{tag}] Transient network error, retrying in {wait:.2f}s...")
                time.sleep(wait)
                continue

            # Other fatal cases
            print(f"[{tag}] Fatal error: {e}")
            return None

    print(f"[{tag}] Failed after maximum retries.")
    return None

# --------------------------------------------------------------------
#   Main function: generates structured JSON + cost + latency metrics
# --------------------------------------------------------------------
def interpret_with_openai(node_id, top_features, node_data, model="gpt-4o-mini"):
    """
    Returns:
        structured_json (dict)
        latency_seconds (float)
        input_tokens (int)
        output_tokens (int)
        cost_usd (float)
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

    # --------------------------------------------------------------
    #  Structured-output prompt (JSON enforced)
    # --------------------------------------------------------------
    prompt = f"""
You are a cryptocurrency forensics analyst.

You will receive:
1. GraphLIME feature importances (importance only)
2. Actual wallet statistics
3. Fraud types

You must return **strict JSON only**:

{{
  "explanation": "...",
  "is_fraud": true/false,
  "fraud_type": "Ponzi schemes" | "phishing attacks" | ... | null,
  "confidence": float between 0 and 1
}}

RULES:
- If behavior appears normal → is_fraud = false, fraud_type = null
- Do NOT hallucinate new fraud types
- Fraud type should be chosen ONLY from the provided list
- Explanation must be concise (2–5 sentences) and discuss why the node is/ isn't fraudulent based on the provided information

---

### Feature Importances (GraphLIME)
{formatted_weights}

### Actual Node Statistics
{formatted_data}

### Known Fraud Types
{fraud_types}

Return ONLY valid JSON.
"""

    def call_api():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0  # reproducibility
        )

    # ----------------------------------
    # LLM call with retry protection
    # ----------------------------------
    start = time.time()
    response = retry_api_call(call_api, tag=f"LLM:{model}")
    latency = time.time() - start

    if response is None:
        return (
            {
                "explanation": "LLM failed after retries",
                "is_fraud": None,
                "fraud_type": None,
                "confidence": 0.0
            },
            latency,
            0, 0, 0.0
        )

    # Extract new-format message
    raw_message = response.choices[0].message.content.strip()

    # Parse JSON robustly
    try:
        structured = json.loads(raw_message)
    except Exception:
        # In case the model wraps JSON in text/etc.
        try:
            structured = json.loads(raw_message.strip("` \n"))
        except:
            structured = {
                "explanation": raw_message,
                "is_fraud": None,
                "fraud_type": None,
                "confidence": 0.0
            }

    # Token usage
    in_tokens = response.usage.prompt_tokens
    out_tokens = response.usage.completion_tokens
    cost_usd = estimate_cost(model, in_tokens, out_tokens)

    return structured, latency, in_tokens, out_tokens, cost_usd
