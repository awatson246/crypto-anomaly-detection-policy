import os
import time
import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI()

# Pricing lookup (you can update when needed)
MODEL_PRICES = {
    "gpt-4o-mini": {"input": 0.15/1_000_000, "output": 0.60/1_000_000},
    "gpt-4o": {"input": 0.80/1_000_000, "output": 3.20/1_000_000},
    "gpt-3.5-turbo": {"input": 0.50/1_000_000, "output": 1.50/1_000_000},
}

def _estimate_cost(model, in_tokens, out_tokens):
    if model not in MODEL_PRICES:
        return None
    pricing = MODEL_PRICES[model]
    return (in_tokens * pricing["input"]) + (out_tokens * pricing["output"])


def _retry_api_call(fn, max_retries=12):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "limit" in err:
                time.sleep(10 * (attempt + 1))
                continue
            if "timeout" in err or "503" in err or "connection" in err:
                time.sleep(5 * (attempt + 1))
                continue
            return f"[ERROR:{e}]"
    return "[FAILED AFTER MAX RETRIES]"


def interpret_with_openai(node_id, top_features, node_data, model="gpt-4o-mini"):
    fraud_types = (
        "Ponzi schemes, phishing attacks, pump-and-dump schemes, ransomware, "
        "SIM swapping, mining malware, giveaway scams, impersonation scams, securities fraud"
    )

    formatted_weights = "\n".join([f"- {n}: {v:.3e}" for n, v in top_features])
    formatted_data = "\n".join([f"- {k}: {v}" for k, v in node_data.items()])

    prompt = f"""
You are a cryptocurrency forensics analyst specializing in wallet-level behavior.

A graph-based anomaly detection model has flagged a wallet as suspicious.
You will receive:
1. GraphLIME feature importance weights (importance only, not actual values)
2. Raw wallet statistics (actual values)
3. A list of fraud types

You must produce:
- A clear explanation of why the wallet may be anomalous
- A fraud-type classification *only if applicable*
- A statement if the wallet likely appears normal

---

### Example 1
Feature importances:
- btc_sent_total: 9.812e-01
- degree: 3.442e-02
- btc_received_total: 0.000e+00

Actual values:
- btc_sent_total: 0.0
- btc_received_total: 45.1
- degree: 24

**Interpretation**: The model heavily weighted sending, but the actual value is 0, while receiving is high. This appears consistent with Ponzi inflow behavior (accumulates funds but does not return them).

---

### Example 2
Feature importances:
- transacted_w_address_mean: 7.623e-01
- degree: 1.124e-01
- btc_sent_total: 4.132e-03

Actual values:
- btc_sent_total: 90.55
- btc_received_total: 21.38
- total_txs: 27
- transacted_w_address_mean: 1.0

**Interpretation**: Despite importance on address-mean, the real pattern suggests layering behavior (high movement of funds through many addresses).

---

### Real Case
Node ID: {node_id}

GraphLIME Important Features:
{formatted_weights}

Actual Node Values:
{formatted_data}

Fraud types available for classification:
{fraud_types}

Your analysis should explain the behavior and assess the likelihood and nature of fraud, if present.
"""

    start = time.time()

    def _do_call():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

    response = _retry_api_call(_do_call)

    latency = time.time() - start

    # If error string returned
    if isinstance(response, str):
        return response, latency, 0, 0, None

    in_tokens = response.usage.prompt_tokens
    out_tokens = response.usage.completion_tokens
    cost = _estimate_cost(model, in_tokens, out_tokens)

    message = response.choices[0].message["content"].strip()

    return message, latency, in_tokens, out_tokens, cost