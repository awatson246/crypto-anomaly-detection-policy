import os
import openai
from dotenv import load_dotenv

load_dotenv()

def interpret_with_openai(node_id, top_features, node_data, model="gpt-3.5-turbo"):
    """
    Uses OpenAI API to generate an interpretation of a flagged node based on
    GraphLIME feature importance and actual node attributes.
    """

    fraud_types = (
        "Ponzi schemes, phishing attacks, pump-and-dump schemes, ransomware, "
        "SIM swapping, mining malware, giveaway scams, impersonation scams, securities fraud"
    )

    formatted_weights = "\n".join(
        [f"- {name}: {value:.3e}" for name, value in top_features]
    )

    formatted_data = "\n".join(
        [f"- {k}: {v}" for k, v in node_data.items()]
    )

    # Few-shot prompt with clear examples
    prompt = f"""
You are a financial crime analyst specializing in cryptocurrency fraud. 
A graph-based anomaly detection model has flagged the following wallet as suspicious.

Your task is to analyze both:
1. The top features that *influenced the model's decision* (from GraphLIME), and
2. The actual transaction statistics of the wallet.

**Note:** The feature importance scores do NOT reflect actual values — they only indicate how strongly each feature contributed to the anomaly detection.

---
**Example 1**  
**Feature Importances (from GraphLIME):**
- btc_sent_total: 9.812e-01  
- degree: 3.442e-02  
- btc_received_total: 0.000e+00  

**Actual Node Data:**
- btc_sent_total: 0.0  
- btc_received_total: 45.1  
- degree: 24  

**Interpretation**:  
Even though the model heavily weighted `btc_sent_total`, the actual value is 0 — indicating the node received a lot of funds but hasn’t sent anything. This may suggest hoarding behavior, commonly seen in Ponzi schemes.

---
**Example 2**  
**Feature Importances (from GraphLIME):**
- transacted_w_address_mean: 7.623e-01  
- degree: 1.124e-01  
- btc_sent_total: 4.132e-03  

**Actual Node Data:**
- btc_sent_total: 90.55  
- btc_received_total: 21.38  
- total_txs: 27  
- transacted_w_address_mean: 1.0  

**Interpretation**:  
The model flagged the wallet based on consistent interactions with unique addresses (`transacted_w_address_mean`), but the node also exhibits significant sending behavior. This could suggest transaction structuring or layering in a money laundering pattern.

---
Now analyze this real case:

**Node ID**: {node_id}

**Features that most influenced the anomaly model (importance scores only):**
{formatted_weights}

**Actual Node Values:**
{formatted_data}

---
Your tasks:
1. Explain the suspicious behavior based on these two views.
2. If appropriate, classify it using known crypto fraud types: {fraud_types}
3. If the behavior appears normal, say so explicitly.
"""

    client = openai.OpenAI()  # Assumes API key is stored in .env
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    return response.choices[0].message.content.strip()
