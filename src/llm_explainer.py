import os
import openai
from dotenv import load_dotenv

load_dotenv()

def interpret_with_openai(insight_text, model="gpt-3.5-turbo"):
    prompt = (
        "You're a financial crime analyst studying cryptocurrency transactions. Based on the following anomaly description, "
        "explain what suspicious behavior might be occurring and whether it may violate anti-money laundering policy:\n\n"
        f"{insight_text}"
    )

    client = openai.OpenAI()  # uses .env key automatically
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    return response.choices[0].message.content


