from dotenv import load_dotenv
load_dotenv()
import os
import json

from openai import OpenAI

def generate_insight(task_type: str, target: str, metrics: dict, dataset_name: str = None):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI API key not found. Set OPENAI_API_KEY as an environment variable."

    client = OpenAI(api_key=api_key)

    prompt = f"""
You are an analytics consultant.

Dataset: {dataset_name or 'N/A'}
Task type: {task_type}
Target variable: {target}
Model metrics:
{json.dumps(metrics, indent=2)}

1. Explain the performance in simple language.
2. Comment on strengths/weaknesses of the models.
3. Suggest 2–3 business actions based on such a model.
4. Keep the answer under 250 words.
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Explain analytics clearly to non-technical business stakeholders."},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content

    except Exception as e:
        return f"Error calling OpenAI: {e}"
