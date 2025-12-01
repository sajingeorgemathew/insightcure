import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_insight(task_type: str,
                     target: str,
                     metrics: dict,
                     dataset_name: str = None,
                     **kwargs):    # <-- accepts extra parameters safely

    if not client.api_key:
        return "Error: OPENAI_API_KEY is missing. Add it in Render Environment Variables."

    prompt = f"""
You are an analytics consultant.

Dataset: {dataset_name or 'N/A'}
Task type: {task_type}
Target variable: {target}

Model metrics:
{json.dumps(metrics, indent=2)}

Additional Information:
{json.dumps(kwargs, indent=2)}

Please provide:
1. A simple explanation of performance.
2. Strengths & weaknesses of the model.
3. 2–3 business actions.
4. Keep response under 250 words.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "Explain analytics clearly to non-technical business stakeholders."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"OpenAI Error: {e}"
