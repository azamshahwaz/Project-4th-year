import os
import json

from groq import Groq

from dotenv import load_dotenv


# =========================================
# LOAD ENV VARIABLES
# =========================================

load_dotenv()


# =========================================
# GROQ CLIENT
# =========================================

client = Groq(

    api_key=os.getenv(
        "GROQ_API_KEY"
    )
)


# =========================================
# LLM RECOMMENDATIONS
# =========================================

def get_llm_recommendations(
    metrics_text
):

    prompt = f"""
You are a Responsible AI expert.

Analyze the dataset metrics below.

Return ONLY valid JSON.

Metrics:
{metrics_text}

Allowed actions:

- apply_smote
- remove_proxy_bias
- fix_skewness
- remove_duplicates
- fill_missing_values

Return JSON like this:

{{
    "apply_smote": true,
    "remove_proxy_bias": false,
    "fix_skewness": true,
    "remove_duplicates": true,
    "fill_missing_values": true
}}
"""

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0
    )

    content = (

        response
        .choices[0]
        .message
        .content
    )

    # =====================================
    # SAFE JSON EXTRACTION
    # =====================================

    start = content.find("{")

    end = content.rfind("}") + 1

    json_text = content[start:end]

    recommendations = json.loads(
        json_text
    )

    return recommendations