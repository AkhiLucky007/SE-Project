from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
import hashlib

app = FastAPI()

API_KEY = os.getenv("GROQ_API_KEY")


class CaptionRequest(BaseModel):
    caption: str


def fallback_caption(original):

    words = original.split()

    if len(words) <= 3:
        return f"{original} ✨ Moments like this matter."

    return f"{original.capitalize()} — living this moment fully."


@app.post("/generate")
def generate(req: CaptionRequest):

    url = "https://api.groq.com/openai/v1/chat/completions"

    prompt = f"""
Rewrite this Instagram caption.

RULES:
- MUST change wording
- KEEP same meaning
- SHORT
- HUMAN
- ENGAGING
- DO NOT repeat original sentence structure

Caption:
{req.caption}

FORMAT:

CAPTION:
<new caption>

HASHTAGS:
8 relevant hashtags
"""

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.5,
            "presence_penalty": 2
        },
        timeout=30
    )

    data = response.json()

    if "choices" not in data:
        return {"result": fallback_caption(req.caption)}

    output = data["choices"][0]["message"]["content"]

    # HARD GUARANTEE caption != original
    if req.caption.lower().strip() in output.lower():

        improved = fallback_caption(req.caption)

        hashtags = "#instagram #content #creator #engagement #post #photo #instadaily #social"

        output = f"""
CAPTION:
{improved}

HASHTAGS:
{hashtags}
"""

    return {
        "result": output
    }
