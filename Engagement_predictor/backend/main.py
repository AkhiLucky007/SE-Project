from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
app = FastAPI()

API_KEY = os.getenv("GROQ_API_KEY")

class CaptionRequest(BaseModel):
    caption: str

@app.post("/generate")
def generate(req: CaptionRequest):

    url = "https://api.groq.com/openai/v1/chat/completions"

    prompt = f"""
You are a top Instagram content creator.

STRICT RULES:
- MUST rewrite the caption (do NOT repeat it)
- Keep same meaning
- Make it more engaging, emotional, human
- Keep it short and punchy

Original Caption:
{req.caption}

FORMAT:

CAPTION:
<rewritten caption>

HASHTAGS:
8 highly relevant hashtags (NO generic tags)
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
            "temperature": 1.4,
            "top_p": 0.9,
            "presence_penalty": 1.5
        },
        timeout=30
    )

    data = response.json()

    if "choices" not in data:
        return {"result": f"API ERROR: {data}"}

    output = data["choices"][0]["message"]["content"]

    # HARD FIX → prevent same caption
    if req.caption.strip().lower() in output.lower():
        output = output.replace(req.caption, "").strip()

    return {
        "result": output
    }