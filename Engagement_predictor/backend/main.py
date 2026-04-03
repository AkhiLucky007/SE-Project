from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()

API_KEY = os.getenv("GROQ_API_KEY")

class CaptionRequest(BaseModel):
    caption: str


def fallback_caption(original):
    return f"{original.capitalize()} — capturing moments that matter ✨"


def fallback_hashtags(text):
    words = text.lower().split()
    base = [w for w in words if len(w) > 3][:5]
    tags = [f"#{w}" for w in base]

    # ensure always 8
    while len(tags) < 8:
        tags.append("#lifestyle")

    return " ".join(tags[:8])


@app.post("/generate")
def generate(req: CaptionRequest):

    prompt = f"""
You are a top Instagram creator.

TASK:
Rewrite the caption into a HIGH-ENGAGEMENT Instagram caption.

STRICT RULES:
- MUST NOT repeat original sentence
- MUST change wording significantly
- Keep same meaning
- Add emotion / storytelling
- Keep it SHORT and natural

EMOJI RULES:
- Add 1–3 relevant emojis
- Emojis must match the vibe (travel, food, gym, etc.)
- DO NOT spam emojis

HASHTAG RULES:
- MUST be specific to caption
- NO generic tags (#instagram #viral)
- Mix niche + broad

Caption:
{req.caption}

FORMAT:

CAPTION:
<new caption with emojis>

HASHTAGS:
#tag1 #tag2 #tag3 #tag4 #tag5 #tag6 #tag7 #tag8
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.3,
            "top_p": 0.9,
            "presence_penalty": 1.5
        },
        timeout=30
    )

    data = response.json()

    # ❌ API failed → fallback
    if "choices" not in data:
        return {
            "result": f"""
CAPTION:
{fallback_caption(req.caption)}

HASHTAGS:
{fallback_hashtags(req.caption)}
"""
        }

    output = data["choices"][0]["message"]["content"]

    # 🚨 HARD FIX → if model still copies input
    if req.caption.lower().strip() in output.lower():

        new_caption = fallback_caption(req.caption)
        new_tags = fallback_hashtags(req.caption)

        output = f"""
CAPTION:
{new_caption}

HASHTAGS:
{new_tags}
"""

    return {"result": output} 