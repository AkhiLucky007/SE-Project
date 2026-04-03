import torch
import joblib
import numpy as np
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class LlamaCaptionEngine:

    def __init__(self):

        base_path = os.path.dirname(__file__)

        print("Loading embeddings...")

        self.embeddings = joblib.load(os.path.join(base_path, "caption_embeddings.pkl"))
        self.corpus = joblib.load(os.path.join(base_path, "caption_corpus.pkl"))


        print("Loading retriever model...")

        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Loading LLM model...")

        # 🔥 IMPORTANT: use lightweight model locally
        model_name = "distilgpt2"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        print("Caption engine ready")

    # -----------------------------
    # 🔍 Retrieval
    # -----------------------------
    def retrieve_examples(self, caption, k=5):

        query_emb = self.embedder.encode([caption])
        sims = cosine_similarity(query_emb, self.embeddings)[0]

        idx = np.argsort(sims)[-k:]
        return [self.corpus[i] for i in idx]

    # -----------------------------
    # ✍️ Generate captions (MULTIPLE)
    # -----------------------------
    def generate_captions(self, caption, n=5):

        examples = self.retrieve_examples(caption)
        example_text = "\n".join(examples)

        prompt = f"""
Instagram caption:
{caption}

Write a better engaging version:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.9,
            do_sample=True,
            num_return_sequences=n
        )

        captions = []

        for out in outputs:
            decoded = self.tokenizer.decode(out, skip_special_tokens=True)
            captions.append(decoded.replace(prompt, "").strip())

        return captions

    # -----------------------------
    # 🏷️ Hashtag generation
    # -----------------------------
    def generate_hashtags(self, caption):

        prompt = f"""
Generate 8 Instagram hashtags for:
{caption}

Only hashtags:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt")

        output = self.model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.9,
            do_sample=True
        )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return decoded.split("\n")[-1].strip()