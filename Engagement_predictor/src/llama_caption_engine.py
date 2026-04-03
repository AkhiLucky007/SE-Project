import torch
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class LlamaCaptionEngine:

    def __init__(self):

        print("Loading embeddings...")

        with open("caption_embeddings.pkl", "rb") as f:
            self.embeddings = pickle.load(f)

        with open("caption_corpus.pkl", "rb") as f:
            self.corpus = pickle.load(f)

        print("Loading retriever model...")

        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Loading LLaMA model...")

        model_name = "meta-llama/Llama-2-7b-chat-hf"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )

        print("Caption engine ready")


    def retrieve_examples(self, caption, k=5):

        query_emb = self.embedder.encode([caption])

        sims = cosine_similarity(query_emb, self.embeddings)[0]

        idx = np.argsort(sims)[-k:]

        return [self.corpus[i] for i in idx]


    def generate_caption(self, caption):

        examples = self.retrieve_examples(caption)

        example_text = "\n".join(examples)

        prompt = f"""
You are an Instagram caption expert.

Original caption:
{caption}

Similar captions:
{example_text}

Rewrite a better Instagram caption.
Keep tone natural and engaging.
Avoid shortening too much.

New caption:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.9,
            do_sample=True,
            num_return_sequences=3
        )

        captions = []

        for out in outputs:

            decoded = self.tokenizer.decode(
                out,
                skip_special_tokens=True
            )

            captions.append(
                decoded.split("New caption:")[-1].strip()
            )

        return captions