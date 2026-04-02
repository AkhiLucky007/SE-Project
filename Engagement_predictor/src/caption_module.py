import re
import os
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class CaptionAnalyzer:

    def __init__(self):
        base_path = os.path.dirname(__file__)

        # Load retrieval data
        self.corpus = joblib.load(os.path.join(base_path, "caption_corpus.pkl"))
        self.tfidf = joblib.load(os.path.join(base_path, "tfidf_vectorizer.pkl"))
        self.tfidf_matrix = joblib.load(os.path.join(base_path, "tfidf_matrix.pkl"))

    # -----------------------------
    # 🧠 Feature extraction
    # -----------------------------
    def extract_features(self, caption: str):
        words = caption.split()

        return {
            "length": len(caption),
            "num_words": len(words),
            "num_hashtags": caption.count("#"),
            "num_mentions": caption.count("@"),
            "has_question": "?" in caption,
            "has_exclamation": "!" in caption,
            "num_emojis": len(re.findall(r'[^\w\s,]', caption))
        }

    # -----------------------------
    # 📊 Caption scoring
    # -----------------------------
    def score_caption(self, features):
        score = 0

        if 50 <= features["length"] <= 200:
            score += 2
        elif features["length"] > 20:
            score += 1

        if 3 <= features["num_hashtags"] <= 8:
            score += 2
        elif features["num_hashtags"] > 0:
            score += 1

        if features["has_question"]:
            score += 2

        if features["has_exclamation"]:
            score += 1

        if 1 <= features["num_emojis"] <= 5:
            score += 2

        return min(score, 10)

    # -----------------------------
    # 💡 Suggestions
    # -----------------------------
    def get_suggestions(self, features):
        suggestions = []

        if features["length"] < 40:
            suggestions.append("Increase caption length for better storytelling")

        if features["num_hashtags"] < 3:
            suggestions.append("Add 3–5 relevant hashtags")

        if not features["has_question"]:
            suggestions.append("Ask a question to boost engagement")

        if features["num_emojis"] == 0:
            suggestions.append("Add emojis to make caption more expressive")

        if features["num_mentions"] == 0:
            suggestions.append("Tag relevant accounts if applicable")

        return suggestions

    # -----------------------------
    # 🔍 Retrieve similar captions
    # -----------------------------
    def get_similar_captions(self, caption, top_k=5):
        vec = self.tfidf.transform([caption])
        sims = cosine_similarity(vec, self.tfidf_matrix)

        idx = sims[0].argsort()[-top_k:][::-1]
        return [self.corpus[i] for i in idx]

    # -----------------------------
    # ✍️ Generate candidates (RAG style)
    # -----------------------------
    def generate_candidates(self, caption):
        similar = self.get_similar_captions(caption)

        candidates = []

        for s in similar:
            candidates.append(caption + " " + s[:80])  # mix
            candidates.append(s + " 🔥")
            candidates.append(caption + " ✨")

        return candidates

    # -----------------------------
    # 🧠 Pick best using ML model
    # -----------------------------
    def pick_best_caption(self, caption, predictor, base_data):

        candidates = self.generate_candidates(caption)

        best_caption = caption
        best_score = -1

        for c in candidates:
            temp = base_data.copy()
            df = pd.DataFrame([temp])
            df["description"] = c

            pred, probs = predictor.predict(df)
            score = probs[0][2]  # HIGH engagement

            if score > best_score:
                best_score = score
                best_caption = c

        return best_caption

    # -----------------------------
    # 🔁 Full analysis
    # -----------------------------
    def analyze(self, caption: str):
        features = self.extract_features(caption)
        score = self.score_caption(features)
        suggestions = self.get_suggestions(features)

        return {
            "score": score,
            "features": features,
            "suggestions": suggestions
        }