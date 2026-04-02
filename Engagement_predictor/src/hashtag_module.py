import re
from collections import Counter

class HashtagRecommender:

    def __init__(self):
        self.top_hashtags = []
        self.hashtag_scores = {}

    # -----------------------------
    # 🧠 Extract hashtags
    # -----------------------------
    def extract_hashtags(self, text):
        return re.findall(r"#\w+", text.lower())

    # -----------------------------
    # 📊 Train on dataset
    # -----------------------------
    def fit(self, df):
        hashtag_counter = Counter()
        engagement_scores = {}

        for _, row in df.iterrows():
            tags = self.extract_hashtags(str(row['description']))
            engagement = (row['likes'] + row['comments']) / row['followers']

            for tag in tags:
                hashtag_counter[tag] += 1

                if tag not in engagement_scores:
                    engagement_scores[tag] = []

                engagement_scores[tag].append(engagement)

        # Average engagement per hashtag
        for tag in engagement_scores:
            engagement_scores[tag] = sum(engagement_scores[tag]) / len(engagement_scores[tag])

        # Store
        self.top_hashtags = [tag for tag, _ in hashtag_counter.most_common(100)]
        self.hashtag_scores = engagement_scores

    # -----------------------------
    # 🔥 Recommend hashtags
    # -----------------------------
    def recommend(self, caption, predictor, base_data, top_k=5):

        candidates = self.top_hashtags[:20]  # shortlist

        scored = []

        for tag in candidates:
            new_caption = caption + " " + tag

            temp = base_data.copy()
            df = pd.DataFrame([temp])
            df["description"] = new_caption

            pred, probs = predictor.predict(df)
            score = probs[0][2]  # HIGH engagement prob

            scored.append((tag, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [tag for tag, _ in scored[:top_k]]