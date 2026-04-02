import pandas as pd

class CaptionABTester:

    def __init__(self, model):
        self.model = model

    # -----------------------------
    # 🧠 Prepare input (same as predict)
    # -----------------------------
    def prepare_input(self, caption, base_data):
        data = base_data.copy()
        data["description"] = caption
        return pd.DataFrame([data])

    # -----------------------------
    # ⚖️ Compare two captions
    # -----------------------------
    def compare(self, caption_a, caption_b, base_data):
        
        df_a = self.prepare_input(caption_a, base_data)
        df_b = self.prepare_input(caption_b, base_data)

        pred_a = self.model.predict(df_a)[0]
        pred_b = self.model.predict(df_b)[0]

        # Map class to score
        proba_a = self.model.predict_proba(df_a)[0]
        proba_b = self.model.predict_proba(df_b)[0]

        score_a = proba_a[2] + 0.5 * proba_a[1]
        score_b = proba_b[2] + 0.5 * proba_b[1]

        if score_a > score_b:
            winner = "A"
        elif score_b > score_a:
            winner = "B"
        else:
            winner = "Tie"

        # Improvement %
        diff = abs(score_a - score_b)
        improvement = (diff / 3) * 100

        return {
            "caption_a_class": pred_a,
            "caption_b_class": pred_b,
            "winner": winner,
            "improvement_percent": round(improvement, 2)
        }