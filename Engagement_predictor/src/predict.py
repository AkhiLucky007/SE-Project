import pandas as pd
import joblib
import os

class Predictor:

    def __init__(self):

        base_path = os.path.dirname(__file__)

        # Load trained engagement model
        self.model = joblib.load(
            os.path.join(base_path, "engagement_model.pkl")
        )

    # -----------------------------
    # 🔮 Predict engagement
    # -----------------------------
    def predict(self, df: pd.DataFrame):

        # These are the ONLY features your model needs
        meta_cols = [
            'hour',
            'day_of_week',
            'followers',
            'following',
            'num_posts',
            'is_business_account'
        ]

        X = df[meta_cols].astype(float).values

        pred = self.model.predict(X)
        probs = self.model.predict_proba(X)

        return pred, probs