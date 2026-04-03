import pandas as pd
import joblib
import os


class Predictor:

    def __init__(self):

        base_path = os.path.dirname(__file__)

        model_bundle = joblib.load(
            os.path.join(base_path, "engagement_model.pkl")
        )

        if isinstance(model_bundle, dict):
            self.model = model_bundle["model"]
        else:
            self.model = model_bundle

        # Correct feature list for THIS trained model
        self.meta_columns = [
            'hour',
            'day_of_week',
            'followers',
            'following',
            'num_posts',
            'is_business_account'
        ]

    def predict(self, df: pd.DataFrame):

        X = df[self.meta_columns].astype(float).values

        pred = self.model.predict(X)
        probs = self.model.predict_proba(X)

        return pred, probs