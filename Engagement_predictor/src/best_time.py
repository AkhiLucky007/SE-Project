import pandas as pd

class BestTimePredictor:

    def __init__(self):
        self.best_hour = None
        self.best_day = None

    # -----------------------------
    # 🧠 Train from dataset
    # -----------------------------
    def fit(self, df):
        df = df.copy()

        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['day'] = df['date'].dt.day_name()

        # Engagement rate
        df['engagement_rate'] = (df['likes'] + df['comments']) / df['followers']

        # Group by hour
        hour_perf = df.groupby('hour')['engagement_rate'].mean()
        self.best_hour = hour_perf.idxmax()

        # Group by day
        day_perf = df.groupby('day')['engagement_rate'].mean()
        self.best_day = day_perf.idxmax()

    # -----------------------------
    # ⏰ Predict best time
    # -----------------------------
    def predict(self):
        return {
            "best_hour": int(self.best_hour),
            "best_day": self.best_day
        }