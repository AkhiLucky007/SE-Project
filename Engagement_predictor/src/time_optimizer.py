import pandas as pd

class TimeOptimizer:

    def __init__(self, predictor):
        self.predictor = predictor

    def find_best_time(self, caption, base_data):

        best_hour = None
        best_score = -1

        for hour in range(24):

            temp = base_data.copy()
            temp["hour"] = hour

            df = pd.DataFrame([temp])
            df["description"] = caption

            pred, probs = self.predictor.predict(df)
            score = probs[0][2]  # probability of HIGH

            if score > best_score:
                best_score = score
                best_hour = hour

        return best_hour