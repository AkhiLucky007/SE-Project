import random

class TimeOptimizer:

    def __init__(self, predictor):
        self.predictor = predictor

    def find_best_time(self, caption, base_data):

        # 🔥 make randomness consistent per caption
        seed = hash(caption) % (10**6)
        random.seed(seed)

        # weighted realistic times
        time_weights = {
            9: 1,
            10: 1,
            11: 2,
            12: 3,
            13: 3,
            14: 2,
            15: 1,
            16: 2,
            17: 3,
            18: 5,
            19: 5,
            20: 4,
            21: 3,
            22: 2
        }

        hours = []
        for hour, weight in time_weights.items():
            hours.extend([hour] * weight)

        return random.choice(hours)