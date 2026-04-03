import random

class TimeOptimizer:

    def __init__(self, predictor):
        self.predictor = predictor

    def find_best_time(self, caption, base_data):

        # SAME caption → SAME time
        seed = hash(caption) % (10**6)
        random.seed(seed)

        # realistic Instagram hours
        time_weights = {
            12: 3,
            13: 3,
            14: 2,
            17: 3,
            18: 5,
            19: 5,
            20: 4,
            21: 3
        }

        hours = []
        for hour, weight in time_weights.items():
            hours.extend([hour] * weight)

        return random.choice(hours)