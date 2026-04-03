import random

class TimeOptimizer:

    def __init__(self, predictor):
        self.predictor = predictor

    def find_best_time(self, caption, base_data):

        seed = hash(caption) % (10**6)
        random.seed(seed)

        good_hours = [11, 12, 13, 18, 19, 20, 21]

        return random.choice(good_hours)