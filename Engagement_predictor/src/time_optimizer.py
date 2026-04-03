import hashlib
import re


class TimeOptimizer:

    def __init__(self, predictor):
        self.predictor = predictor

        # Caption category → best engagement hours
        self.category_hours = {
            "travel": 19,
            "fitness": 18,
            "food": 13,
            "selfie": 20,
            "friends": 21,
            "fashion": 19,
            "nature": 17,
            "motivation": 9,
            "study": 10,
            "work": 12,
            "night": 20,
            "morning": 9,
            "default": 18
        }

    # -----------------------------
    # 🔍 Detect caption category
    # -----------------------------
    def detect_category(self, caption):

        caption = caption.lower()

        keyword_map = {
            "travel": ["travel", "trip", "vacation", "beach", "mountain"],
            "fitness": ["gym", "workout", "fitness", "training"],
            "food": ["food", "eat", "dinner", "lunch", "coffee"],
            "selfie": ["selfie", "me", "myself"],
            "friends": ["friends", "squad", "bros", "gang"],
            "fashion": ["outfit", "style", "fashion", "ootd"],
            "nature": ["nature", "sunset", "sunrise", "sky"],
            "motivation": ["motivation", "grind", "focus"],
            "study": ["study", "exam", "books"],
            "work": ["office", "work", "meeting"]
        }

        for category, keywords in keyword_map.items():
            for word in keywords:
                if word in caption:
                    return category

        return "default"

    # -----------------------------
    # 🔒 deterministic fallback hash
    # -----------------------------
    def caption_hash_bucket(self, caption):

        caption_hash = hashlib.md5(
            caption.encode()
        ).hexdigest()

        bucket = int(caption_hash[:4], 16)

        valid_hours = [12, 13, 17, 18, 19, 20, 21]

        return valid_hours[bucket % len(valid_hours)]

    # -----------------------------
    # ⏰ MAIN function
    # -----------------------------
    def find_best_time(self, caption, base_data):

        category = self.detect_category(caption)

        if category in self.category_hours:
            return self.category_hours[category]

        return self.caption_hash_bucket(caption)