class EngagementEstimator:

    def __init__(self):
        # % ranges based on engagement class
        self.ranges = {
            0: (0.01, 0.05),   # Low
            1: (0.05, 0.15),   # Medium
            2: (0.15, 0.30)    # High
        }

    # -----------------------------
    # 📊 Estimate likes range
    # -----------------------------
    def estimate_likes(self, followers, engagement_class):

        if followers < 10000:
            scale = 1.0
        elif followers < 100000:
            scale = 0.6
        else:
            scale = 0.3

        low, high = self.ranges.get(engagement_class, (0, 0))

        return {
        "min_likes": int(followers * low * scale),
        "max_likes": int(followers * high * scale)
        }
    def estimate_reach(self, engagement_class):
        if engagement_class == 0:
            return "Limited reach (mostly followers)"
        elif engagement_class == 1:
            return "Moderate reach (some non-followers)"
        else:
            return "High reach (likely explore page)"