import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import joblib
import numpy as np

from src.caption_module import CaptionAnalyzer
from src.engagement_utils import EngagementEstimator
from src.ab_testing import CaptionABTester
from src.hashtag_module import HashtagRecommender
from src.predict import Predictor
from src.time_optimizer import TimeOptimizer
from src.llama_caption_engine import LlamaCaptionEngine

# -----------------------------
# 🔥 Load model (for A/B testing)
# -----------------------------
model = joblib.load(
    os.path.join(os.path.dirname(__file__), "..", "src", "engagement_model.pkl")
)

# -----------------------------
# 🔧 Initialize modules
# -----------------------------
caption_analyzer = CaptionAnalyzer()
estimator = EngagementEstimator()
tester = CaptionABTester(model)
llama_engine = LlamaCaptionEngine()

predictor = Predictor()
time_optimizer = TimeOptimizer(predictor)
hashtag_model = HashtagRecommender()

# -----------------------------
# 🔹 Load best time (static fallback)
# -----------------------------
def load_best_time():
    path = os.path.join(os.path.dirname(__file__), "..", "results", "best_time.txt")
    with open(path, "r") as f:
        day, hour = f.read().split(",")
        return day, int(hour)

best_day, best_hour = load_best_time()

# -----------------------------
# 🧠 UI
# -----------------------------
st.set_page_config(page_title="Instagram AI Assistant", layout="centered")
st.title("📸 Instagram AI Assistant")

# -----------------------------
# 📝 Inputs
# -----------------------------
description = st.text_area("Caption")

col1, col2 = st.columns(2)
with col1:
    followers = st.number_input("Followers", min_value=1)
    following = st.number_input("Following", min_value=0)

with col2:
    num_posts = st.number_input("Total Posts", min_value=0)
    is_business = st.selectbox("Business Account?", [0, 1])

date = st.datetime_input("Post Time")

# -----------------------------
# 🚀 MAIN PREDICTION
# -----------------------------
if st.button("Predict"):

    df = pd.DataFrame([{
        "description": description,
        "followers": followers,
        "following": following,
        "num_posts": num_posts,
        "is_business_account": is_business,
        "date": str(date)
    }])

    # Feature engineering
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek

    df['followers'] = np.log1p(df['followers'])
    df['following'] = np.log1p(df['following'])
    df['num_posts'] = np.log1p(df['num_posts'])

    # -----------------------------
    # 🔮 Prediction
    # -----------------------------
    pred, probs = predictor.predict(df)
    pred = pred[0]
    probs = probs[0]
    confidence = max(probs)

    mapping = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    st.subheader("📊 Engagement Prediction")
    st.success(mapping[pred])
    st.write(f"Confidence: {confidence:.2f}")

    # -----------------------------
    # ❤️ Engagement Range
    # -----------------------------
    likes = estimator.estimate_likes(followers, pred)
    reach = estimator.estimate_reach(pred)

    st.subheader("❤️ Expected Performance")
    st.write(f"Likes: {likes['min_likes']} – {likes['max_likes']}")
    st.write(f"Reach: {reach}")

    # -----------------------------
    # 📝 Caption Analysis
    # -----------------------------
    result = caption_analyzer.analyze(description)

    st.subheader("📊 Caption Score")
    st.write(f"{result['score']} / 10")

    st.subheader("💡 Suggestions")
    for s in result["suggestions"]:
        st.write("-", s)

    optimized_caption = llama_engine.generate_caption(description)

    st.subheader("🤖 AI Optimized Caption")
    st.write(optimized_caption)

    st.subheader("✍️ Optimized Caption (AI)")
    st.write(optimized_caption)

    # -----------------------------
    # ⏰ Best Time (static fallback)
    # -----------------------------
    st.subheader("⏰ Best Time (General)")
    st.info(f"{best_day} at {best_hour}:00")

    # -----------------------------
    # 🔥 ML Optimized Time + Hashtags
    # -----------------------------
    base_data = {
        "followers": followers,
        "following": following,
        "num_posts": num_posts,
        "is_business_account": is_business,
        "hour": df["hour"].iloc[0],
        "day_of_week": df["day_of_week"].iloc[0]
    }

    best_hour_opt = time_optimizer.find_best_time(description, base_data)

    st.subheader("⏰ Optimized Best Time")
    st.info(f"Best hour: {best_hour_opt}:00")

    hashtags = hashtag_model.recommend(description, predictor, base_data)

    st.subheader("🏷️ Recommended Hashtags")
    st.write(" ".join(hashtags))

# -----------------------------
# 🔥 A/B TESTING SECTION
# -----------------------------
st.divider()
st.subheader("⚖️ Compare Two Captions")

caption_a = st.text_area("Caption A")
caption_b = st.text_area("Caption B")

if st.button("Compare"):

    base_data = {
        "followers": followers,
        "following": following,
        "num_posts": num_posts,
        "is_business_account": is_business,
        "date": str(date),
        "hour": pd.to_datetime(date).hour,
        "day_of_week": pd.to_datetime(date).dayofweek
    }

    result = tester.compare(caption_a, caption_b, base_data)

    st.write(f"Winner: {result['winner']}")
    st.write(f"Improvement: {result['improvement_percent']}%")