import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st # type: ignore
import pandas as pd
import joblib # type: ignore
import numpy as np
import requests # type: ignore

from src.engagement_utils import EngagementEstimator
from src.ab_testing import CaptionABTester
from src.predict import Predictor
from src.time_optimizer import TimeOptimizer

model = joblib.load(
    os.path.join(os.path.dirname(__file__), "..", "src", "engagement_model.pkl")
)

estimator = EngagementEstimator()
tester = CaptionABTester(model)
predictor = Predictor()
time_optimizer = TimeOptimizer(predictor)

st.set_page_config(page_title="Instagram AI Assistant", layout="centered")
st.title("📸 Instagram AI Assistant")

description = st.text_area("Caption")

col1, col2 = st.columns(2)
with col1:
    followers = st.number_input("Followers", min_value=1)
    following = st.number_input("Following", min_value=0)

with col2:
    num_posts = st.number_input("Total Posts", min_value=0)
    is_business = st.selectbox("Business Account?", [0, 1])

date = st.datetime_input("Post Time")

if st.button("Predict"):

    df = pd.DataFrame([{
        "description": description,
        "followers": followers,
        "following": following,
        "num_posts": num_posts,
        "is_business_account": is_business,
        "date": str(date)
    }])

    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek

    df['followers'] = np.log1p(df['followers'])
    df['following'] = np.log1p(df['following'])
    df['num_posts'] = np.log1p(df['num_posts'])

    pred, probs = predictor.predict(df)
    confidence = max(probs[0])

    mapping = {0: "Low", 1: "Medium", 2: "High"}

    st.subheader("📊 Engagement Prediction")
    st.success(mapping[pred[0]])
    st.write(f"Confidence: {confidence:.2f}")

    likes = estimator.estimate_likes(followers, pred[0])
    reach = estimator.estimate_reach(pred[0])

    st.subheader("❤️ Expected Performance")
    st.write(f"Likes: {likes['min_likes']} – {likes['max_likes']}")
    st.write(f"Reach: {reach}")

    base_data = {
        "followers": followers,
        "following": following,
        "num_posts": num_posts,
        "is_business_account": is_business,
        "hour": df["hour"].iloc[0],
        "day_of_week": df["day_of_week"].iloc[0]
    }

    response = requests.post(
        "http://127.0.0.1:8000/generate",
        json={"caption": description}
    )

    result_text = response.json().get("result", "")

    # ✅ NEW PARSER (IMPORTANT)
    best_caption = description
    if "CAPTION:" in result_text:
        best_caption = result_text.split("CAPTION:")[1].split("HASHTAGS:")[0].strip()

    hashtags = []
    if "HASHTAGS:" in result_text:
        tag_part = result_text.split("HASHTAGS:")[1]
        hashtags = [w for w in tag_part.split() if w.startswith("#")]

    st.subheader("🤖 AI Optimized Caption")
    st.write(best_caption)

    st.subheader("🏷️ AI Hashtags")
    st.write(" ".join(hashtags))

    best_hour = time_optimizer.find_best_time(description, base_data)

    st.subheader("⏰ Best Time to Post")
    st.info(f"{best_hour}:00")