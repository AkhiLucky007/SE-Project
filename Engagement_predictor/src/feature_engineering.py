import pandas as pd
import numpy as np
import re

def add_time_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    return df


def log_transform(df):
    df['followers'] = np.log1p(df['followers'])
    df['following'] = np.log1p(df['following'])
    df['num_posts'] = np.log1p(df['num_posts'])
    return df


def extract_hashtags(text):
    return re.findall(r"#(\w+)", str(text).lower())


def add_hashtags(df):
    df['hashtags'] = df['description'].apply(extract_hashtags)
    return df


def get_best_time(df):
    best_hour = df.groupby('hour')['engagement_rate'].mean().idxmax()
    best_day = df.groupby('day_of_week')['engagement_rate'].mean().idxmax()
    return best_hour, best_day