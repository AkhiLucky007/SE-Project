import pandas as pd
import numpy as np

def load_data(df):
    df = df.copy()

    df = df[df['followers'] > 0]

    df['engagement_rate'] = (df['likes'] + df['comments']) / df['followers']

    upper = df['engagement_rate'].quantile(0.99)
    df = df[df['engagement_rate'] <= upper]

    return df


def create_classes(df):
    low = df['engagement_rate'].quantile(0.33)
    high = df['engagement_rate'].quantile(0.66)

    def classify(x):
        if x <= low:
            return 0
        elif x <= high:
            return 1
        else:
            return 2

    df['engagement_class'] = df['engagement_rate'].apply(classify)
    return df