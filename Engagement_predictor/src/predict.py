import pandas as pd

def predict_single(model, caption, metadata):
    df = pd.DataFrame([{
        "description": caption,
        "hour": metadata['hour'],
        "day_of_week": metadata['day'],
        "followers": metadata['followers'],
        "following": metadata['following'],
        "num_posts": metadata['num_posts'],
        "is_business_account": metadata['business']
    }])

    return model.predict(df)[0]