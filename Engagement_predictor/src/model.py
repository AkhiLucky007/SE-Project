from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from xgboost import XGBClassifier  # type: ignore
import scipy.sparse as sp  # type: ignore
import joblib


class EngagementModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            min_df=5
        )

        self.meta_columns = [
            'hour',
            'day_of_week',
            'followers',
            'following',
            'num_posts',
            'is_business_account',
            'caption_length',
            'num_hashtags',
            'time_bucket'
        ]

        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            tree_method='hist'
        )

    def fit(self, df):
        # Text features
        X_text = self.vectorizer.fit_transform(df['description'])

        # Metadata features
        meta = df[self.meta_columns].copy()
        meta['is_business_account'] = meta['is_business_account'].astype(int)
        meta = meta.astype(float).values

        # Combine
        X = sp.hstack([X_text, meta])

        y = df['engagement_class']

        self.model.fit(X, y)

    def predict(self, df):
        # Text features
        X_text = self.vectorizer.transform(df['description'])

        # Metadata features
        meta = df[self.meta_columns].copy()
        meta['is_business_account'] = meta['is_business_account'].astype(int)
        meta = meta.astype(float).values

        # Combine
        X = sp.hstack([X_text, meta])

        return self.model.predict(X)

    def show_feature_importance(self, top_n=20):
        """
        Show top important features from the trained model
        """

        # TF‑IDF feature names
        text_features = self.vectorizer.get_feature_names_out()

        # Metadata feature names
        meta_features = self.meta_columns

        # Combine feature names
        all_features = list(text_features) + meta_features

        # Get importance scores
        importances = self.model.feature_importances_

        # Pair and sort
        feature_scores = list(zip(all_features, importances))
        feature_scores.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop {top_n} Important Features:\n")

        for feature, score in feature_scores[:top_n]:
            print(f"{feature}: {score:.5f}")

    def save_model(self, path):
        joblib.dump({
            "model": self.model,
            "vectorizer": self.vectorizer,
            "meta_columns": self.meta_columns
        }, path)


    def load_model(self, path):
        data = joblib.load(path)
        self.model = data["model"]
        self.vectorizer = data["vectorizer"]
        self.meta_columns = data["meta_columns"]