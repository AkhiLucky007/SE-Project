from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from xgboost import XGBClassifier # type: ignore
import scipy.sparse as sp # type: ignore

class EngagementModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1,2),
            min_df=5
        )
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
        X_text = self.vectorizer.fit_transform(df['description'])

        meta = df[['hour','day_of_week','followers',
                   'following','num_posts','is_business_account']].copy()

        meta['is_business_account'] = meta['is_business_account'].astype(int)
        meta = meta.astype(float).values

        X = sp.hstack([X_text, meta])
        y = df['engagement_class']

        self.model.fit(X, y)

    def predict(self, df):
        X_text = self.vectorizer.transform(df['description'])

        meta = df[['hour','day_of_week','followers',
                   'following','num_posts','is_business_account']].copy()

        meta['is_business_account'] = meta['is_business_account'].astype(int)
        meta = meta.astype(float).values

        X = sp.hstack([X_text, meta])

        return self.model.predict(X)