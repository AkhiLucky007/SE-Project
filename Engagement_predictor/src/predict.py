import numpy as np
import pandas as pd
import joblib
import os
from transformers import DistilBertTokenizer, DistilBertModel
import torch



class Predictor:

    def __init__(self):
        # Load model
        model_path = os.path.join(os.path.dirname(__file__), "engagement_model.pkl")
        self.model = joblib.load(model_path)

        # Load BERT
        self.device = torch.device("cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.bert.eval()

        # TF-IDF (⚠️ IMPORTANT: must match training)
        tfidf_path = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")
        self.tfidf = joblib.load(tfidf_path)

    # -----------------------------
    # BERT embeddings
    # -----------------------------
    def get_bert_embedding(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.bert(**inputs)

        return outputs.last_hidden_state[:, 0, :].numpy()

    # -----------------------------
    # Predict
    # -----------------------------
    def predict(self, df):
        text = df['description'].astype(str).tolist()

        # BERT
        X_bert = np.vstack([self.get_bert_embedding(t) for t in text])

        # TF-IDF (fit on input — not perfect but works for now)
        X_tfidf = self.tfidf.transform(text).toarray()

        # Metadata
        meta_cols = ['hour','day_of_week','followers',
                     'following','num_posts','is_business_account']

        X_meta = df[meta_cols].astype(float).values

        # Combine
        X = np.hstack([X_bert, X_tfidf, X_meta])

        return self.model.predict(X), self.model.predict_proba(X)