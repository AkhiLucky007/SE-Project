from datasets import load_dataset
import pandas as pd
import joblib

# -----------------------------
# 📥 Load dataset (HuggingFace)
# -----------------------------
print("🚀 Loading dataset...")

dataset = load_dataset("vargr/main_instagram")
df = pd.DataFrame(dataset["train"])

# -----------------------------
# 🔍 Check column
# -----------------------------
if "description" not in df.columns:
    raise ValueError("❌ 'description' column not found")

print("✅ Dataset loaded. Rows:", len(df))

# -----------------------------
# 📦 Load saved TF-IDF
# -----------------------------
print("🚀 Loading TF-IDF vectorizer...")

tfidf = joblib.load("src/tfidf_vectorizer.pkl")

# -----------------------------
# 🔄 Recreate TF-IDF matrix
# -----------------------------
print("🚀 Creating TF-IDF matrix...")

X_tfidf = tfidf.transform(df["description"]).toarray()

# -----------------------------
# 💾 Save files
# -----------------------------
print("🚀 Saving files...")

joblib.dump(df["description"].tolist(), "src/caption_corpus.pkl")
joblib.dump(X_tfidf, "src/tfidf_matrix.pkl")

print("✅ DONE — retrieval system ready!")