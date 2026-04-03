from datasets import load_dataset  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import classification_report  # type: ignore

from preprocessing import load_data, create_classes
from feature_engineering import add_time_features, log_transform
from model import EngagementModel
from feature_engineering import add_time_features, log_transform, add_time_bucket

def main():

    print(" Loading dataset...")
    dataset = load_dataset("vargr/main_instagram")
    df = dataset['train'].to_pandas()

    print(" Running preprocessing...")
    df = load_data(df)
    df = create_classes(df)
    df = add_time_features(df)
    df = log_transform(df)
    df = add_time_bucket(df)

    #  NEW FEATURE ENGINEERING (important accuracy boost)
    print(" Creating caption features...")

    df["caption_length"] = df["description"].astype(str).apply(len)
    df["num_hashtags"] = df["description"].astype(str).str.count("#")

    print(" Splitting dataset...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["engagement_class"]   # keeps class balance
    )

    print(" Training model...")
    model = EngagementModel()
    model.fit(train_df)

    print(" Evaluating model...")
    preds = model.predict(test_df)

    print("\n Classification Report:\n")
    print(classification_report(test_df["engagement_class"], preds))

    print("\n Feature Importance:\n")
    model.show_feature_importance(top_n=20)

    print("\n Saving trained model...")
    model.save_model("engagement_model.pkl")

    print("\n Training complete!")


if __name__ == "__main__":
    main()