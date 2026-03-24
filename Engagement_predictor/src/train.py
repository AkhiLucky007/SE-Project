from datasets import load_dataset # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import classification_report # type: ignore

from preprocessing import load_data, create_classes
from feature_engineering import add_time_features, log_transform
from model import EngagementModel

def main():
    dataset = load_dataset("vargr/main_instagram")
    df = dataset['train'].to_pandas()

    df = load_data(df)
    df = create_classes(df)
    df = add_time_features(df)
    df = log_transform(df)

    df = df.sample(n=100000, random_state=42)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    model = EngagementModel()
    model.fit(train_df)

    preds = model.predict(test_df)

    print(classification_report(test_df['engagement_class'], preds))


if __name__ == "__main__":
    main()