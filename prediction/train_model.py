"""Inspect the heart dataset, train a Random Forest, and save the model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "heart.csv"
MODEL_PATH = Path(__file__).resolve().parent / "heart_disease_model_deduplicated.joblib"
TARGET_COLUMN = "target"


def main():
    data = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column.")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")

    feature_columns = [column for column in data.columns if column != TARGET_COLUMN]
    label_counts = data.groupby(feature_columns, dropna=False)[TARGET_COLUMN].nunique(
        dropna=False
    )
    conflicting_features = label_counts[label_counts > 1].index
    if len(conflicting_features):
        feature_index = data.set_index(feature_columns).index
        conflicting_rows = data.loc[feature_index.isin(conflicting_features)]
        conflicts = (
            conflicting_rows.groupby(feature_columns, dropna=False)[TARGET_COLUMN]
            .agg(lambda labels: sorted(labels.unique()))
            .reset_index(name="conflicting_targets")
        )
        print("Conflicting target labels found for identical feature vectors:")
        print(conflicts.to_string(index=False))
        raise ValueError("Cannot train while identical features have conflicting labels.")

    original_row_count = len(data)
    data = data.drop_duplicates(subset=feature_columns, keep="first").copy()
    removed_duplicate_count = original_row_count - len(data)
    print(f"Original row count: {original_row_count}")
    print(f"Removed duplicate feature rows: {removed_duplicate_count}")
    print(f"Remaining row count: {len(data)}")

    X = data[feature_columns]
    y = data[TARGET_COLUMN]

    print("Dataset inspection")
    print(f"Rows after feature deduplication: {len(data)}")
    print("Columns and data types:")
    print(data.dtypes.to_string())
    print("Missing values by column:")
    print(data.isna().sum().to_string())
    print("Target distribution:")
    print(y.value_counts(dropna=False).sort_index().to_string())

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("\nEvaluation on stratified 20% test split")
    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}")
    print("Confusion matrix (labels [0, 1]):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
