"""Create a reproducible diabetes sample and train a small grouped forest."""

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
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_PATH = BASE_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"
SAMPLED_DATA_PATH = Path(__file__).resolve().parent / "diabetes_sample_12000.csv"
MODEL_PATH = Path(__file__).resolve().parent / "diabetes_model_12000_tmp.joblib"
TARGET_COLUMN = "Diabetes_binary"
SAMPLE_SIZE = 12000


def tree_array_memory(model):
    return sum(
        array.nbytes
        for estimator in model.estimators_
        for array in estimator.tree_.__getstate__().values()
        if hasattr(array, "nbytes")
    )


def main():
    data = pd.read_csv(SOURCE_PATH)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column.")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")
    if len(data) < SAMPLE_SIZE:
        raise ValueError(f"Dataset must contain at least {SAMPLE_SIZE} rows.")

    original_row_count = len(data)
    sampled_data, _ = train_test_split(
        data,
        train_size=SAMPLE_SIZE,
        random_state=42,
        stratify=data[TARGET_COLUMN],
    )
    sampled_row_count = len(sampled_data)
    sampled_data = sampled_data.drop_duplicates(keep="first").copy()
    duplicate_rows_removed = sampled_row_count - len(sampled_data)
    sampled_data.to_csv(SAMPLED_DATA_PATH, index=False)

    feature_columns = [column for column in sampled_data.columns if column != TARGET_COLUMN]
    X = sampled_data[feature_columns]
    y = sampled_data[TARGET_COLUMN]
    grouped = sampled_data.groupby(feature_columns, dropna=False, sort=False)
    groups = grouped.ngroup()
    unique_feature_groups = int(groups.nunique())
    conflicting_feature_groups = int(
        (grouped[TARGET_COLUMN].nunique(dropna=False) > 1).sum()
    )

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, test_indices = next(splitter.split(X, y, groups))
    x_train, x_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])
    overlapping_groups = train_groups.intersection(test_groups)

    model = RandomForestClassifier(
        n_estimators=30,
        max_depth=12,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    joblib.dump(model, MODEL_PATH, compress=3)
    reloaded_model = joblib.load(MODEL_PATH)
    reload_predictions = reloaded_model.predict(x_test)
    if not (predictions == reload_predictions).all():
        raise RuntimeError("Saved model predictions do not match the trained model.")

    print(f"Original row count: {original_row_count}")
    print(f"Stratified sample size before duplicate removal: {sampled_row_count}")
    print(f"Exact duplicate rows removed: {duplicate_rows_removed}")
    print(f"Final sampled row count: {len(sampled_data)}")
    print("Class distribution:")
    print(y.value_counts().sort_index().to_string())
    print(f"Unique feature groups: {unique_feature_groups}")
    print(f"Conflicting feature groups: {conflicting_feature_groups}")
    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")
    print(f"Train/test feature groups overlap: {bool(overlapping_groups)}")
    print(f"Overlapping feature group count: {len(overlapping_groups)}")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}")
    print("Confusion matrix (labels [0, 1]):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(f"Model file size: {MODEL_PATH.stat().st_size:,} bytes")
    memory_bytes = tree_array_memory(reloaded_model)
    print(f"Tree-array memory: {memory_bytes:,} bytes ({memory_bytes / (1024 ** 2):.2f} MiB)")
    print(f"Reloaded model type: {type(reloaded_model).__name__}")
    print(f"Reloaded predictions match: {(predictions == reload_predictions).all()}")
    print(f"Saved sample dataset to: {SAMPLED_DATA_PATH}")
    print(f"Saved temporary model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
