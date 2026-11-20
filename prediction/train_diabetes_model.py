"""Build and evaluate the reproducible, balanced 15,000-row diabetes model."""

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
from sklearn.model_selection import StratifiedGroupKFold


BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_PATH = BASE_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"
SAMPLED_DATA_PATH = Path(__file__).resolve().parent / "diabetes_sample_15000.csv"
MODEL_PATH = Path(__file__).resolve().parent / "diabetes_model_15000_tmp.joblib"
TARGET_COLUMN = "Diabetes_binary"
CLASS_COUNTS = {0: 9000, 1: 6000}


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

    original_row_count = len(data)
    feature_columns = [column for column in data.columns if column != TARGET_COLUMN]
    all_groups = data.groupby(feature_columns, dropna=False, sort=False)
    conflicting_features = all_groups[TARGET_COLUMN].nunique(dropna=False)
    conflicting_keys = conflicting_features[conflicting_features > 1].index

    feature_index = data.set_index(feature_columns).index
    unique_rows = data.loc[~feature_index.isin(conflicting_keys)].drop_duplicates(
        subset=feature_columns,
        keep="first",
    )
    class_0 = unique_rows[unique_rows[TARGET_COLUMN] == 0]
    class_1 = unique_rows[unique_rows[TARGET_COLUMN] == 1]
    if len(class_0) < CLASS_COUNTS[0] or len(class_1) < CLASS_COUNTS[1]:
        raise ValueError("Not enough unique, non-conflicting records for requested class counts.")

    sampled_data = pd.concat(
        [
            class_0.sample(n=CLASS_COUNTS[0], random_state=42),
            class_1.sample(n=CLASS_COUNTS[1], random_state=42),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    if len(sampled_data) != 15000:
        raise ValueError(f"Expected 15,000 sample rows, got {len(sampled_data)}.")
    if sampled_data.duplicated(subset=feature_columns).any():
        raise ValueError("Sample contains duplicate feature vectors.")
    class_counts = sampled_data[TARGET_COLUMN].value_counts().to_dict()
    if class_counts != CLASS_COUNTS:
        raise ValueError(f"Unexpected sample class distribution: {class_counts}")
    if list(sampled_data.columns) != list(data.columns):
        raise ValueError("Sample columns do not match the original dataset.")

    sampled_data.to_csv(SAMPLED_DATA_PATH, index=False)
    X = sampled_data[feature_columns]
    y = sampled_data[TARGET_COLUMN]
    groups = sampled_data.groupby(feature_columns, dropna=False, sort=False).ngroup()

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, test_indices = next(splitter.split(X, y, groups))
    x_train, x_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])
    overlapping_groups = train_groups.intersection(test_groups)
    if overlapping_groups:
        raise RuntimeError("Train/test feature groups overlap.")

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
        raise RuntimeError("Saved candidate predictions do not match the trained model.")

    print(f"Original row count: {original_row_count}")
    print(f"Conflicting feature groups excluded: {len(conflicting_keys)}")
    print(f"Sample rows: {len(sampled_data)}")
    print(f"Unique sample feature vectors: {sampled_data[feature_columns].drop_duplicates().shape[0]}")
    print(f"Class 0: {class_counts[0]} ({class_counts[0] / len(sampled_data) * 100:.2f}%)")
    print(f"Class 1: {class_counts[1]} ({class_counts[1] / len(sampled_data) * 100:.2f}%)")
    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")
    print(f"Train/test feature-group overlap: {len(overlapping_groups)}")
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
    print(f"Saved sample to: {SAMPLED_DATA_PATH}")
    print(f"Saved temporary model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
