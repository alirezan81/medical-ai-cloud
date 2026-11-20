"""Reproducibly train and evaluate the final 20,000-row diabetes model."""

from datetime import datetime
from hashlib import sha256
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
PREDICTION_DIR = Path(__file__).resolve().parent
SOURCE_PATH = BASE_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"
ACTIVE_MODEL_PATH = PREDICTION_DIR / "diabetes_model.joblib"
TARGET_COLUMN = "Diabetes_binary"
SAMPLE_CLASS_COUNTS = {0: 12000, 1: 8000}
TRAIN_CLASS_COUNTS = {0: 9600, 1: 6400}
COMPARISON_CLASS_COUNTS = {0: 9000, 1: 6000}
TEST_CLASS_COUNTS = {0: 2400, 1: 1600}


def new_output_paths():
    """Choose fresh timestamped artifact paths without replacing prior outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = timestamp
    sequence = 1
    while (
        (PREDICTION_DIR / f"diabetes_sample_20000_{suffix}.csv").exists()
        or (PREDICTION_DIR / f"diabetes_model_20000_tmp_{suffix}.joblib").exists()
    ):
        suffix = f"{timestamp}_{sequence}"
        sequence += 1
    return (
        PREDICTION_DIR / f"diabetes_sample_20000_{suffix}.csv",
        PREDICTION_DIR / f"diabetes_model_20000_tmp_{suffix}.joblib",
    )


def file_sha256(path):
    digest = sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_model():
    return RandomForestClassifier(
        n_estimators=30,
        max_depth=12,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
    )


def tree_array_memory(model):
    return sum(
        array.nbytes
        for estimator in model.estimators_
        for array in estimator.tree_.__getstate__().values()
        if hasattr(array, "nbytes")
    )


def report_metrics(name, y_true, predictions):
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    correct = int((y_true.to_numpy() == predictions).sum())
    total = len(y_true)
    incorrect = total - correct
    tn, fp, fn, tp = matrix.ravel()
    print(f"{name} results:")
    print(f"  Accuracy:  {accuracy_score(y_true, predictions):.4f}")
    print(f"  Precision: {precision_score(y_true, predictions, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_true, predictions, zero_division=0):.4f}")
    print(f"  F1-score:  {f1_score(y_true, predictions, zero_division=0):.4f}")
    print(f"  Confusion matrix [0, 1]: {matrix.tolist()}")
    print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  Correct: {correct}/{total} ({correct / total * 100:.2f}%)")
    print(f"  Incorrect: {incorrect}/{total} ({incorrect / total * 100:.2f}%)")


def main():
    sample_path, candidate_model_path = new_output_paths()
    active_hash_before = file_sha256(ACTIVE_MODEL_PATH)
    data = pd.read_csv(SOURCE_PATH)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column.")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")

    original_columns = list(data.columns)
    feature_columns = [column for column in original_columns if column != TARGET_COLUMN]
    if len(feature_columns) != 21:
        raise ValueError(f"Expected 21 feature columns, found {len(feature_columns)}.")

    all_feature_groups = data.groupby(feature_columns, dropna=False, sort=False)
    conflicting_labels = all_feature_groups[TARGET_COLUMN].nunique(dropna=False)
    conflicting_keys = conflicting_labels[conflicting_labels > 1].index
    feature_index = data.set_index(feature_columns).index
    eligible_rows = data.loc[~feature_index.isin(conflicting_keys)]
    clean_unique = eligible_rows.drop_duplicates(subset=feature_columns, keep="first")
    eligible_counts = clean_unique[TARGET_COLUMN].value_counts().to_dict()
    if (
        eligible_counts.get(0, 0) < SAMPLE_CLASS_COUNTS[0]
        or eligible_counts.get(1, 0) < SAMPLE_CLASS_COUNTS[1]
    ):
        raise ValueError(f"Not enough eligible unique feature vectors: {eligible_counts}")

    # Preserve the established workflow: sample each class independently,
    # then shuffle the combined 60/40 sample, all with random_state=42.
    sample_20000 = pd.concat(
        [
            clean_unique[clean_unique[TARGET_COLUMN] == 0].sample(
                n=SAMPLE_CLASS_COUNTS[0], random_state=42
            ),
            clean_unique[clean_unique[TARGET_COLUMN] == 1].sample(
                n=SAMPLE_CLASS_COUNTS[1], random_state=42
            ),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    if len(sample_20000) != 20000:
        raise RuntimeError(f"Expected 20,000 rows, got {len(sample_20000)}.")
    if sample_20000.duplicated(subset=feature_columns).any():
        raise RuntimeError("The sample contains duplicate feature vectors.")
    if list(sample_20000.columns) != original_columns:
        raise RuntimeError("The sample columns/order differ from the source CSV.")
    sample_class_counts = sample_20000[TARGET_COLUMN].value_counts().to_dict()
    if sample_class_counts != SAMPLE_CLASS_COUNTS:
        raise RuntimeError(f"Unexpected sample class distribution: {sample_class_counts}")
    sample_20000.to_csv(sample_path, index=False)

    X = sample_20000[feature_columns]
    y = sample_20000[TARGET_COLUMN]
    groups = sample_20000.groupby(feature_columns, dropna=False, sort=False).ngroup()
    train_indices, test_indices = train_test_split(
        sample_20000.index.to_numpy(),
        test_size=4000,
        random_state=42,
        stratify=y,
    )
    x_train_20000, x_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train_20000, y_test = y.iloc[train_indices], y.iloc[test_indices]
    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])
    overlap_20000 = train_groups.intersection(test_groups)
    if overlap_20000:
        raise RuntimeError(f"20k train/test feature groups overlap: {len(overlap_20000)}")
    if y_test.value_counts().to_dict() != TEST_CLASS_COUNTS:
        raise RuntimeError(f"Unexpected test distribution: {y_test.value_counts().to_dict()}")
    if y_train_20000.value_counts().to_dict() != TRAIN_CLASS_COUNTS:
        raise RuntimeError(
            f"Unexpected 20k training distribution: {y_train_20000.value_counts().to_dict()}"
        )

    # Retain the established fair comparison: train a 15k model from the same
    # training pool and score it against the identical 4k holdout.
    train_pool = sample_20000.iloc[train_indices]
    train_15000 = pd.concat(
        [
            train_pool[train_pool[TARGET_COLUMN] == 0].sample(n=9000, random_state=42),
            train_pool[train_pool[TARGET_COLUMN] == 1].sample(n=6000, random_state=42),
        ]
    ).sample(frac=1, random_state=42)
    if train_15000[TARGET_COLUMN].value_counts().to_dict() != COMPARISON_CLASS_COUNTS:
        raise RuntimeError("The 15,000-row comparison training subset is not 60/40.")
    comparison_groups = set(groups.iloc[train_15000.index])
    overlap_15000 = comparison_groups.intersection(test_groups)
    if overlap_15000:
        raise RuntimeError(f"15k train/test feature groups overlap: {len(overlap_15000)}")

    model_20000 = make_model()
    model_20000.fit(x_train_20000, y_train_20000)
    predictions_20000 = model_20000.predict(x_test)
    joblib.dump(model_20000, candidate_model_path, compress=3)
    loaded_candidate = joblib.load(candidate_model_path)
    if not isinstance(loaded_candidate, RandomForestClassifier):
        raise RuntimeError("Reloaded candidate is not a RandomForestClassifier.")
    if loaded_candidate.n_features_in_ != len(feature_columns):
        raise RuntimeError(f"Candidate expects {loaded_candidate.n_features_in_} features.")
    candidate_predictions = loaded_candidate.predict(x_test)
    if not (predictions_20000 == candidate_predictions).all():
        raise RuntimeError("Reloaded 20k candidate predictions do not match.")

    model_15000 = make_model()
    model_15000.fit(train_15000[feature_columns], train_15000[TARGET_COLUMN])
    predictions_15000 = model_15000.predict(x_test)

    active_hash_after = file_sha256(ACTIVE_MODEL_PATH)
    if active_hash_before != active_hash_after:
        raise RuntimeError("The active diabetes model changed during the training run.")

    print("=== Diabetes model training report ===")
    print(f"Original dataset dimensions: {data.shape[0]} rows x {data.shape[1]} columns")
    print(f"Conflicting feature groups excluded: {len(conflicting_keys):,}")
    print(f"Eligible unique rows by class: {eligible_counts}")
    print(
        "Sampling methodology: exclude conflicting groups, retain one row per feature vector, "
        "sample 12,000 class-0 and 8,000 class-1 rows (random_state=42), then shuffle (42)."
    )
    print("Preprocessing: original feature values; no scaling or normalization.")
    print(f"Final sample dimensions: {sample_20000.shape[0]} rows x {sample_20000.shape[1]} columns")
    print(f"Final sample class distribution: {sample_class_counts}")
    print("Class percentages: class 0=60.00%, class 1=40.00%")
    print(f"20,000-row training class counts: {y_train_20000.value_counts().sort_index().to_dict()}")
    print(f"15,000-row comparison training class counts: {train_15000[TARGET_COLUMN].value_counts().sort_index().to_dict()}")
    print(f"Common test class counts: {y_test.value_counts().sort_index().to_dict()}")
    print(
        f"Train/test sizes: 20k model={len(x_train_20000)}/{len(x_test)}; "
        f"15k comparison={len(train_15000)}/{len(x_test)}"
    )
    print(f"Feature-group overlap: 20k train/test={len(overlap_20000)}; 15k train/test={len(overlap_15000)}")
    print("Random Forest configuration: n_estimators=30, max_depth=12, min_samples_leaf=10,")
    print("  max_features='sqrt', class_weight='balanced', random_state=42")
    report_metrics("20,000-row model", y_test, predictions_20000)
    report_metrics("15,000-row comparison on the same test set", y_test, predictions_15000)
    print(f"Candidate model file size: {candidate_model_path.stat().st_size:,} bytes")
    memory_bytes = tree_array_memory(loaded_candidate)
    print(f"Candidate tree-array memory: {memory_bytes:,} bytes ({memory_bytes / (1024 ** 2):.2f} MiB)")
    print(f"Reloaded model type: {type(loaded_candidate).__name__}")
    print(f"Reloaded model expects {loaded_candidate.n_features_in_} features: verified")
    print(f"Reloaded predictions match original fitted model: {(predictions_20000 == candidate_predictions).all()}")
    print(f"Active model SHA-256 before: {active_hash_before}")
    print(f"Active model SHA-256 after:  {active_hash_after}")
    print(f"Generated sample path: {sample_path}")
    print(f"Generated candidate model path: {candidate_model_path}")


if __name__ == "__main__":
    main()
