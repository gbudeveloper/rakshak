from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)


# ============================================================
# RAKSHAK — Transformer Embedding Baseline v0.1
#
# Purpose:
#   Compare semantic sentence embeddings against the existing
#   TF-IDF + Logistic Regression baseline without fine-tuning
#   the transformer.
#
# Model:
#   sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
#
# Important:
#   - Uses the existing leakage-safe train/validation/test split.
#   - Validation selects the operating threshold.
#   - Test is evaluated only after the model is fitted.
#   - No OIL/private data is used.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "sif_splits_v0.1"
EXPERIMENT_DIR = (
    PROJECT_ROOT / "experiments" / "baseline_transformer_embedding_v0.1"
)

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
TEST_PATH = SPLIT_DIR / "test.csv"

MODEL_DIR = EXPERIMENT_DIR / "encoder"
CLASSIFIER_PATH = EXPERIMENT_DIR / "classifier.joblib"
CONFIG_PATH = EXPERIMENT_DIR / "config.json"
METRICS_PATH = EXPERIMENT_DIR / "metrics.json"
PREDICTIONS_PATH = EXPERIMENT_DIR / "test_predictions.csv"
VALIDATION_PREDICTIONS_PATH = EXPERIMENT_DIR / "validation_predictions.csv"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

RANDOM_STATE = 42
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 256

TARGET_MAP = {"NO": 0, "YES": 1}


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required split not found:\n{path}")

    df = pd.read_csv(path)

    required = {"report_id", "description", "final_sif_potential"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: "
            + ", ".join(sorted(missing))
        )

    labels = (
        df["final_sif_potential"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    unknown = sorted(set(labels.unique()) - set(TARGET_MAP))
    if unknown:
        raise ValueError(
            f"{path.name} contains unsupported labels: {unknown}"
        )

    if df["report_id"].duplicated().any():
        duplicates = df.loc[
            df["report_id"].duplicated(keep=False), "report_id"
        ].astype(str).tolist()
        raise ValueError(
            f"{path.name} contains duplicate report_id values: {duplicates}"
        )

    return df.copy()


def make_labels(df: pd.DataFrame) -> np.ndarray:
    return (
        df["final_sif_potential"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map(TARGET_MAP)
        .astype(int)
        .to_numpy()
    )


def encode_texts(
    encoder: SentenceTransformer,
    texts: list[str],
) -> np.ndarray:
    embeddings = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def best_f2_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    # Test a dense, deterministic threshold grid. With only a few
    # validation rows, avoid treating the exact validation optimum
    # as statistically meaningful.
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.95, 181),
                np.asarray(probabilities, dtype=float),
                np.asarray([0.50]),
            ]
        )
    )

    best_threshold = 0.50
    best_f2 = -1.0

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = fbeta_score(
            y_true,
            predictions,
            beta=2.0,
            zero_division=0,
        )

        if score > best_f2:
            best_f2 = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_f2


def evaluate_binary(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    result = {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f2": float(
            fbeta_score(
                y_true,
                predictions,
                beta=2.0,
                zero_division=0,
            )
        ),
        "confusion_matrix": cm.tolist(),
    }

    # Average precision is equivalent to the area under the
    # precision-recall curve and is useful for imbalanced data.
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:
        result["pr_auc"] = float(
            average_precision_score(y_true, probabilities)
        )
    else:
        result["pr_auc"] = None

    result["false_negative_count"] = int(
        ((y_true == 1) & (predictions == 0)).sum()
    )
    result["false_positive_count"] = int(
        ((y_true == 0) & (predictions == 1)).sum()
    )

    return result


def make_prediction_frame(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    output = df[
        [
            "report_id",
            "final_sif_potential",
            "description",
        ]
    ].copy()

    output["sif_probability"] = probabilities
    output["prediction_at_0.50"] = np.where(
        probabilities >= 0.50,
        "YES",
        "NO",
    )
    output["prediction_at_validation_threshold"] = np.where(
        probabilities >= threshold,
        "YES",
        "NO",
    )

    output["error_type_at_0.50"] = np.select(
        [
            (
                output["final_sif_potential"].eq("NO")
                & output["prediction_at_0.50"].eq("YES")
            ),
            (
                output["final_sif_potential"].eq("YES")
                & output["prediction_at_0.50"].eq("NO")
            ),
        ],
        [
            "FALSE_POSITIVE",
            "FALSE_NEGATIVE",
        ],
        default="CORRECT",
    )

    return output


def main() -> None:
    set_seed()

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split(TRAIN_PATH)
    validation = load_split(VALIDATION_PATH)
    test = load_split(TEST_PATH)

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    train_ids = set(train["report_id"].astype(str))
    validation_ids = set(validation["report_id"].astype(str))
    test_ids = set(test["report_id"].astype(str))

    if train_ids & validation_ids:
        raise RuntimeError("Train/validation report_id overlap detected.")
    if train_ids & test_ids:
        raise RuntimeError("Train/test report_id overlap detected.")
    if validation_ids & test_ids:
        raise RuntimeError("Validation/test report_id overlap detected.")

    print("=" * 78)
    print("RAKSHAK TRANSFORMER EMBEDDING BASELINE v0.1")
    print("=" * 78)
    print(f"Model       : {MODEL_NAME}")
    print(f"Device      : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Max tokens  : {MAX_SEQ_LENGTH}")
    print()
    print(
        f"Train       : {len(train)} "
        f"(YES={int((make_labels(train) == 1).sum())}, "
        f"NO={int((make_labels(train) == 0).sum())})"
    )
    print(
        f"Validation  : {len(validation)} "
        f"(YES={int((make_labels(validation) == 1).sum())}, "
        f"NO={int((make_labels(validation) == 0).sum())})"
    )
    print(
        f"Test        : {len(test)} "
        f"(YES={int((make_labels(test) == 1).sum())}, "
        f"NO={int((make_labels(test) == 0).sum())})"
    )

    # --------------------------------------------------------
    # Load encoder
    # --------------------------------------------------------

    print()
    print("-" * 78)
    print("LOADING PRETRAINED EMBEDDING MODEL")
    print("-" * 78)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )
    encoder.max_seq_length = MAX_SEQ_LENGTH

    print(f"Embedding dimension: {encoder.get_sentence_embedding_dimension()}")

    # --------------------------------------------------------
    # Encode all splits separately
    # --------------------------------------------------------

    print()
    print("-" * 78)
    print("ENCODING TRAIN")
    print("-" * 78)

    train_embeddings = encode_texts(
        encoder,
        train["description"].fillna("").astype(str).tolist(),
    )

    print()
    print("-" * 78)
    print("ENCODING VALIDATION")
    print("-" * 78)

    validation_embeddings = encode_texts(
        encoder,
        validation["description"].fillna("").astype(str).tolist(),
    )

    print()
    print("-" * 78)
    print("ENCODING TEST")
    print("-" * 78)

    test_embeddings = encode_texts(
        encoder,
        test["description"].fillna("").astype(str).tolist(),
    )

    np.save(EXPERIMENT_DIR / "train_embeddings.npy", train_embeddings)
    np.save(EXPERIMENT_DIR / "validation_embeddings.npy", validation_embeddings)
    np.save(EXPERIMENT_DIR / "test_embeddings.npy", test_embeddings)

    # --------------------------------------------------------
    # Train lightweight classifier
    # --------------------------------------------------------

    print()
    print("-" * 78)
    print("TRAINING LOGISTIC REGRESSION")
    print("-" * 78)

    y_train = make_labels(train)
    y_validation = make_labels(validation)
    y_test = make_labels(test)

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )

    classifier.fit(train_embeddings, y_train)

    # --------------------------------------------------------
    # Validation threshold selection
    # --------------------------------------------------------

    validation_probabilities = classifier.predict_proba(
        validation_embeddings
    )[:, 1]

    validation_threshold, validation_best_f2 = best_f2_threshold(
        y_validation,
        validation_probabilities,
    )

    validation_metrics = evaluate_binary(
        y_validation,
        validation_probabilities,
        validation_threshold,
    )

    validation_metrics["best_f2_from_threshold_search"] = float(
        validation_best_f2
    )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    test_probabilities = classifier.predict_proba(
        test_embeddings
    )[:, 1]

    test_metrics_at_050 = evaluate_binary(
        y_test,
        test_probabilities,
        0.50,
    )

    test_metrics_at_validation_threshold = evaluate_binary(
        y_test,
        test_probabilities,
        validation_threshold,
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    validation_predictions = make_prediction_frame(
        validation,
        validation_probabilities,
        validation_threshold,
    )

    test_predictions = make_prediction_frame(
        test,
        test_probabilities,
        validation_threshold,
    )

    validation_predictions.to_csv(
        VALIDATION_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8",
    )

    test_predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Save classifier + configuration
    # --------------------------------------------------------

    joblib.dump(
        classifier,
        CLASSIFIER_PATH,
    )

    config = {
        "experiment": "baseline_transformer_embedding_v0.1",
        "model_name": MODEL_NAME,
        "device": device,
        "batch_size": BATCH_SIZE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "embedding_dimension": int(
            encoder.get_sentence_embedding_dimension()
        ),
        "normalization": "L2-normalized sentence embeddings",
        "classifier": "LogisticRegression",
        "classifier_parameters": {
            "class_weight": "balanced",
            "max_iter": 2000,
            "solver": "liblinear",
            "random_state": RANDOM_STATE,
        },
        "splits": {
            "train": str(TRAIN_PATH.relative_to(PROJECT_ROOT)),
            "validation": str(
                VALIDATION_PATH.relative_to(PROJECT_ROOT)
            ),
            "test": str(TEST_PATH.relative_to(PROJECT_ROOT)),
        },
    }

    CONFIG_PATH.write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics = {
        "experiment": "baseline_transformer_embedding_v0.1",
        "model_name": MODEL_NAME,
        "train_records": int(len(train)),
        "validation_records": int(len(validation)),
        "test_records": int(len(test)),
        "validation_threshold": float(validation_threshold),
        "validation": validation_metrics,
        "test_at_0.50": test_metrics_at_050,
        "test_at_validation_threshold": (
            test_metrics_at_validation_threshold
        ),
        "interpretation": (
            "Diagnostic embedding baseline on a very small adjudicated "
            "dataset. Results are not a reliable estimate of production "
            "generalization."
        ),
    }

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("VALIDATION")
    print("=" * 78)
    print(
        f"Selected F2 threshold : {validation_threshold:.6f}"
    )
    print(
        f"Accuracy              : {validation_metrics['accuracy']:.6f}"
    )
    print(
        f"Precision             : {validation_metrics['precision']:.6f}"
    )
    print(
        f"Recall                : {validation_metrics['recall']:.6f}"
    )
    print(
        f"F2                    : {validation_metrics['f2']:.6f}"
    )
    print(
        f"PR-AUC                : "
        f"{validation_metrics['pr_auc']}"
    )

    print()
    print("=" * 78)
    print("TEST @ THRESHOLD 0.50")
    print("=" * 78)
    for key in [
        "accuracy",
        "precision",
        "recall",
        "f2",
        "pr_auc",
        "false_positive_count",
        "false_negative_count",
    ]:
        print(f"{key:<24}: {test_metrics_at_050[key]}")

    print()
    print("=" * 78)
    print("TEST @ VALIDATION-SELECTED THRESHOLD")
    print("=" * 78)
    for key in [
        "accuracy",
        "precision",
        "recall",
        "f2",
        "pr_auc",
        "false_positive_count",
        "false_negative_count",
    ]:
        print(
            f"{key:<24}: "
            f"{test_metrics_at_validation_threshold[key]}"
        )

    print()
    print("=" * 78)
    print("TEST PREDICTIONS")
    print("=" * 78)

    print(
        test_predictions[
            [
                "report_id",
                "final_sif_potential",
                "sif_probability",
                "prediction_at_0.50",
                "prediction_at_validation_threshold",
                "error_type_at_0.50",
            ]
        ]
        .sort_values(
            "sif_probability",
            ascending=False,
        )
        .to_string(index=False)
    )

    print()
    print("=" * 78)
    print("FILES SAVED")
    print("=" * 78)
    print(f"Classifier        : {CLASSIFIER_PATH}")
    print(f"Configuration     : {CONFIG_PATH}")
    print(f"Metrics           : {METRICS_PATH}")
    print(f"Validation preds  : {VALIDATION_PREDICTIONS_PATH}")
    print(f"Test predictions  : {PREDICTIONS_PATH}")
    print(f"Train embeddings  : {EXPERIMENT_DIR / 'train_embeddings.npy'}")
    print(
        f"Validation embed. : "
        f"{EXPERIMENT_DIR / 'validation_embeddings.npy'}"
    )
    print(f"Test embeddings   : {EXPERIMENT_DIR / 'test_embeddings.npy'}")

    print()
    print("=" * 78)
    print("EXPERIMENT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
