from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESOLVED_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "resolved_annotations_v0.1.csv"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sif_splits_v0.1"
)

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
TEST_PATH = SPLIT_DIR / "test.csv"

EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "hybrid_tfidf_transformer_v0.1"
)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 256

TFIDF_MAX_FEATURES = 10000


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
    )


def resolve_label_column(df: pd.DataFrame) -> str:
    candidates = [
        "final_sif_potential",
        "sif_potential",
        "resolved_sif_potential",
    ]
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        f"No binary SIF label column found. Columns: {list(df.columns)}"
    )


def load_resolved() -> pd.DataFrame:
    if not RESOLVED_PATH.exists():
        raise FileNotFoundError(f"Missing:\n{RESOLVED_PATH}")

    df = pd.read_csv(RESOLVED_PATH)
    label_column = resolve_label_column(df)

    if "description" not in df.columns:
        raise ValueError(
            f"'description' not found in {RESOLVED_PATH.name}"
        )

    out = df[
        [
            "description",
            label_column,
        ]
    ].copy()

    out = out.rename(columns={label_column: "label"})
    out["description"] = out["description"].fillna("").astype(str)
    out["label"] = normalize_label(out["label"])

    out = out[out["label"].isin(["YES", "NO"])].reset_index(drop=True)

    if len(out) < 10:
        raise ValueError(
            f"Only {len(out)} binary records available."
        )

    return out


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing:\n{path}")

    df = pd.read_csv(path)

    required = {
        "report_id",
        "description",
        "final_sif_potential",
    }
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} missing columns: {sorted(missing)}"
        )

    df = df.copy()
    df["final_sif_potential"] = normalize_label(
        df["final_sif_potential"]
    )
    df["description"] = (
        df["description"].fillna("").astype(str)
    )

    return df


def y_values(series: pd.Series) -> np.ndarray:
    return normalize_label(series).map(
        {"NO": 0, "YES": 1}
    ).astype(int).to_numpy()


def metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.50,
) -> dict:
    predictions = (probabilities >= threshold).astype(int)

    return {
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
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "false_positive_count": int(
            ((y_true == 0) & (predictions == 1)).sum()
        ),
        "false_negative_count": int(
            ((y_true == 1) & (predictions == 0)).sum()
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }


def summary_stats(
    results_df: pd.DataFrame,
    model: str,
) -> dict:
    subset = results_df[
        results_df["model"].eq(model)
    ]

    result = {
        "model": model,
        "folds": int(len(subset)),
        "metrics": {},
    }

    for name in [
        "accuracy",
        "precision",
        "recall",
        "f2",
        "pr_auc",
        "false_positive_count",
        "false_negative_count",
    ]:
        values = subset[name].astype(float)
        result["metrics"][name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    return result


def main() -> None:
    set_seed()

    EXPERIMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved = load_resolved()

    texts = resolved["description"].tolist()
    y = y_values(resolved["label"])

    print("=" * 78)
    print("RAKSHAK HYBRID SIF MODEL — TF-IDF + TRANSFORMER EMBEDDINGS v0.1")
    print("=" * 78)
    print(f"Records     : {len(resolved)}")
    print(f"YES         : {int((y == 1).sum())}")
    print(f"NO          : {int((y == 0).sum())}")
    print(f"CV          : {N_SPLITS}-fold × {N_REPEATS} repeats")
    print(f"Total folds : {N_SPLITS * N_REPEATS}")
    print()
    print(
        "Hybrid features = TF-IDF word unigrams/bigrams "
        "+ frozen transformer sentence embeddings."
    )

    # --------------------------------------------------------
    # Frozen transformer encoding
    # --------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print()
    print("-" * 78)
    print("LOADING TRANSFORMER")
    print("-" * 78)
    print(f"Model : {MODEL_NAME}")
    print(f"Device: {device}")

    encoder = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )
    encoder.max_seq_length = MAX_SEQ_LENGTH

    embeddings = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    embedding_dim = embeddings.shape[1]

    print(f"Embedding dimension: {embedding_dim}")

    # --------------------------------------------------------
    # Repeated CV
    # --------------------------------------------------------

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    results = []

    print()
    print("-" * 78)
    print("RUNNING HYBRID CROSS-VALIDATION")
    print("-" * 78)

    for fold_number, (train_idx, test_idx) in enumerate(
        splitter.split(texts, y),
        start=1,
    ):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # Fit TF-IDF only on the fold's training narratives.
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            sublinear_tf=True,
            max_features=TFIDF_MAX_FEATURES,
        )

        X_train_tfidf = vectorizer.fit_transform(
            train_texts
        )
        X_test_tfidf = vectorizer.transform(
            test_texts
        )

        # Frozen embeddings are precomputed before CV because the
        # encoder itself is not trained on the SIF labels.
        X_train_embedding = csr_matrix(
            embeddings[train_idx]
        )
        X_test_embedding = csr_matrix(
            embeddings[test_idx]
        )

        X_train_hybrid = hstack(
            [
                X_train_tfidf,
                X_train_embedding,
            ],
            format="csr",
        )

        X_test_hybrid = hstack(
            [
                X_test_tfidf,
                X_test_embedding,
            ],
            format="csr",
        )

        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )

        classifier.fit(
            X_train_hybrid,
            y_train,
        )

        probabilities = classifier.predict_proba(
            X_test_hybrid
        )[:, 1]

        fold_metrics = metrics(
            y_test,
            probabilities,
            threshold=0.50,
        )

        results.append(
            {
                "fold": fold_number,
                "model": "Hybrid TF-IDF + Transformer",
                **fold_metrics,
            }
        )

        print(
            f"Fold {fold_number:02d}/"
            f"{N_SPLITS * N_REPEATS}: "
            f"F2={fold_metrics['f2']:.3f}, "
            f"PR-AUC={fold_metrics['pr_auc']:.3f}, "
            f"Recall={fold_metrics['recall']:.3f}"
        )

    results_df = pd.DataFrame(results)

    fold_results_path = (
        EXPERIMENT_DIR / "hybrid_cv_fold_results.csv"
    )

    results_df.to_csv(
        fold_results_path,
        index=False,
        encoding="utf-8",
    )

    cv_summary = summary_stats(
        results_df,
        "Hybrid TF-IDF + Transformer",
    )

    # --------------------------------------------------------
    # Locked split evaluation
    #
    # IMPORTANT: the 6-record test split remains untouched by
    # the CV above. We fit a fresh hybrid classifier only on
    # the existing 28-record train split and select threshold
    # from the existing 6-record validation split.
    # --------------------------------------------------------

    train = load_split(TRAIN_PATH)
    validation = load_split(VALIDATION_PATH)
    test = load_split(TEST_PATH)

    train_texts = train["description"].tolist()
    validation_texts = validation["description"].tolist()
    test_texts = test["description"].tolist()

    y_train = y_values(train["final_sif_potential"])
    y_validation = y_values(
        validation["final_sif_potential"]
    )
    y_test = y_values(test["final_sif_potential"])

    # Encode only the locked split narratives with the same
    # frozen encoder.
    train_emb = encoder.encode(
        train_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    validation_emb = encoder.encode(
        validation_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    test_emb = encoder.encode(
        test_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    locked_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
        max_features=TFIDF_MAX_FEATURES,
    )

    train_tfidf = locked_vectorizer.fit_transform(
        train_texts
    )
    validation_tfidf = locked_vectorizer.transform(
        validation_texts
    )
    test_tfidf = locked_vectorizer.transform(
        test_texts
    )

    train_hybrid = hstack(
        [
            train_tfidf,
            csr_matrix(train_emb),
        ],
        format="csr",
    )

    validation_hybrid = hstack(
        [
            validation_tfidf,
            csr_matrix(validation_emb),
        ],
        format="csr",
    )

    test_hybrid = hstack(
        [
            test_tfidf,
            csr_matrix(test_emb),
        ],
        format="csr",
    )

    locked_classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )

    locked_classifier.fit(
        train_hybrid,
        y_train,
    )

    validation_probabilities = (
        locked_classifier.predict_proba(
            validation_hybrid
        )[:, 1]
    )

    # Select validation threshold for F2. We keep this deterministic
    # and separate from the final test evaluation.
    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.95, 181),
                validation_probabilities,
                np.asarray([0.50]),
            ]
        )
    )

    best_threshold = 0.50
    best_validation_f2 = -1.0

    for threshold in candidate_thresholds:
        predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        score = fbeta_score(
            y_validation,
            predictions,
            beta=2.0,
            zero_division=0,
        )

        if score > best_validation_f2:
            best_validation_f2 = float(score)
            best_threshold = float(threshold)

    test_probabilities = (
        locked_classifier.predict_proba(
            test_hybrid
        )[:, 1]
    )

    test_metrics_050 = metrics(
        y_test,
        test_probabilities,
        threshold=0.50,
    )

    test_metrics_selected = metrics(
        y_test,
        test_probabilities,
        threshold=best_threshold,
    )

    predictions = test[
        [
            "report_id",
            "description",
            "final_sif_potential",
        ]
    ].copy()

    predictions["sif_probability"] = test_probabilities
    predictions["prediction_at_0.50"] = np.where(
        test_probabilities >= 0.50,
        "YES",
        "NO",
    )
    predictions[
        "prediction_at_validation_threshold"
    ] = np.where(
        test_probabilities >= best_threshold,
        "YES",
        "NO",
    )

    predictions.to_csv(
        EXPERIMENT_DIR / "locked_test_predictions.csv",
        index=False,
        encoding="utf-8",
    )

    # Save locked artifacts.
    joblib.dump(
        locked_vectorizer,
        EXPERIMENT_DIR / "tfidf_vectorizer.joblib",
    )
    joblib.dump(
        locked_classifier,
        EXPERIMENT_DIR / "classifier.joblib",
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {
        "experiment": "hybrid_tfidf_transformer_v0.1",
        "transformer_model": MODEL_NAME,
        "device": device,
        "embedding_dimension": int(embedding_dim),
        "max_seq_length": MAX_SEQ_LENGTH,
        "cv": {
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "folds_per_model": N_SPLITS * N_REPEATS,
            "random_state": RANDOM_STATE,
        },
        "cv_summary": cv_summary,
        "locked_split": {
            "train_records": int(len(train)),
            "validation_records": int(len(validation)),
            "test_records": int(len(test)),
            "validation_best_f2": float(best_validation_f2),
            "validation_selected_threshold": float(best_threshold),
            "test_at_0.50": test_metrics_050,
            "test_at_validation_threshold": test_metrics_selected,
        },
        "warning": (
            "The adjudicated dataset is only 40 records and the "
            "locked test set is 6 records. These results are "
            "diagnostic research evidence, not production "
            "generalization estimates."
        ),
    }

    summary_path = EXPERIMENT_DIR / "summary.json"

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("HYBRID CV SUMMARY")
    print("=" * 78)

    for metric_name in [
        "accuracy",
        "precision",
        "recall",
        "f2",
        "pr_auc",
    ]:
        stats = cv_summary["metrics"][metric_name]

        print(
            f"{metric_name:<12}: "
            f"{stats['mean']:.4f} ± {stats['std']:.4f} "
            f"(range {stats['min']:.4f}–{stats['max']:.4f})"
        )

    print()
    print("=" * 78)
    print("LOCKED TEST — THRESHOLD 0.50")
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
            f"{test_metrics_050[key]}"
        )

    print()
    print("=" * 78)
    print("LOCKED TEST — VALIDATION-SELECTED THRESHOLD")
    print("=" * 78)

    print(
        f"Selected threshold      : "
        f"{best_threshold:.6f}"
    )
    print(
        f"Validation F2           : "
        f"{best_validation_f2:.6f}"
    )

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
            f"{test_metrics_selected[key]}"
        )

    print()
    print("=" * 78)
    print("LOCKED TEST PREDICTIONS")
    print("=" * 78)

    print(
        predictions[
            [
                "report_id",
                "final_sif_potential",
                "sif_probability",
                "prediction_at_0.50",
                "prediction_at_validation_threshold",
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
    print("FILES")
    print("=" * 78)
    print(f"CV folds  : {fold_results_path}")
    print(f"Summary   : {summary_path}")
    print(
        f"Test pred.: "
        f"{EXPERIMENT_DIR / 'locked_test_predictions.csv'}"
    )

    print()
    print("=" * 78)
    print("HYBRID EXPERIMENT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
