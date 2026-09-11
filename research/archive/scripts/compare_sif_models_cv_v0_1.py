from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "resolved_annotations_v0.1.csv"
)

EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "model_comparison_cv_v0.1"
)

RESULTS_CSV = EXPERIMENT_DIR / "fold_results.csv"
SUMMARY_JSON = EXPERIMENT_DIR / "summary.json"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 256


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Resolved annotation file not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    label_candidates = [
        "final_sif_potential",
        "sif_potential",
        "resolved_sif_potential",
    ]

    label_column = next(
        (c for c in label_candidates if c in df.columns),
        None,
    )

    if label_column is None:
        raise ValueError(
            "Could not find a SIF label column. "
            f"Expected one of: {label_candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    if "description" not in df.columns:
        raise ValueError(
            "Resolved annotation file does not contain 'description'. "
            f"Available columns: {list(df.columns)}"
        )

    out = df[["description", label_column]].copy()
    out = out.rename(columns={label_column: "label"})

    out["description"] = (
        out["description"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    out["label"] = (
        out["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    out = out[out["label"].isin(["YES", "NO"])].reset_index(drop=True)

    if out.empty:
        raise ValueError("No binary YES/NO resolved annotations were found.")

    if len(out) < 10:
        raise ValueError(
            f"Only {len(out)} binary records found; CV is not meaningful."
        )

    return out


def binary_labels(df: pd.DataFrame) -> np.ndarray:
    return (df["label"].eq("YES")).astype(int).to_numpy()


def metric_dict(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, float]:
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
    }


def summarize(results: pd.DataFrame, model_name: str) -> dict:
    subset = results[results["model"] == model_name]

    metrics = {}
    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f2",
        "pr_auc",
        "false_positive_count",
        "false_negative_count",
    ]:
        values = subset[metric].astype(float)

        metrics[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    return {
        "model": model_name,
        "folds": int(len(subset)),
        "metrics": metrics,
    }


def main() -> None:
    set_seed(RANDOM_STATE)

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    texts = df["description"].tolist()
    y = binary_labels(df)

    yes_count = int((y == 1).sum())
    no_count = int((y == 0).sum())

    print("=" * 78)
    print("RAKSHAK SIF MODEL COMPARISON — REPEATED STRATIFIED CV v0.1")
    print("=" * 78)
    print(f"Dataset     : {DATA_PATH}")
    print(f"Records     : {len(df)}")
    print(f"YES         : {yes_count}")
    print(f"NO          : {no_count}")
    print(f"CV          : {N_SPLITS}-fold × {N_REPEATS} repeats")
    print(f"Evaluations : {N_SPLITS * N_REPEATS} folds/model")
    print("Primary     : PR-AUC + F2/Recall")
    print("Threshold   : 0.50 for fold-level F2/Recall")
    print()

    # --------------------------------------------------------
    # Precompute frozen transformer embeddings once.
    # The encoder is pretrained and never fitted on our labels.
    # --------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("-" * 78)
    print("LOADING TRANSFORMER ENCODER")
    print("-" * 78)
    print(f"Model : {MODEL_NAME}")
    print(f"Device: {device}")

    encoder = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    # Use the new API where available.
    encoder.max_seq_length = MAX_SEQ_LENGTH

    embeddings = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    print(
        f"Embedding dimension: "
        f"{embeddings.shape[1]}"
    )

    # --------------------------------------------------------
    # Repeated stratified folds.
    # --------------------------------------------------------

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    results: list[dict] = []

    print()
    print("-" * 78)
    print("RUNNING CROSS-VALIDATION")
    print("-" * 78)

    for fold_number, (train_idx, test_idx) in enumerate(
        splitter.split(texts, y),
        start=1,
    ):
        X_text_train = [texts[i] for i in train_idx]
        X_text_test = [texts[i] for i in test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # ====================================================
        # MODEL 1 — TF-IDF + Logistic Regression
        # ====================================================

        tfidf = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            sublinear_tf=True,
            max_features=10000,
        )

        X_train_tfidf = tfidf.fit_transform(X_text_train)
        X_test_tfidf = tfidf.transform(X_text_test)

        tfidf_classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )

        tfidf_classifier.fit(
            X_train_tfidf,
            y_train,
        )

        tfidf_probabilities = (
            tfidf_classifier.predict_proba(X_test_tfidf)[:, 1]
        )

        tfidf_metrics = metric_dict(
            y_test,
            tfidf_probabilities,
        )

        results.append(
            {
                "fold": fold_number,
                "model": "TF-IDF + Logistic Regression",
                **tfidf_metrics,
            }
        )

        # ====================================================
        # MODEL 2 — Transformer Embeddings + Logistic Regression
        # ====================================================

        X_train_emb = embeddings[train_idx]
        X_test_emb = embeddings[test_idx]

        emb_classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )

        emb_classifier.fit(
            X_train_emb,
            y_train,
        )

        emb_probabilities = (
            emb_classifier.predict_proba(X_test_emb)[:, 1]
        )

        emb_metrics = metric_dict(
            y_test,
            emb_probabilities,
        )

        results.append(
            {
                "fold": fold_number,
                "model": (
                    "Transformer Embeddings + "
                    "Logistic Regression"
                ),
                **emb_metrics,
            }
        )

        print(
            f"Fold {fold_number:02d}/{N_SPLITS * N_REPEATS}: "
            f"TFIDF F2={tfidf_metrics['f2']:.3f}, "
            f"TFIDF PR-AUC={tfidf_metrics['pr_auc']:.3f} | "
            f"Transformer F2={emb_metrics['f2']:.3f}, "
            f"Transformer PR-AUC={emb_metrics['pr_auc']:.3f}"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_CSV,
        index=False,
        encoding="utf-8",
    )

    tfidf_summary = summarize(
        results_df,
        "TF-IDF + Logistic Regression",
    )

    transformer_summary = summarize(
        results_df,
        "Transformer Embeddings + Logistic Regression",
    )

    # --------------------------------------------------------
    # Comparison deltas
    # --------------------------------------------------------

    delta = {}

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f2",
        "pr_auc",
    ]:
        tfidf_mean = tfidf_summary["metrics"][metric]["mean"]
        transformer_mean = transformer_summary["metrics"][metric]["mean"]

        delta[metric] = {
            "transformer_minus_tfidf": float(
                transformer_mean - tfidf_mean
            )
        }

    summary = {
        "experiment": "model_comparison_cv_v0.1",
        "dataset": str(
            DATA_PATH.relative_to(PROJECT_ROOT)
        ),
        "records": int(len(df)),
        "yes_records": yes_count,
        "no_records": no_count,
        "cv": {
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "total_folds_per_model": N_SPLITS * N_REPEATS,
            "random_state": RANDOM_STATE,
        },
        "transformer": {
            "model_name": MODEL_NAME,
            "device": device,
            "embedding_dimension": int(embeddings.shape[1]),
            "max_seq_length": MAX_SEQ_LENGTH,
            "batch_size": BATCH_SIZE,
            "fine_tuned": False,
        },
        "models": {
            "tfidf": tfidf_summary,
            "transformer_embeddings": transformer_summary,
        },
        "delta_transformer_minus_tfidf": delta,
        "interpretation": (
            "Repeated cross-validation on the 40 resolved binary "
            "annotations. These results are more informative than "
            "the 6-record locked test comparison, but the dataset "
            "remains small and should not be treated as production "
            "generalization evidence."
        ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 78)

    for model_name, model_summary in [
        (
            "TF-IDF + Logistic Regression",
            tfidf_summary,
        ),
        (
            "Transformer Embeddings + Logistic Regression",
            transformer_summary,
        ),
    ]:
        print()
        print(model_name)
        print("-" * 78)

        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f2",
            "pr_auc",
        ]:
            stats = model_summary["metrics"][metric]
            print(
                f"{metric:<12}: "
                f"{stats['mean']:.4f} ± {stats['std']:.4f} "
                f"(range {stats['min']:.4f}–{stats['max']:.4f})"
            )

    print()
    print("TRANSFORMER − TF-IDF DELTAS")
    print("-" * 78)

    for metric, values in delta.items():
        print(
            f"{metric:<12}: "
            f"{values['transformer_minus_tfidf']:+.4f}"
        )

    print()
    print("FILES")
    print("-" * 78)
    print(f"Fold results : {RESULTS_CSV}")
    print(f"Summary      : {SUMMARY_JSON}")

    print()
    print("=" * 78)
    print("CV EXPERIMENT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
