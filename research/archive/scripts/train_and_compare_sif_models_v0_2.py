from __future__ import annotations

import json
import random
from pathlib import Path

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

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sif_splits_v0.2"
)

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
TEST_PATH = SPLIT_DIR / "test.csv"

EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "model_comparison_v0.2"
)

TFIDF_DIR = EXPERIMENT_ROOT / "tfidf"
TRANSFORMER_DIR = EXPERIMENT_ROOT / "transformer_embedding"
HYBRID_DIR = EXPERIMENT_ROOT / "hybrid"

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


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split:\n{path}")

    df = pd.read_csv(path).copy()

    required = {
        "report_id",
        "description",
        "final_sif_potential",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    df["report_id"] = (
        df["report_id"].astype(str).str.strip()
    )
    df["description"] = (
        df["description"].fillna("").astype(str).str.strip()
    )
    df["final_sif_potential"] = (
        df["final_sif_potential"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if df["report_id"].duplicated().any():
        raise ValueError(
            f"{path.name} contains duplicate report_id values."
        )

    if not df["final_sif_potential"].isin(["YES", "NO"]).all():
        bad = sorted(
            set(df["final_sif_potential"])
            - {"YES", "NO"}
        )
        raise ValueError(
            f"{path.name} contains unsupported labels: {bad}"
        )

    if df["description"].eq("").any():
        raise ValueError(
            f"{path.name} contains empty descriptions."
        )

    return df.reset_index(drop=True)


def labels(df: pd.DataFrame) -> np.ndarray:
    return (
        df["final_sif_potential"]
        .map({"NO": 0, "YES": 1})
        .astype(int)
        .to_numpy()
    )


def validate_locked_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    train_ids = set(train["report_id"])
    validation_ids = set(validation["report_id"])
    test_ids = set(test["report_id"])

    if train_ids & validation_ids:
        raise RuntimeError(
            "Train/validation report-ID overlap detected."
        )
    if train_ids & test_ids:
        raise RuntimeError(
            "Train/test report-ID overlap detected."
        )
    if validation_ids & test_ids:
        raise RuntimeError(
            "Validation/test report-ID overlap detected."
        )

    if (
        "duplicate_group_id" in train.columns
        and "duplicate_group_id" in validation.columns
        and "duplicate_group_id" in test.columns
    ):
        train_groups = set(
            train["duplicate_group_id"].fillna("").astype(str)
        )
        validation_groups = set(
            validation["duplicate_group_id"].fillna("").astype(str)
        )
        test_groups = set(
            test["duplicate_group_id"].fillna("").astype(str)
        )

        if train_groups & validation_groups:
            raise RuntimeError(
                "Duplicate-group overlap: train/validation."
            )
        if train_groups & test_groups:
            raise RuntimeError(
                "Duplicate-group overlap: train/test."
            )
        if validation_groups & test_groups:
            raise RuntimeError(
                "Duplicate-group overlap: validation/test."
            )


def evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (
        probabilities >= threshold
    ).astype(int)

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
                beta=2,
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


def summarize(
    results: pd.DataFrame,
    model: str,
) -> dict:
    subset = results[
        results["model"].eq(model)
    ]

    output = {
        "model": model,
        "folds": int(len(subset)),
        "metrics": {},
    }

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
        output["metrics"][metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    return output


def encode(
    encoder: SentenceTransformer,
    texts: list[str],
) -> np.ndarray:
    return encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def main() -> None:
    set_seed()

    EXPERIMENT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    TFIDF_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFORMER_DIR.mkdir(parents=True, exist_ok=True)
    HYBRID_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split(TRAIN_PATH)
    validation = load_split(VALIDATION_PATH)
    test = load_split(TEST_PATH)

    validate_locked_split(
        train,
        validation,
        test,
    )

    all_texts = (
        pd.concat(
            [
                train[["report_id", "description"]],
                validation[["report_id", "description"]],
                test[["report_id", "description"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates("report_id")
    )

    # --------------------------------------------------------
    # Report split composition.
    # --------------------------------------------------------

    print("=" * 78)
    print("RAKSHAK MODEL COMPARISON v0.2 — 70-LABEL DATASET")
    print("=" * 78)

    for name, df in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:
        y = labels(df)
        print(
            f"{name:<12}: {len(df):>3} records "
            f"(YES={int((y == 1).sum()):>2}, "
            f"NO={int((y == 0).sum()):>2})"
        )

    print()
    print(
        "Locked test set is never used for model or threshold selection."
    )

    # --------------------------------------------------------
    # Transformer embeddings.
    # --------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print()
    print("-" * 78)
    print("LOADING TRANSFORMER ENCODER")
    print("-" * 78)
    print(f"Model : {MODEL_NAME}")
    print(f"Device: {device}")

    encoder = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )
    encoder.max_seq_length = MAX_SEQ_LENGTH

    # Encode each unique narrative exactly once.
    print()
    print("Encoding all unique split narratives...")

    unique_texts = all_texts["description"].tolist()
    unique_ids = all_texts["report_id"].tolist()

    unique_embeddings = encode(
        encoder,
        unique_texts,
    )

    embedding_lookup = {
        report_id: unique_embeddings[i]
        for i, report_id in enumerate(unique_ids)
    }

    def embeddings_for(df: pd.DataFrame) -> np.ndarray:
        return np.stack(
            [
                embedding_lookup[report_id]
                for report_id in df["report_id"]
            ]
        ).astype(np.float32)

    train_emb = embeddings_for(train)
    validation_emb = embeddings_for(validation)
    test_emb = embeddings_for(test)

    np.save(
        TRANSFORMER_DIR / "train_embeddings.npy",
        train_emb,
    )
    np.save(
        TRANSFORMER_DIR / "validation_embeddings.npy",
        validation_emb,
    )
    np.save(
        TRANSFORMER_DIR / "test_embeddings.npy",
        test_emb,
    )

    # --------------------------------------------------------
    # Repeated CV over all 70 resolved labels.
    # --------------------------------------------------------

    resolved = pd.concat(
        [
            train,
            validation,
            test,
        ],
        ignore_index=True,
    )

    resolved_texts = resolved["description"].tolist()
    resolved_ids = resolved["report_id"].tolist()
    resolved_y = labels(resolved)

    resolved_embedding_matrix = np.stack(
        [
            embedding_lookup[report_id]
            for report_id in resolved_ids
        ]
    ).astype(np.float32)

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    results: list[dict] = []

    print()
    print("-" * 78)
    print("RUNNING 50-FOLD REPEATED STRATIFIED CV")
    print("-" * 78)

    for fold_number, (train_idx, fold_idx) in enumerate(
        splitter.split(
            resolved_texts,
            resolved_y,
        ),
        start=1,
    ):
        fold_train_texts = [
            resolved_texts[i]
            for i in train_idx
        ]
        fold_test_texts = [
            resolved_texts[i]
            for i in fold_idx
        ]

        y_train = resolved_y[train_idx]
        y_fold = resolved_y[fold_idx]

        # ============================
        # TF-IDF
        # ============================

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
            fold_train_texts
        )
        X_fold_tfidf = vectorizer.transform(
            fold_test_texts
        )

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
            tfidf_classifier.predict_proba(
                X_fold_tfidf
            )[:, 1]
        )

        tfidf_metrics = evaluate(
            y_fold,
            tfidf_probabilities,
            0.50,
        )

        results.append(
            {
                "fold": fold_number,
                "model": "TF-IDF + Logistic Regression",
                **tfidf_metrics,
            }
        )

        # ============================
        # Transformer embeddings
        # ============================

        X_train_emb = resolved_embedding_matrix[
            train_idx
        ]
        X_fold_emb = resolved_embedding_matrix[
            fold_idx
        ]

        transformer_classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )
        transformer_classifier.fit(
            X_train_emb,
            y_train,
        )

        transformer_probabilities = (
            transformer_classifier.predict_proba(
                X_fold_emb
            )[:, 1]
        )

        transformer_metrics = evaluate(
            y_fold,
            transformer_probabilities,
            0.50,
        )

        results.append(
            {
                "fold": fold_number,
                "model": (
                    "Transformer Embeddings + "
                    "Logistic Regression"
                ),
                **transformer_metrics,
            }
        )

        # ============================
        # Hybrid
        # ============================

        X_train_hybrid = hstack(
            [
                X_train_tfidf,
                csr_matrix(X_train_emb),
            ],
            format="csr",
        )

        X_fold_hybrid = hstack(
            [
                X_fold_tfidf,
                csr_matrix(X_fold_emb),
            ],
            format="csr",
        )

        hybrid_classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )
        hybrid_classifier.fit(
            X_train_hybrid,
            y_train,
        )

        hybrid_probabilities = (
            hybrid_classifier.predict_proba(
                X_fold_hybrid
            )[:, 1]
        )

        hybrid_metrics = evaluate(
            y_fold,
            hybrid_probabilities,
            0.50,
        )

        results.append(
            {
                "fold": fold_number,
                "model": "Hybrid TF-IDF + Transformer",
                **hybrid_metrics,
            }
        )

        print(
            f"Fold {fold_number:02d}/50 | "
            f"TFIDF F2={tfidf_metrics['f2']:.3f} "
            f"PR={tfidf_metrics['pr_auc']:.3f} | "
            f"TRANS F2={transformer_metrics['f2']:.3f} "
            f"PR={transformer_metrics['pr_auc']:.3f} | "
            f"HYBRID F2={hybrid_metrics['f2']:.3f} "
            f"PR={hybrid_metrics['pr_auc']:.3f}"
        )

    cv_results = pd.DataFrame(results)

    cv_results_path = (
        EXPERIMENT_ROOT
        / "repeated_cv_results.csv"
    )
    cv_results.to_csv(
        cv_results_path,
        index=False,
        encoding="utf-8",
    )

    summaries = {
        model: summarize(
            cv_results,
            model,
        )
        for model in [
            "TF-IDF + Logistic Regression",
            "Transformer Embeddings + Logistic Regression",
            "Hybrid TF-IDF + Transformer",
        ]
    }

    # --------------------------------------------------------
    # Locked split experiment.
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("LOCKED SPLIT EVALUATION")
    print("=" * 78)

    # Fit TF-IDF only on train.
    locked_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
        max_features=TFIDF_MAX_FEATURES,
    )

    X_train_tfidf = locked_vectorizer.fit_transform(
        train["description"].tolist()
    )
    X_validation_tfidf = locked_vectorizer.transform(
        validation["description"].tolist()
    )
    X_test_tfidf = locked_vectorizer.transform(
        test["description"].tolist()
    )

    y_train = labels(train)
    y_validation = labels(validation)
    y_test = labels(test)

    # ---- TF-IDF ----
    tfidf_locked = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )
    tfidf_locked.fit(
        X_train_tfidf,
        y_train,
    )

    tfidf_test_probability = (
        tfidf_locked.predict_proba(
            X_test_tfidf
        )[:, 1]
    )

    tfidf_validation_probability = (
        tfidf_locked.predict_proba(
            X_validation_tfidf
        )[:, 1]
    )

    # ---- Transformer ----
    transformer_locked = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )
    transformer_locked.fit(
        train_emb,
        y_train,
    )

    transformer_test_probability = (
        transformer_locked.predict_proba(
            test_emb
        )[:, 1]
    )

    transformer_validation_probability = (
        transformer_locked.predict_proba(
            validation_emb
        )[:, 1]
    )

    # ---- Hybrid ----
    X_train_hybrid = hstack(
        [
            X_train_tfidf,
            csr_matrix(train_emb),
        ],
        format="csr",
    )

    X_validation_hybrid = hstack(
        [
            X_validation_tfidf,
            csr_matrix(validation_emb),
        ],
        format="csr",
    )

    X_test_hybrid = hstack(
        [
            X_test_tfidf,
            csr_matrix(test_emb),
        ],
        format="csr",
    )

    hybrid_locked = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )
    hybrid_locked.fit(
        X_train_hybrid,
        y_train,
    )

    hybrid_test_probability = (
        hybrid_locked.predict_proba(
            X_test_hybrid
        )[:, 1]
    )

    hybrid_validation_probability = (
        hybrid_locked.predict_proba(
            X_validation_hybrid
        )[:, 1]
    )

    # --------------------------------------------------------
    # Threshold handling:
    #
    # With v0.1, validation-selected threshold collapsed to
    # 0.05 on the hybrid model. We therefore report:
    #   1) fixed 0.50 results
    #   2) validation-selected threshold as diagnostic only
    #
    # We do not claim that the validation-derived threshold is
    # operational.
    # --------------------------------------------------------

    def choose_f2_threshold(
        y_true: np.ndarray,
        probabilities: np.ndarray,
    ) -> tuple[float, float]:
        thresholds = np.unique(
            np.concatenate(
                [
                    np.linspace(0.10, 0.90, 161),
                    probabilities,
                    np.asarray([0.50]),
                ]
            )
        )

        best_threshold = 0.50
        best_score = -1.0

        for threshold in thresholds:
            prediction = (
                probabilities >= threshold
            ).astype(int)

            score = fbeta_score(
                y_true,
                prediction,
                beta=2,
                zero_division=0,
            )

            if score > best_score:
                best_score = float(score)
                best_threshold = float(threshold)

        return best_threshold, best_score

    model_probabilities = {
        "TF-IDF + Logistic Regression": (
            tfidf_validation_probability,
            tfidf_test_probability,
        ),
        "Transformer Embeddings + Logistic Regression": (
            transformer_validation_probability,
            transformer_test_probability,
        ),
        "Hybrid TF-IDF + Transformer": (
            hybrid_validation_probability,
            hybrid_test_probability,
        ),
    }

    locked_summary = {}

    for model_name, (
        validation_probability,
        test_probability,
    ) in model_probabilities.items():
        validation_threshold, validation_f2 = (
            choose_f2_threshold(
                y_validation,
                validation_probability,
            )
        )

        locked_summary[model_name] = {
            "validation_selected_threshold": float(
                validation_threshold
            ),
            "validation_f2_at_selected_threshold": float(
                validation_f2
            ),
            "test_at_0.50": evaluate(
                y_test,
                test_probability,
                0.50,
            ),
            "test_at_validation_selected_threshold": evaluate(
                y_test,
                test_probability,
                validation_threshold,
            ),
        }

    # --------------------------------------------------------
    # Save locked predictions.
    # --------------------------------------------------------

    prediction_frame = test[
        [
            "report_id",
            "description",
            "final_sif_potential",
        ]
    ].copy()

    prediction_frame[
        "tfidf_probability"
    ] = tfidf_test_probability

    prediction_frame[
        "transformer_probability"
    ] = transformer_test_probability

    prediction_frame[
        "hybrid_probability"
    ] = hybrid_test_probability

    prediction_frame[
        "tfidf_prediction_at_0.50"
    ] = np.where(
        tfidf_test_probability >= 0.50,
        "YES",
        "NO",
    )

    prediction_frame[
        "transformer_prediction_at_0.50"
    ] = np.where(
        transformer_test_probability >= 0.50,
        "YES",
        "NO",
    )

    prediction_frame[
        "hybrid_prediction_at_0.50"
    ] = np.where(
        hybrid_test_probability >= 0.50,
        "YES",
        "NO",
    )

    prediction_path = (
        EXPERIMENT_ROOT
        / "locked_test_predictions.csv"
    )

    prediction_frame.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Persist fitted locked models.
    # --------------------------------------------------------

    import joblib

    joblib.dump(
        locked_vectorizer,
        TFIDF_DIR / "tfidf_vectorizer.joblib",
    )

    joblib.dump(
        tfidf_locked,
        TFIDF_DIR / "classifier.joblib",
    )

    joblib.dump(
        transformer_locked,
        TRANSFORMER_DIR / "classifier.joblib",
    )

    joblib.dump(
        hybrid_locked,
        HYBRID_DIR / "classifier.joblib",
    )

    config = {
        "experiment": "model_comparison_v0.2",
        "data_version": "resolved_annotations_v0.2",
        "split_version": "sif_splits_v0.2",
        "transformer_model": MODEL_NAME,
        "device": device,
        "max_seq_length": MAX_SEQ_LENGTH,
        "batch_size": BATCH_SIZE,
        "embedding_dimension": int(
            unique_embeddings.shape[1]
        ),
        "cv": {
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "folds_per_model": N_SPLITS * N_REPEATS,
            "random_state": RANDOM_STATE,
        },
        "models": [
            "TF-IDF + Logistic Regression",
            "Transformer Embeddings + Logistic Regression",
            "Hybrid TF-IDF + Transformer",
        ],
        "threshold_policy": (
            "0.50 is the current fixed diagnostic threshold. "
            "Validation-selected thresholds are reported only "
            "as sensitivity analysis because the validation set "
            "is small."
        ),
    }

    summary = {
        "config": config,
        "cv_summary": summaries,
        "locked_split": locked_summary,
        "files": {
            "cv_results": str(
                cv_results_path.relative_to(PROJECT_ROOT)
            ),
            "test_predictions": str(
                prediction_path.relative_to(PROJECT_ROOT)
            ),
        },
        "warning": (
            "The 70-label dataset remains small. Repeated CV and "
            "the locked test are diagnostic research evidence, "
            "not production generalization estimates."
        ),
    }

    summary_path = (
        EXPERIMENT_ROOT
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console summary.
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("REPEATED CV SUMMARY")
    print("=" * 78)

    for model_name, model_summary in summaries.items():
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
    print("=" * 78)
    print("LOCKED TEST SUMMARY @ FIXED THRESHOLD 0.50")
    print("=" * 78)

    for model_name, values in locked_summary.items():
        metrics_050 = values["test_at_0.50"]

        print()
        print(model_name)
        print(
            f"Accuracy    : {metrics_050['accuracy']:.4f}"
        )
        print(
            f"Precision   : {metrics_050['precision']:.4f}"
        )
        print(
            f"Recall      : {metrics_050['recall']:.4f}"
        )
        print(
            f"F2          : {metrics_050['f2']:.4f}"
        )
        print(
            f"PR-AUC      : {metrics_050['pr_auc']:.4f}"
        )
        print(
            f"FP          : {metrics_050['false_positive_count']}"
        )
        print(
            f"FN          : {metrics_050['false_negative_count']}"
        )

    print()
    print("=" * 78)
    print("LOCKED TEST PREDICTIONS @ 0.50")
    print("=" * 78)

    print(
        prediction_frame[
            [
                "report_id",
                "final_sif_potential",
                "tfidf_probability",
                "transformer_probability",
                "hybrid_probability",
                "tfidf_prediction_at_0.50",
                "transformer_prediction_at_0.50",
                "hybrid_prediction_at_0.50",
            ]
        ]
        .sort_values(
            "hybrid_probability",
            ascending=False,
        )
        .to_string(index=False)
    )

    print()
    print("=" * 78)
    print("FILES")
    print("=" * 78)
    print(f"Summary       : {summary_path}")
    print(f"CV results    : {cv_results_path}")
    print(f"Test results  : {prediction_path}")

    print()
    print("=" * 78)
    print("MODEL COMPARISON v0.2 COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
