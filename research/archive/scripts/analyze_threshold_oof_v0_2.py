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
    average_precision_score,
    fbeta_score,
    precision_recall_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESOLVED_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "resolved_annotations_v0.2.csv"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sif_splits_v0.2"
)

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
TEST_PATH = SPLIT_DIR / "test.csv"

OUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "threshold_analysis_v0.2"
)

OOF_PATH = OUT_DIR / "oof_predictions.csv"
THRESHOLD_PATH = OUT_DIR / "threshold_table.csv"
SUMMARY_PATH = OUT_DIR / "summary.json"

MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 256
MAX_FEATURES = 10000


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    df = pd.read_csv(path).copy()

    required = {"report_id", "description", "final_sif_potential"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} missing columns: {sorted(missing)}"
        )

    df["report_id"] = df["report_id"].astype(str).str.strip()
    df["description"] = (
        df["description"].fillna("").astype(str).str.strip()
    )
    df["final_sif_potential"] = (
        df["final_sif_potential"]
        .astype(str).str.strip().str.upper()
    )

    if not df["final_sif_potential"].isin(["YES", "NO"]).all():
        raise ValueError(
            f"{path.name} contains labels outside YES/NO."
        )

    return df.reset_index(drop=True)


def labels(df: pd.DataFrame) -> np.ndarray:
    return (
        df["final_sif_potential"]
        .map({"NO": 0, "YES": 1})
        .astype(int)
        .to_numpy()
    )


def metric_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    pred = (scores >= threshold).astype(int)

    tp = int(((y_true == 1) & (pred == 1)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f2 = fbeta_score(
        y_true,
        pred,
        beta=2.0,
        zero_division=0,
    )

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f2": float(f2),
        "false_positive_count": fp,
        "false_negative_count": fn,
    }


def choose_threshold_from_oof(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    # Restrict the operational candidate region to 0.20–0.80.
    # Thresholds below 0.20 commonly become "flag almost everything"
    # on very small datasets and are not operationally useful.
    thresholds = np.unique(
        np.round(
            np.concatenate(
                [
                    np.linspace(0.20, 0.80, 601),
                    scores,
                    np.asarray([0.50]),
                ]
            ),
            6,
        )
    )

    best_threshold = 0.50
    best_f2 = -1.0

    for threshold in thresholds:
        current = metric_at_threshold(
            y_true,
            scores,
            float(threshold),
        )

        # Tie-break toward the higher threshold to reduce unnecessary
        # false positives when F2 is identical.
        if (
            current["f2"] > best_f2
            or (
                np.isclose(
                    current["f2"],
                    best_f2,
                )
                and threshold > best_threshold
            )
        ):
            best_f2 = current["f2"]
            best_threshold = float(threshold)

    return best_threshold, best_f2


def main() -> None:
    set_seed()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = load_split(TRAIN_PATH)
    validation = load_split(VALIDATION_PATH)
    test = load_split(TEST_PATH)

    resolved = pd.concat(
        [train, validation, test],
        ignore_index=True,
    )

    # Sanity check: the fixed split is exactly the 70 resolved records.
    if len(resolved) != 70:
        raise ValueError(
            f"Expected 70 resolved records across the split, got {len(resolved)}."
        )

    y = labels(resolved)
    texts = resolved["description"].tolist()
    ids = resolved["report_id"].tolist()

    print("=" * 78)
    print("RAKSHAK OOF THRESHOLD ANALYSIS v0.2")
    print("=" * 78)
    print(
        "Goal: choose a threshold from out-of-fold predictions, "
        "then evaluate it once on the locked 20-record test set."
    )
    print()
    print(f"Resolved records : {len(resolved)}")
    print(f"YES              : {int((y == 1).sum())}")
    print(f"NO               : {int((y == 0).sum())}")
    print(f"CV               : {N_SPLITS}-fold × {N_REPEATS} repeats")
    print()

    # --------------------------------------------------------
    # Load frozen transformer once.
    # --------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    print()
    print("Encoding all resolved narratives...")

    embeddings = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # --------------------------------------------------------
    # Repeated CV OOF-style predictions.
    #
    # Each record appears in many held-out folds because the CV
    # is repeated. We average its out-of-fold probabilities over
    # only the folds in which it was held out.
    # --------------------------------------------------------

    splitter = __import__(
        "sklearn.model_selection",
        fromlist=["RepeatedStratifiedKFold"],
    ).RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    models = {
        "TF-IDF + Logistic Regression": [],
        "Transformer Embeddings + Logistic Regression": [],
        "Hybrid TF-IDF + Transformer": [],
    }

    prediction_lists = {
        name: [[] for _ in range(len(resolved))]
        for name in models
    }

    fold_counter = 0

    print()
    print("-" * 78)
    print("GENERATING OOF PREDICTIONS")
    print("-" * 78)

    for train_idx, holdout_idx in splitter.split(
        texts,
        y,
    ):
        fold_counter += 1

        fold_train_texts = [texts[i] for i in train_idx]
        fold_holdout_texts = [texts[i] for i in holdout_idx]
        y_train = y[train_idx]

        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            sublinear_tf=True,
            max_features=MAX_FEATURES,
        )

        X_train_tfidf = vectorizer.fit_transform(
            fold_train_texts
        )
        X_holdout_tfidf = vectorizer.transform(
            fold_holdout_texts
        )

        # TF-IDF model
        tfidf_model = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )
        tfidf_model.fit(
            X_train_tfidf,
            y_train,
        )
        tfidf_prob = tfidf_model.predict_proba(
            X_holdout_tfidf
        )[:, 1]

        # Transformer model
        X_train_emb = embeddings[train_idx]
        X_holdout_emb = embeddings[holdout_idx]

        transformer_model = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )
        transformer_model.fit(
            X_train_emb,
            y_train,
        )
        transformer_prob = transformer_model.predict_proba(
            X_holdout_emb
        )[:, 1]

        # Hybrid
        X_train_hybrid = hstack(
            [
                X_train_tfidf,
                csr_matrix(X_train_emb),
            ],
            format="csr",
        )
        X_holdout_hybrid = hstack(
            [
                X_holdout_tfidf,
                csr_matrix(X_holdout_emb),
            ],
            format="csr",
        )

        hybrid_model = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )
        hybrid_model.fit(
            X_train_hybrid,
            y_train,
        )
        hybrid_prob = hybrid_model.predict_proba(
            X_holdout_hybrid
        )[:, 1]

        for local_position, absolute_index in enumerate(
            holdout_idx
        ):
            prediction_lists[
                "TF-IDF + Logistic Regression"
            ][absolute_index].append(
                float(tfidf_prob[local_position])
            )

            prediction_lists[
                "Transformer Embeddings + Logistic Regression"
            ][absolute_index].append(
                float(
                    transformer_prob[
                        local_position
                    ]
                )
            )

            prediction_lists[
                "Hybrid TF-IDF + Transformer"
            ][absolute_index].append(
                float(
                    hybrid_prob[
                        local_position
                    ]
                )
            )

        print(
            f"Fold {fold_counter:02d}/50 complete"
        )

    oof = resolved[
        [
            "report_id",
            "description",
            "final_sif_potential",
        ]
    ].copy()

    for model_name, lists in prediction_lists.items():
        oof[model_name] = [
            float(np.mean(values))
            if values
            else np.nan
            for values in lists
        ]

    if oof[
        [
            "TF-IDF + Logistic Regression",
            "Transformer Embeddings + Logistic Regression",
            "Hybrid TF-IDF + Transformer",
        ]
    ].isna().any().any():
        raise RuntimeError(
            "Some records did not receive OOF predictions."
        )

    # --------------------------------------------------------
    # Threshold tables.
    # --------------------------------------------------------

    threshold_rows = []
    model_summaries = {}

    for model_name in models:
        scores = oof[model_name].to_numpy(dtype=float)

        threshold, best_f2 = choose_threshold_from_oof(
            y,
            scores,
        )

        table = []

        for t in np.round(
            np.arange(0.20, 0.801, 0.01),
            2,
        ):
            current = metric_at_threshold(
                y,
                scores,
                float(t),
            )
            current["model"] = model_name
            table.append(current)

        threshold_rows.extend(table)

        model_summaries[model_name] = {
            "oof_pr_auc": float(
                average_precision_score(
                    y,
                    scores,
                )
            ),
            "selected_threshold": float(threshold),
            "selected_oof_f2": float(best_f2),
            "selected_oof_metrics": metric_at_threshold(
                y,
                scores,
                threshold,
            ),
        }

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    threshold_df.to_csv(
        THRESHOLD_PATH,
        index=False,
        encoding="utf-8",
    )

    oof.to_csv(
        OOF_PATH,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Locked test evaluation.
    #
    # Important: thresholds were selected from OOF predictions
    # only. The test set is not used here for threshold choice.
    # --------------------------------------------------------

    print()
    print("-" * 78)
    print("LOCKED TEST EVALUATION")
    print("-" * 78)

    # Fit each model on the 30-record train split only.
    y_train = labels(train)
    y_validation = labels(validation)
    y_test = labels(test)

    train_texts = train["description"].tolist()
    validation_texts = validation["description"].tolist()
    test_texts = test["description"].tolist()

    train_emb = encoder.encode(
        train_texts,
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

    validation_emb = encoder.encode(
        validation["description"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
        max_features=MAX_FEATURES,
    )

    train_tfidf = vectorizer.fit_transform(
        train_texts
    )
    test_tfidf = vectorizer.transform(
        test_texts
    )
    validation_tfidf = vectorizer.transform(
        validation_texts
    )

    # Fit three locked models.
    tfidf_model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )
    tfidf_model.fit(
        train_tfidf,
        y_train,
    )
    tfidf_test = tfidf_model.predict_proba(
        test_tfidf
    )[:, 1]

    transformer_model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )
    transformer_model.fit(
        train_emb,
        y_train,
    )
    transformer_test = transformer_model.predict_proba(
        test_emb
    )[:, 1]

    hybrid_train = hstack(
        [
            train_tfidf,
            csr_matrix(train_emb),
        ],
        format="csr",
    )

    hybrid_test = hstack(
        [
            test_tfidf,
            csr_matrix(test_emb),
        ],
        format="csr",
    )

    hybrid_model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )
    hybrid_model.fit(
        hybrid_train,
        y_train,
    )
    hybrid_test = hybrid_model.predict_proba(
        hybrid_test
    )[:, 1]

    locked_predictions = test[
        [
            "report_id",
            "description",
            "final_sif_potential",
        ]
    ].copy()

    locked_predictions["tfidf_probability"] = tfidf_test
    locked_predictions[
        "transformer_probability"
    ] = transformer_test
    locked_predictions["hybrid_probability"] = hybrid_test

    locked_test_metrics = {}

    probability_map = {
        "TF-IDF + Logistic Regression": tfidf_test,
        "Transformer Embeddings + Logistic Regression": transformer_test,
        "Hybrid TF-IDF + Transformer": hybrid_test,
    }

    for model_name, probabilities in probability_map.items():
        selected_threshold = model_summaries[
            model_name
        ]["selected_threshold"]

        fixed = metric_at_threshold(
            y_test,
            probabilities,
            0.50,
        )

        selected = metric_at_threshold(
            y_test,
            probabilities,
            selected_threshold,
        )

        locked_test_metrics[model_name] = {
            "selected_threshold": float(
                selected_threshold
            ),
            "at_0.50": fixed,
            "at_oof_selected_threshold": selected,
        }

        locked_predictions[
            f"{model_name}_prediction_at_oof_threshold"
        ] = np.where(
            probabilities >= selected_threshold,
            "YES",
            "NO",
        )

    locked_predictions_path = (
        OUT_DIR / "locked_test_predictions.csv"
    )

    locked_predictions.to_csv(
        locked_predictions_path,
        index=False,
        encoding="utf-8",
    )

    summary = {
        "experiment": "threshold_analysis_v0.2",
        "dataset_version": "resolved_annotations_v0.2",
        "split_version": "sif_splits_v0.2",
        "cv": {
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "total_folds": N_SPLITS * N_REPEATS,
            "oof_aggregation": (
                "mean probability across held-out repeats"
            ),
        },
        "threshold_policy": (
            "Selected from repeated-CV out-of-fold predictions "
            "using F2 within the 0.20–0.80 operating range. "
            "Locked test was not used for threshold selection."
        ),
        "oof_models": model_summaries,
        "locked_test": locked_test_metrics,
        "files": {
            "oof_predictions": str(
                OOF_PATH.relative_to(PROJECT_ROOT)
            ),
            "threshold_table": str(
                THRESHOLD_PATH.relative_to(PROJECT_ROOT)
            ),
            "locked_test_predictions": str(
                locked_predictions_path.relative_to(PROJECT_ROOT)
            ),
        },
        "warning": (
            "The labeled dataset contains only 70 records. "
            "OOF threshold selection is more robust than using "
            "a 20-record validation set alone, but results remain "
            "diagnostic rather than production generalization evidence."
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("OOF THRESHOLD SUMMARY")
    print("=" * 78)

    for model_name, values in model_summaries.items():
        selected = values["selected_oof_metrics"]

        print()
        print(model_name)
        print("-" * 78)
        print(
            f"OOF PR-AUC          : "
            f"{values['oof_pr_auc']:.4f}"
        )
        print(
            f"Selected threshold  : "
            f"{values['selected_threshold']:.4f}"
        )
        print(
            f"OOF F2              : "
            f"{values['selected_oof_f2']:.4f}"
        )
        print(
            f"OOF precision       : "
            f"{selected['precision']:.4f}"
        )
        print(
            f"OOF recall          : "
            f"{selected['recall']:.4f}"
        )
        print(
            f"OOF false positives : "
            f"{selected['false_positive_count']}"
        )
        print(
            f"OOF false negatives : "
            f"{selected['false_negative_count']}"
        )

    print()
    print("=" * 78)
    print("LOCKED TEST — 0.50 VS OOF-SELECTED THRESHOLD")
    print("=" * 78)

    for model_name, values in locked_test_metrics.items():
        fixed = values["at_0.50"]
        selected = values["at_oof_selected_threshold"]

        print()
        print(model_name)
        print(
            f"OOF threshold      : "
            f"{values['selected_threshold']:.4f}"
        )
        print(
            f"@0.50      → "
            f"F2={fixed['f2']:.4f}, "
            f"Recall={fixed['recall']:.4f}, "
            f"Precision={fixed['precision']:.4f}, "
            f"FP={fixed['false_positive_count']}, "
            f"FN={fixed['false_negative_count']}"
        )
        print(
            f"@OOF thresh → "
            f"F2={selected['f2']:.4f}, "
            f"Recall={selected['recall']:.4f}, "
            f"Precision={selected['precision']:.4f}, "
            f"FP={selected['false_positive_count']}, "
            f"FN={selected['false_negative_count']}"
        )

    print()
    print("=" * 78)
    print("FILES")
    print("=" * 78)
    print(f"OOF predictions      : {OOF_PATH}")
    print(f"Threshold table      : {THRESHOLD_PATH}")
    print(f"Locked test          : {locked_predictions_path}")
    print(f"Summary              : {SUMMARY_PATH}")

    print()
    print("=" * 78)
    print("THRESHOLD ANALYSIS COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
