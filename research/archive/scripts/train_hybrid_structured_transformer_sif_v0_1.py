"""
RAKSHAK Hybrid Structured + Transformer SIF Model v0.1
=======================================================

Leakage-safe experiment for SIH26165 SIF precursor detection.

Compares three representations on the same v0.2 locked split:
    1) Transformer embeddings only
    2) Mechanism-only structured features only
    3) Transformer + mechanism-only structured features

Design:
- Split IDs come from data/processed/sif_splits_v0.2.
- Test split is never used for model/threshold selection.
- Primary model is regularized Logistic Regression.
- Threshold is selected on validation using F2 with precision >= 0.50 when
  such an operating point exists.
- Repeated stratified CV is run on the 68 binary-labelled records as a
  secondary robustness estimate. CV is not used to unlock the test set.

Outputs:
    experiments/sif_hybrid_structured_transformer_v0.1/
        representation_comparison.csv
        repeated_cv_results.csv
        repeated_cv_summary.csv
        validation_thresholds.csv
        locked_test_predictions.csv
        summary.json

This is a research baseline, not a production safety decision system.
"""

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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "sif_feature_builder_v0.1"
    / "sif_features.csv"
)

LABEL_PATH = (
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "sif_hybrid_structured_transformer_v0.1"
)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RANDOM_STATE = 42
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 8
N_SPLITS = 5
N_REPEATS = 10


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def load_binary_annotations() -> pd.DataFrame:
    if not LABEL_PATH.exists():
        raise FileNotFoundError(f"Missing label file: {LABEL_PATH}")

    df = pd.read_csv(LABEL_PATH)
    required = {"report_id", "description", "final_sif_potential"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in labels: {sorted(missing)}")

    df = df.copy()
    df["report_id"] = df["report_id"].astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["final_sif_potential"] = normalize_label(df["final_sif_potential"])
    df = df[df["final_sif_potential"].isin(["YES", "NO"])].copy()
    df["label"] = (df["final_sif_potential"] == "YES").astype(int)

    return df[["report_id", "description", "label"]].drop_duplicates("report_id")


def load_feature_table() -> pd.DataFrame:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Missing feature table: {FEATURE_PATH}")

    df = pd.read_csv(FEATURE_PATH)
    if "report_id" not in df.columns:
        raise ValueError("Feature table must contain report_id")
    df["report_id"] = df["report_id"].astype(str)
    return df


def load_split_ids() -> dict[str, set[str]]:
    result = {}
    for split in ("train", "validation", "test"):
        path = SPLIT_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split: {path}")
        ids = pd.read_csv(path)["report_id"].astype(str)
        result[split] = set(ids)
    return result


def mechanism_columns(feature_df: pd.DataFrame) -> list[str]:
    excluded = {
        "report_id",
        "sif_potential",
        "final_sif_potential",
        "resolved_sif_potential",
        "actual_outcome",
        "potential_accident_level",
    }

    # The v0.1 feature builder uses actual_* and potential_* prefixes for
    # direct consequence information. Exclude both for mechanism-only mode.
    cols = []
    for col in feature_df.columns:
        if col in excluded:
            continue
        if col.startswith("actual_") or col.startswith("potential_"):
            continue
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            cols.append(col)
    return cols


def metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (np.asarray(prob) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2.0, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, prob))
        if len(np.unique(y_true)) == 2 else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_search(y: np.ndarray, prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.linspace(0.20, 0.80, 121):
        m = metrics(y, prob, float(threshold))
        rows.append({"threshold": float(threshold), **m})

    table = pd.DataFrame(rows)
    candidates = table[table["precision"] >= 0.50]
    if candidates.empty:
        candidates = table

    # Safety-oriented selection: F2 first, then recall, precision, then the
    # lower threshold as a deterministic tie-break.
    best = candidates.sort_values(
        ["f2", "recall", "precision", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return float(best["threshold"]), table


def make_classifier() -> Pipeline:
    # L2 is deliberately used for the hybrid dense representation: the
    # transformer contributes 384 continuous dimensions and the structured
    # representation contributes a small set of binary/numeric features.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            solver="liblinear",
            penalty="l2",
            C=0.50,
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )),
    ])


def make_l1_classifier() -> Pipeline:
    # Structured-only comparison uses the current leading configuration from
    # the previous experiment: mechanism-only + L1 C=0.50.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            solver="liblinear",
            penalty="l1",
            C=0.50,
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )),
    ])


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    model.max_seq_length = MAX_SEQ_LENGTH
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
        device=device,
    )
    return np.asarray(embeddings, dtype=np.float32)


def select_rows(
    ids: set[str],
    index_by_id: dict[str, int],
) -> np.ndarray:
    missing = ids - set(index_by_id)
    if missing:
        raise ValueError(f"Split contains IDs absent from merged data: {sorted(missing)[:10]}")
    return np.array([index_by_id[x] for x in sorted(ids)], dtype=int)


def evaluate_locked_split(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier: Pipeline,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    classifier.fit(X_train, y_train)
    val_prob = classifier.predict_proba(X_val)[:, 1]
    threshold, threshold_table = threshold_search(y_val, val_prob)

    test_prob = classifier.predict_proba(X_test)[:, 1]
    result = {
        "representation": name,
        "validation_threshold": threshold,
        "validation_metrics_at_selected_threshold": metrics(y_val, val_prob, threshold),
        "locked_test_metrics_at_selected_threshold": metrics(y_test, test_prob, threshold),
        "locked_test_metrics_at_0.50": metrics(y_test, test_prob, 0.50),
    }

    threshold_table = threshold_table.copy()
    threshold_table.insert(0, "representation", name)
    return result, threshold_table, test_prob


def repeated_cv(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    classifier_factory,
) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    rows = []
    for fold_no, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        model = classifier_factory()
        model.fit(X[train_idx], y[train_idx])
        prob = model.predict_proba(X[test_idx])[:, 1]
        m = metrics(y[test_idx], prob, 0.50)
        rows.append({
            "representation": name,
            "fold": fold_no,
            **m,
        })

    return pd.DataFrame(rows)


def summarize_cv(results: pd.DataFrame) -> pd.DataFrame:
    numeric = ["accuracy", "precision", "recall", "f2", "pr_auc", "fp", "fn"]
    rows = []
    for name, group in results.groupby("representation"):
        row = {"representation": name, "folds": int(len(group))}
        for metric_name in numeric:
            values = group[metric_name].astype(float)
            row[f"{metric_name}_mean"] = float(values.mean())
            row[f"{metric_name}_std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = load_binary_annotations()
    features = load_feature_table()
    splits = load_split_ids()

    merged = labels.merge(features, on="report_id", how="inner", validate="one_to_one")
    if len(merged) != len(labels):
        missing = sorted(set(labels["report_id"]) - set(merged["report_id"]))
        raise ValueError(f"Feature table missing {len(missing)} labelled records: {missing[:10]}")

    # Preserve the split IDs exactly. No random re-splitting here.
    id_index = {rid: i for i, rid in enumerate(merged["report_id"].astype(str))}
    train_idx = select_rows(splits["train"], id_index)
    val_idx = select_rows(splits["validation"], id_index)
    test_idx = select_rows(splits["test"], id_index)

    X_text = merged["description"].fillna("").astype(str).tolist()
    y = merged["label"].to_numpy(dtype=int)

    feature_cols = mechanism_columns(merged)
    if len(feature_cols) == 0:
        raise ValueError("No numeric mechanism-only features found.")
    X_struct = merged[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    print(f"Records: {len(merged)} | YES={int(y.sum())} | NO={int((y == 0).sum())}")
    print(f"Mechanism-only features: {len(feature_cols)}")
    print(f"Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    print(f"Loading encoder: {MODEL_NAME}")
    encoder = SentenceTransformer(MODEL_NAME)
    embeddings = embed_texts(encoder, X_text)
    np.save(OUTPUT_DIR / "transformer_embeddings.npy", embeddings)

    representations = {
        "transformer_only": embeddings,
        "mechanism_only": X_struct,
        "transformer_plus_mechanism": np.hstack([
            embeddings,
            X_struct,
        ]),
    }

    classifiers = {
        "transformer_only": make_classifier,
        "mechanism_only": make_l1_classifier,
        "transformer_plus_mechanism": make_classifier,
    }

    comparison_rows = []
    threshold_tables = []
    prediction_rows = []

    for name, X in representations.items():
        print(f"\n=== {name} ===")
        result, threshold_table, test_prob = evaluate_locked_split(
            name=name,
            X_train=X[train_idx],
            y_train=y[train_idx],
            X_val=X[val_idx],
            y_val=y[val_idx],
            X_test=X[test_idx],
            y_test=y[test_idx],
            classifier=classifiers[name](),
        )
        comparison_rows.append(result)
        threshold_tables.append(threshold_table)

        test_ids = merged.iloc[test_idx]["report_id"].astype(str).to_numpy()
        for rid, true_label, prob in zip(test_ids, y[test_idx], test_prob):
            prediction_rows.append({
                "representation": name,
                "report_id": rid,
                "true_label": int(true_label),
                "probability": float(prob),
                "prediction_at_0.50": int(prob >= 0.50),
                "prediction_at_selected_threshold": int(
                    prob >= result["validation_threshold"]
                ),
            })

        print(json.dumps(result, indent=2))

        # Save the trained locked-split model for inspection/reuse.
        final_model = classifiers[name]()
        final_model.fit(X[train_idx], y[train_idx])
        joblib.dump(final_model, OUTPUT_DIR / f"{name}_model.joblib")

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUTPUT_DIR / "representation_comparison.csv", index=False)
    pd.concat(threshold_tables, ignore_index=True).to_csv(
        OUTPUT_DIR / "validation_thresholds.csv", index=False
    )
    pd.DataFrame(prediction_rows).to_csv(
        OUTPUT_DIR / "locked_test_predictions.csv", index=False
    )

    # Repeated CV at threshold 0.50 is a secondary stability estimate. The
    # locked test remains untouched by this CV.
    cv_frames = []
    for name, X in representations.items():
        print(f"\nRunning repeated CV: {name}")
        cv_frames.append(
            repeated_cv(name, X, y, classifiers[name])
        )

    cv_results = pd.concat(cv_frames, ignore_index=True)
    cv_summary = summarize_cv(cv_results)
    cv_results.to_csv(OUTPUT_DIR / "repeated_cv_results.csv", index=False)
    cv_summary.to_csv(OUTPUT_DIR / "repeated_cv_summary.csv", index=False)

    summary = {
        "experiment": "RAKSHAK Hybrid Structured + Transformer SIF Model v0.1",
        "model_name": MODEL_NAME,
        "dataset_size": int(len(merged)),
        "yes_count": int(y.sum()),
        "no_count": int((y == 0).sum()),
        "split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "mechanism_feature_count": int(len(feature_cols)),
        "embedding_dimension": int(embeddings.shape[1]),
        "repeated_cv": {
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "folds_per_representation": N_SPLITS * N_REPEATS,
            "threshold": 0.50,
        },
        "comparison": comparison_rows,
        "caveats": [
            "The locked test set contains only 19 records.",
            "Repeated CV is a secondary robustness estimate, not a replacement for an external test set.",
            "Transformer embeddings are frozen; this experiment does not fine-tune the encoder.",
            "Mechanism-only structured features exclude actual_* and potential_* consequence features.",
        ],
    }

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)
    print("\nRepeated CV summary:")
    print(cv_summary.to_string(index=False))


if __name__ == "__main__":
    main()
