"""
RAKSHAK Hybrid Structured + Transformer SIF Model v0.3
=======================================================

Clean experiment after v0.2 leakage investigation.

Structured features are narrative-only and target-independent.
Repeated CV is duplicate-group-safe via StratifiedGroupKFold.

Representations:
1) transformer_only
2) precursor_structured_only
3) transformer_plus_precursor

Locked split:
data/processed/sif_splits_v0.2
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "sif_precursor_feature_builder_v0.2"
    / "sif_precursor_features.csv"
)
LABEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "resolved_annotations_v0.2.csv"
)
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "sif_splits_v0.2"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "sif_hybrid_structured_transformer_v0.3"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RANDOM_STATE = 42
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 8
N_SPLITS = 5
N_REPEATS = 10


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2.0, zero_division=0)),
        "pr_auc": (
            float(average_precision_score(y_true, prob))
            if len(np.unique(y_true)) == 2 else None
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_search(y: np.ndarray, prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.linspace(0.20, 0.80, 121):
        rows.append({
            "threshold": float(threshold),
            **metrics(y, prob, float(threshold)),
        })
    table = pd.DataFrame(rows)

    candidates = table[table["precision"] >= 0.50]
    if candidates.empty:
        candidates = table

    best = candidates.sort_values(
        ["f2", "recall", "precision", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return float(best["threshold"]), table


def load_data() -> tuple[pd.DataFrame, dict[str, set[str]]]:
    labels = pd.read_csv(LABEL_PATH)
    features = pd.read_csv(FEATURE_PATH)

    labels["report_id"] = labels["report_id"].astype(str)
    labels["final_sif_potential"] = (
        labels["final_sif_potential"].astype(str).str.strip().str.upper()
    )
    labels = labels[labels["final_sif_potential"].isin(["YES", "NO"])].copy()
    labels["label"] = (labels["final_sif_potential"] == "YES").astype(int)

    features["report_id"] = features["report_id"].astype(str)

    merged = labels[["report_id", "description", "label"]].merge(
        features,
        on="report_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(labels):
        raise ValueError("Feature/label merge is incomplete.")

    splits: dict[str, set[str]] = {}
    for name in ("train", "validation", "test"):
        path = SPLIT_DIR / f"{name}.csv"
        frame = pd.read_csv(path)
        splits[name] = set(frame["report_id"].astype(str))

    # Strong locked-split checks.
    train_ids, val_ids, test_ids = splits["train"], splits["validation"], splits["test"]
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Locked split has report_id overlap.")

    return merged, splits


def fit_model(mode: str) -> Pipeline:
    # Pure L1 for structured-only; L2 for embedding/hybrid.
    l1_ratio = 1.0 if mode == "precursor_structured_only" else 0.0
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            solver="saga",
            l1_ratio=l1_ratio,
            C=0.50,
            class_weight="balanced",
            max_iter=10000,
            random_state=RANDOM_STATE,
        )),
    ])


def drop_all_missing_from_train(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keep = ~np.all(np.isnan(x_train), axis=0)
    if not np.any(keep):
        raise ValueError("All structured features are missing in training.")
    return x_train[:, keep], x_val[:, keep], x_test[:, keep], keep


def embed_all(texts: list[str]) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(MODEL_NAME, device=device)
    encoder.max_seq_length = MAX_SEQ_LENGTH
    return encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
        device=device,
    ).astype(np.float32)


def build_group_series(
    merged: pd.DataFrame,
    splits: dict[str, set[str]],
    split_name: str,
) -> np.ndarray:
    """
    Duplicate group IDs are taken from the current split CSV.
    A split CSV is expected to contain duplicate_group_id.
    """
    path = SPLIT_DIR / f"{split_name}.csv"
    frame = pd.read_csv(path)
    if "duplicate_group_id" not in frame.columns:
        raise ValueError(
            f"{path} lacks duplicate_group_id. "
            "Re-run the fixed group-safe dataset splitter first."
        )
    lookup = dict(
        zip(frame["report_id"].astype(str), frame["duplicate_group_id"].fillna("").astype(str))
    )
    return np.array([lookup.get(rid, f"single::{rid}") for rid in merged["report_id"]], dtype=object)


def repeated_group_cv(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    rows = []

    # 10 distinct randomized group-safe repetitions.
    for repeat in range(N_REPEATS):
        cv = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + repeat,
        )
        for fold, (tr, te) in enumerate(cv.split(X, y, groups), start=1):
            x_tr = X[tr]
            x_te = X[te]

            if name == "precursor_structured_only" or name == "transformer_plus_precursor":
                keep = ~np.all(np.isnan(x_tr), axis=0)
                x_tr = x_tr[:, keep]
                x_te = x_te[:, keep]

            model = fit_model(name)
            model.fit(x_tr, y[tr])
            prob = model.predict_proba(x_te)[:, 1]
            m = metrics(y[te], prob, 0.50)

            rows.append({
                "representation": name,
                "repeat": repeat + 1,
                "fold": fold,
                **m,
            })

    return pd.DataFrame(rows)


def main() -> None:
    seed_everything()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    merged, splits = load_data()

    print(f"Records: {len(merged)} | YES={int(merged['label'].sum())} | NO={int((merged['label']==0).sum())}")
    print(
        f"Split sizes: train={len(splits['train'])} | "
        f"validation={len(splits['validation'])} | test={len(splits['test'])}"
    )

    id_to_idx = {rid: i for i, rid in enumerate(merged["report_id"].astype(str))}
    train_idx = np.array([id_to_idx[r] for r in sorted(splits["train"])])
    val_idx = np.array([id_to_idx[r] for r in sorted(splits["validation"])])
    test_idx = np.array([id_to_idx[r] for r in sorted(splits["test"])])

    y = merged["label"].to_numpy(dtype=int)

    structured_cols = [c for c in pd.read_csv(FEATURE_PATH, nrows=1).columns if c != "report_id"]
    X_struct = merged[structured_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    texts = merged["description"].fillna("").astype(str).tolist()

    print(f"Loading encoder: {MODEL_NAME}")
    embeddings = embed_all(texts)
    np.save(OUTPUT_DIR / "transformer_embeddings.npy", embeddings)

    representations = {
        "transformer_only": embeddings,
        "precursor_structured_only": X_struct,
        "transformer_plus_precursor": np.hstack([embeddings, X_struct]),
    }

    # Locked validation/test comparison.
    comparisons = []
    thresholds = []
    predictions = []

    for name, X in representations.items():
        x_tr, x_val, x_te = X[train_idx], X[val_idx], X[test_idx]

        if name != "transformer_only":
            x_tr, x_val, x_te, keep = drop_all_missing_from_train(x_tr, x_val, x_te)
            used_features = int(keep.sum())
        else:
            used_features = int(x_tr.shape[1])

        model = fit_model(name)
        model.fit(x_tr, y[train_idx])

        val_prob = model.predict_proba(x_val)[:, 1]
        threshold, threshold_table = threshold_search(y[val_idx], val_prob)
        test_prob = model.predict_proba(x_te)[:, 1]

        comparisons.append({
            "representation": name,
            "validation_threshold": threshold,
            "validation": metrics(y[val_idx], val_prob, threshold),
            "locked_test_selected_threshold": metrics(y[test_idx], test_prob, threshold),
            "locked_test_0.50": metrics(y[test_idx], test_prob, 0.50),
            "features_used": used_features,
        })

        threshold_table.insert(0, "representation", name)
        thresholds.append(threshold_table)

        for rid, true_y, p in zip(merged.iloc[test_idx]["report_id"], y[test_idx], test_prob):
            predictions.append({
                "representation": name,
                "report_id": rid,
                "true_label": int(true_y),
                "probability": float(p),
                "prediction_0.50": int(p >= 0.50),
                "prediction_selected_threshold": int(p >= threshold),
            })

        # Save model package with feature-mask metadata.
        joblib.dump(
            {"model": model, "representation": name},
            OUTPUT_DIR / f"{name}_model.joblib",
        )

        print(f"\n=== {name} ===")
        print(json.dumps(comparisons[-1], indent=2))

    # Duplicate-group-safe repeated CV over the full 70-row labelled pool.
    groups = np.empty(len(merged), dtype=object)
    for split_name in ("train", "validation", "test"):
        path = SPLIT_DIR / f"{split_name}.csv"
        frame = pd.read_csv(path)
        lookup = dict(
            zip(
                frame["report_id"].astype(str),
                frame["duplicate_group_id"].fillna("").astype(str),
            )
        )
        for i, rid in enumerate(merged["report_id"].astype(str)):
            if rid in lookup:
                groups[i] = lookup[rid]

    if np.any(pd.isna(groups)):
        raise ValueError("Some records lack duplicate_group_id for group-safe CV.")

    cv_frames = []
    for name, X in representations.items():
        print(f"\nRunning group-safe repeated CV: {name}")
        cv_frames.append(repeated_group_cv(name, X, y, groups))

    cv_results = pd.concat(cv_frames, ignore_index=True)
    numeric = ["accuracy", "precision", "recall", "f2", "pr_auc", "fp", "fn"]

    rows = []
    for name, g in cv_results.groupby("representation"):
        row = {"representation": name, "folds": len(g)}
        for metric_name in numeric:
            vals = g[metric_name].astype(float)
            row[f"{metric_name}_mean"] = float(vals.mean())
            row[f"{metric_name}_std"] = float(vals.std(ddof=1))
        rows.append(row)
    cv_summary = pd.DataFrame(rows)

    pd.DataFrame(comparisons).to_json(
        OUTPUT_DIR / "representation_comparison.json",
        orient="records",
        indent=2,
    )
    pd.DataFrame(comparisons).to_csv(
        OUTPUT_DIR / "representation_comparison.csv",
        index=False,
    )
    pd.concat(thresholds, ignore_index=True).to_csv(
        OUTPUT_DIR / "validation_thresholds.csv",
        index=False,
    )
    pd.DataFrame(predictions).to_csv(
        OUTPUT_DIR / "locked_test_predictions.csv",
        index=False,
    )
    cv_results.to_csv(OUTPUT_DIR / "repeated_group_cv_results.csv", index=False)
    cv_summary.to_csv(OUTPUT_DIR / "repeated_group_cv_summary.csv", index=False)

    summary = {
        "experiment": "RAKSHAK Hybrid Structured + Transformer SIF Model v0.3",
        "dataset_size": int(len(merged)),
        "yes_count": int(y.sum()),
        "no_count": int((y == 0).sum()),
        "split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "structured_feature_count": int(X_struct.shape[1]),
        "embedding_dimension": int(embeddings.shape[1]),
        "repeated_group_cv": {
            "splits": N_SPLITS,
            "repeats": N_REPEATS,
            "folds_per_representation": N_SPLITS * N_REPEATS,
            "group_safe": True,
            "threshold": 0.50,
        },
        "comparison": comparisons,
        "critical_note": (
            "Do not interpret perfect v0.2 structured scores as model performance. "
            "v0.3 uses narrative-only precursor features and duplicate-group-safe CV."
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)
    print("\nRepeated group-safe CV summary:")
    print(cv_summary.to_string(index=False))


if __name__ == "__main__":
    main()
