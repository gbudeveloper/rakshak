"""
RAKSHAK Final Research Pipeline v1.2
====================================

Purpose
-------
Freeze the current 70-record research dataset and provide one reproducible,
leakage-aware training/evaluation pipeline for the strongest current
representation:

    Transformer embedding + narrative pathway-aware features

The pipeline:
1. Loads the fixed adjudicated labels.
2. Loads the fixed duplicate-group-safe train/validation/test split.
3. Rebuilds pathway features directly from description only.
4. Drops all-missing structured columns using training data only.
5. Encodes narratives with the same SentenceTransformer model.
6. Selects C on validation only.
7. Runs duplicate-group-safe repeated CV for the selected C.
8. Produces locked-test predictions.
9. Computes bootstrap confidence intervals on the locked test set
   (diagnostic only; not a substitute for a larger external test set).
10. Fits the final model on train+validation using the preselected C.
11. Saves a single model package and structured incident explanations.

Important
---------
- No SIF label is used as an input feature.
- No actual outcome, potential consequence, potential accident level,
  annotation note, or LSR annotation is consumed as a feature.
- Test remains locked during model/C selection.
- Confidence intervals on n=20 are descriptive and should be treated
  cautiously.
- This is a research/triage component, not an autonomous safety decision
  system.
"""

from __future__ import annotations

import json
import random
import re
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

LABEL_PATH = PROJECT_ROOT / "data" / "annotations" / "resolved_annotations_v0.2.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "sif_splits_v0.2"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "rakshak_final_v1.2"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

RANDOM_STATE = 42
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 256

N_SPLITS = 5
N_REPEATS = 10

C_GRID = (0.10, 0.25, 0.50, 1.00)

# Fixed operational reference threshold. We do not optimize a threshold
# aggressively on the tiny validation set.
REFERENCE_THRESHOLD = 0.50

BOOTSTRAP_REPS = 5000


HAZARD_PATTERNS = {
    "electrical": [r"\belectric", r"\benergized\b", r"\bvoltage\b", r"\bpanel\b", r"\bcable\b"],
    "mechanical": [r"\bmachin", r"\bconveyor\b", r"\broller\b", r"\brotat", r"\bmoving part", r"\bpress\b"],
    "pressure": [r"\bpressur", r"\bcompressed\b", r"\bstored energy\b", r"\bpressure\b"],
    "chemical": [r"\bammonia\b", r"\bchlorine\b", r"\bchemical\b", r"\bsolvent\b", r"\bacid\b", r"\bcaustic\b", r"\btoxic\b", r"\bcorrosive\b", r"\bleak", r"\bspill", r"\brelease\b"],
    "thermal": [r"\bsteam\b", r"\bhot\b", r"\bheat\b", r"\bthermal\b"],
    "fire_explosion": [r"\bfire\b", r"\bexplos", r"\bflammab", r"\bignition\b"],
    "vehicle": [r"\bvehicle\b", r"\btruck\b", r"\bforklift\b", r"\bcrane\b", r"\bexcavat", r"\bloader\b", r"\bdriv(?:e|er|ing)\b"],
    "fall_height": [r"\bladder\b", r"\bscaffold", r"\bplatform\b", r"\bheight\b", r"\belevated\b", r"\bfell\b", r"\bfall\b"],
    "caught_between": [r"\bpinch\b", r"\bcaught\b", r"\btrapped\b", r"\bbetween\b"],
    "dropped_object": [r"\bdropped\b", r"\bfalling object\b", r"\btool\b.*\bfall", r"\bmaterial\b.*\bfall"],
}

EXPOSURE_PATTERNS = {
    "direct_contact": [r"\bcontact(?:ed|ing)?\b", r"\btouched\b", r"\bhand\b.*\b(?:near|on|against)\b"],
    "line_of_fire": [r"\bline of fire\b", r"\bin the path\b", r"\btrajectory\b", r"\bstruck by\b", r"\bstruck\b"],
    "caught_between": [r"\bpinch\b", r"\bcaught\b", r"\btrapped\b", r"\bbetween\b"],
    "fall": [r"\bfell\b", r"\bfall\b", r"\bdropped from\b"],
    "chemical": [r"\bsplash\b", r"\bleak", r"\bspill", r"\bexpos(?:e|ed|ure)\b", r"\binhal", r"\bfume", r"\bgas\b", r"\bvapou?r\b"],
    "thermal": [r"\bhot\b", r"\bsteam\b", r"\bheat\b", r"\bburn"],
    "electrical": [r"\belectric", r"\benergized\b", r"\bvoltage\b", r"\bshock\b"],
    "vehicle": [r"\bvehicle\b", r"\btruck\b", r"\bforklift\b", r"\bdriver\b", r"\bpedestrian\b"],
}

PATHWAY_PATTERNS = {
    "struck_by_projectile": [r"\bstruck by\b", r"\bstruck\b", r"\bproject(?:ed|ile)\b", r"\beject(?:ed|ion)\b", r"\bthrown\b"],
    "caught_crush": [r"\bpinch\b", r"\bcaught\b", r"\btrapped\b", r"\bcrush"],
    "fall_from_height": [r"\bladder\b", r"\bscaffold", r"\bplatform\b", r"\bheight\b", r"\belevated\b", r"\bfell\b"],
    "electrical_contact": [r"\belectric", r"\benergized\b", r"\bvoltage\b", r"\bshock\b", r"\bcontact\b"],
    "chemical_release_exposure": [r"\bleak", r"\bspill", r"\brelease", r"\bsplash", r"\binhal", r"\bfume", r"\bgas\b", r"\bvapou?r\b"],
    "thermal_contact": [r"\bhot\b", r"\bsteam\b", r"\bheat\b", r"\bthermal\b"],
    "vehicle_person_collision": [r"\bvehicle\b.*\bperson\b", r"\bperson\b.*\bvehicle\b", r"\bpedestrian\b", r"\bstruck\b.*\bvehicle\b"],
}


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def has_any(text: str, patterns: list[str]) -> int:
    return int(any(re.search(p, text, flags=re.I) for p in patterns))


def build_pathway_features(description: str) -> dict[str, float]:
    text = norm(description)
    f: dict[str, float] = {}

    hazard_hits = {}
    exposure_hits = {}
    pathway_hits = {}

    for name, pats in HAZARD_PATTERNS.items():
        value = has_any(text, pats)
        hazard_hits[name] = value
        f[f"path_hazard_{name}"] = value

    for name, pats in EXPOSURE_PATTERNS.items():
        value = has_any(text, pats)
        exposure_hits[name] = value
        f[f"path_exposure_{name}"] = value

    for name, pats in PATHWAY_PATTERNS.items():
        value = has_any(text, pats)
        pathway_hits[name] = value
        f[f"pathway_{name}"] = value

    hazard_count = sum(hazard_hits.values())
    exposure_count = sum(exposure_hits.values())
    pathway_count = sum(pathway_hits.values())

    f["path_hazard_count"] = hazard_count
    f["path_exposure_count"] = exposure_count
    f["pathway_count"] = pathway_count

    f["path_hazard_exposure"] = int(hazard_count > 0 and exposure_count > 0)
    f["path_hazard_pathway"] = int(hazard_count > 0 and pathway_count > 0)
    f["path_exposure_pathway"] = int(exposure_count > 0 and pathway_count > 0)
    f["path_complete"] = int(
        hazard_count > 0 and exposure_count > 0 and pathway_count > 0
    )

    f["path_electrical_complete"] = int(
        hazard_hits["electrical"]
        and exposure_hits["electrical"]
        and pathway_hits["electrical_contact"]
    )
    f["path_chemical_complete"] = int(
        hazard_hits["chemical"]
        and exposure_hits["chemical"]
        and pathway_hits["chemical_release_exposure"]
    )
    f["path_fall_complete"] = int(
        hazard_hits["fall_height"]
        and exposure_hits["fall"]
        and pathway_hits["fall_from_height"]
    )
    f["path_vehicle_complete"] = int(
        hazard_hits["vehicle"]
        and exposure_hits["vehicle"]
        and pathway_hits["vehicle_person_collision"]
    )
    f["path_caught_complete"] = int(
        hazard_hits["caught_between"]
        and exposure_hits["caught_between"]
        and pathway_hits["caught_crush"]
    )
    f["path_line_fire"] = int(
        exposure_hits["line_of_fire"]
        and (
            pathway_hits["struck_by_projectile"]
            or hazard_hits["vehicle"]
            or hazard_hits["dropped_object"]
        )
    )

    return f


def load_dataset() -> tuple[pd.DataFrame, dict[str, set[str]], np.ndarray]:
    labels = pd.read_csv(LABEL_PATH)
    labels["report_id"] = labels["report_id"].astype(str)
    labels["description"] = labels["description"].fillna("").astype(str)
    labels["final_sif_potential"] = (
        labels["final_sif_potential"].astype(str).str.strip().str.upper()
    )
    labels = labels[labels["final_sif_potential"].isin(["YES", "NO"])].copy()
    labels["label"] = (labels["final_sif_potential"] == "YES").astype(int)

    split_ids = {}
    split_frames = {}

    for name in ("train", "validation", "test"):
        frame = pd.read_csv(SPLIT_DIR / f"{name}.csv")
        frame["report_id"] = frame["report_id"].astype(str)
        split_frames[name] = frame
        split_ids[name] = set(frame["report_id"])

    train_ids, val_ids, test_ids = (
        split_ids["train"],
        split_ids["validation"],
        split_ids["test"],
    )

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Locked split contains report-ID overlap.")

    merged = labels[
        ["report_id", "description", "label", "final_sif_potential"]
    ].copy()

    # Ensure every labelled record belongs to one split.
    split_union = train_ids | val_ids | test_ids
    missing = set(merged["report_id"]) - split_union
    if missing:
        raise ValueError(f"Labelled records missing from split manifest: {sorted(missing)}")

    # Group lookup for duplicate-safe CV.
    group_lookup = {}
    for frame in split_frames.values():
        if "duplicate_group_id" not in frame.columns:
            raise ValueError("Split file lacks duplicate_group_id.")
        group_lookup.update(
            dict(
                zip(
                    frame["report_id"],
                    frame["duplicate_group_id"].fillna("").astype(str),
                )
            )
        )

    groups = np.array(
        [group_lookup[rid] for rid in merged["report_id"]],
        dtype=object,
    )

    return merged.reset_index(drop=True), split_ids, groups


def load_or_create_embeddings(texts: list[str]) -> np.ndarray:
    embedding_path = OUTPUT_DIR / "all_embeddings.npy"

    if embedding_path.exists():
        emb = np.load(embedding_path)
        if emb.shape[0] == len(texts):
            return emb.astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading encoder: {MODEL_NAME}")
    encoder = SentenceTransformer(MODEL_NAME, device=device)
    encoder.max_seq_length = MAX_SEQ_LENGTH

    emb = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
        device=device,
    ).astype(np.float32)

    np.save(embedding_path, emb)
    return emb


def make_model(C: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            solver="saga",
            l1_ratio=0.0,
            C=C,
            class_weight="balanced",
            max_iter=10000,
            random_state=RANDOM_STATE,
        )),
    ])


def build_representation_table(
    df: pd.DataFrame,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    base_feature_dicts = [
        build_pathway_features(text)
        for text in df["description"].tolist()
    ]
    path_df = pd.DataFrame(base_feature_dicts, index=df.index)

    # Only the pathway representation is used in the final model.
    X_struct = path_df.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    X = np.hstack([embeddings, X_struct])

    feature_names = (
        [f"embedding_{i:03d}" for i in range(embeddings.shape[1])]
        + path_df.columns.tolist()
    )
    return X, feature_names


def drop_structured_all_missing(
    X_train: np.ndarray,
    X_other: list[np.ndarray],
    embedding_dim: int,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    struct = X_train[:, embedding_dim:]
    keep_struct = ~np.all(np.isnan(struct), axis=0)
    keep = np.concatenate([
        np.ones(embedding_dim, dtype=bool),
        keep_struct,
    ])
    return (
        X_train[:, keep],
        [x[:, keep] for x in X_other],
        keep,
    )


def choose_c_on_validation(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    embedding_dim: int,
) -> tuple[float, pd.DataFrame]:
    rows = []

    for C in C_GRID:
        x_train, [x_val], keep = drop_structured_all_missing(
            X[train_idx],
            [X[val_idx]],
            embedding_dim,
        )

        model = make_model(C)
        model.fit(x_train, y[train_idx])
        val_prob = model.predict_proba(x_val)[:, 1]
        m = metrics(y[val_idx], val_prob, REFERENCE_THRESHOLD)

        rows.append({
            "C": C,
            **m,
            "structured_features_used": int(keep.sum() - embedding_dim),
        })

    table = pd.DataFrame(rows)

    best = table.sort_values(
        ["f2", "recall", "pr_auc", "precision", "C"],
        ascending=[False, False, False, False, True],
    ).iloc[0]

    return float(best["C"]), table


def metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float = REFERENCE_THRESHOLD,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2.0, zero_division=0)),
        "pr_auc": (
            float(average_precision_score(y_true, prob))
            if len(np.unique(y_true)) == 2
            else None
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def repeated_group_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C: float,
    embedding_dim: int,
) -> pd.DataFrame:
    rows = []

    for repeat in range(N_REPEATS):
        cv = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + repeat,
        )

        for fold, (tr, te) in enumerate(cv.split(X, y, groups), start=1):
            x_tr, [x_te], keep = drop_structured_all_missing(
                X[tr],
                [X[te]],
                embedding_dim,
            )

            model = make_model(C)
            model.fit(x_tr, y[tr])
            prob = model.predict_proba(x_te)[:, 1]
            m = metrics(y[te], prob, REFERENCE_THRESHOLD)

            rows.append({
                "repeat": repeat + 1,
                "fold": fold,
                "C": C,
                **m,
            })

    return pd.DataFrame(rows)


def bootstrap_metric_ci(
    y: np.ndarray,
    prob: np.ndarray,
    metric_name: str,
    threshold: float = REFERENCE_THRESHOLD,
    reps: int = BOOTSTRAP_REPS,
) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y)
    values = []

    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        pb = prob[idx]

        # Require both classes where needed.
        if metric_name == "pr_auc" and len(np.unique(yb)) < 2:
            continue

        m = metrics(yb, pb, threshold)

        if metric_name == "pr_auc":
            values.append(m["pr_auc"])
        else:
            values.append(m[metric_name])

    if not values:
        return {
            "metric": metric_name,
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "valid_bootstrap_samples": 0,
        }

    arr = np.asarray(values, dtype=float)

    return {
        "metric": metric_name,
        "estimate": float(metrics(y, prob, threshold)[metric_name]),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "valid_bootstrap_samples": int(len(arr)),
    }


def extract_evidence(description: str) -> dict:
    text = norm(description)

    matched_hazards = []
    matched_exposures = []
    matched_pathways = []

    for name, patterns in HAZARD_PATTERNS.items():
        terms = [p for p in patterns if re.search(p, text, flags=re.I)]
        if terms:
            matched_hazards.append(name)

    for name, patterns in EXPOSURE_PATTERNS.items():
        terms = [p for p in patterns if re.search(p, text, flags=re.I)]
        if terms:
            matched_exposures.append(name)

    for name, patterns in PATHWAY_PATTERNS.items():
        terms = [p for p in patterns if re.search(p, text, flags=re.I)]
        if terms:
            matched_pathways.append(name)

    strength = int(bool(matched_hazards)) + int(bool(matched_exposures)) + int(bool(matched_pathways))

    if strength == 3:
        assessment = "SUPPORTED"
    elif strength == 2:
        assessment = "PARTIAL"
    elif strength == 1:
        assessment = "WEAK"
    else:
        assessment = "NONE"

    return {
        "hazards": " | ".join(sorted(matched_hazards)),
        "exposures": " | ".join(sorted(matched_exposures)),
        "pathways": " | ".join(sorted(matched_pathways)),
        "pathway_signal_strength": strength,
        "pathway_assessment": assessment,
    }


def main() -> None:
    seed_everything()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, splits, groups = load_dataset()
    y = df["label"].to_numpy(int)

    print("=" * 78)
    print("RAKSHAK FINAL RESEARCH PIPELINE v1.2")
    print("=" * 78)
    print(f"Records: {len(df)} | YES={int(y.sum())} | NO={int((y == 0).sum())}")
    print(
        f"Train/Val/Test: "
        f"{len(splits['train'])}/"
        f"{len(splits['validation'])}/"
        f"{len(splits['test'])}"
    )

    lookup = {rid: i for i, rid in enumerate(df["report_id"].astype(str))}

    train_idx = np.array([lookup[r] for r in sorted(splits["train"])])
    val_idx = np.array([lookup[r] for r in sorted(splits["validation"])])
    test_idx = np.array([lookup[r] for r in sorted(splits["test"])])

    embeddings = load_or_create_embeddings(
        df["description"].fillna("").astype(str).tolist()
    )

    X, feature_names = build_representation_table(df, embeddings)
    embedding_dim = embeddings.shape[1]

    print(f"Embedding dimension : {embedding_dim}")
    print(f"Pathway features    : {X.shape[1] - embedding_dim}")
    print(f"Combined features   : {X.shape[1]}")

    # ------------------------------------------------------------
    # Validation-only C selection.
    # ------------------------------------------------------------
    selected_c, c_table = choose_c_on_validation(
        X,
        y,
        train_idx,
        val_idx,
        embedding_dim,
    )

    print(f"\nSelected C on validation: {selected_c}")
    print(c_table.to_string(index=False))
    c_table.to_csv(
        OUTPUT_DIR / "validation_C_selection.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Locked test evaluation.
    # ------------------------------------------------------------
    x_train, [x_val, x_test], keep = drop_structured_all_missing(
        X[train_idx],
        [X[val_idx], X[test_idx]],
        embedding_dim,
    )

    model = make_model(selected_c)
    model.fit(x_train, y[train_idx])

    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    val_metrics = metrics(y[val_idx], val_prob)
    test_metrics = metrics(y[test_idx], test_prob)

    print("\nLOCKED TEST")
    print(json.dumps(test_metrics, indent=2))

    # ------------------------------------------------------------
    # Repeated duplicate-group-safe CV using selected C.
    # ------------------------------------------------------------
    print("\nRunning repeated group-safe CV...")
    cv_results = repeated_group_cv(
        X,
        y,
        groups,
        selected_c,
        embedding_dim,
    )

    cv_summary = {
        "folds": int(len(cv_results)),
        "accuracy_mean": float(cv_results["accuracy"].mean()),
        "accuracy_std": float(cv_results["accuracy"].std(ddof=1)),
        "precision_mean": float(cv_results["precision"].mean()),
        "precision_std": float(cv_results["precision"].std(ddof=1)),
        "recall_mean": float(cv_results["recall"].mean()),
        "recall_std": float(cv_results["recall"].std(ddof=1)),
        "f2_mean": float(cv_results["f2"].mean()),
        "f2_std": float(cv_results["f2"].std(ddof=1)),
        "pr_auc_mean": float(cv_results["pr_auc"].mean()),
        "pr_auc_std": float(cv_results["pr_auc"].std(ddof=1)),
        "fp_mean": float(cv_results["fp"].mean()),
        "fn_mean": float(cv_results["fn"].mean()),
    }

    print("\nREPEATED GROUP-SAFE CV")
    print(json.dumps(cv_summary, indent=2))

    cv_results.to_csv(
        OUTPUT_DIR / "repeated_group_safe_cv_results.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "repeated_group_safe_cv_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(cv_summary, f, indent=2)

    # ------------------------------------------------------------
    # Locked test bootstrap intervals.
    # ------------------------------------------------------------
    ci_rows = []
    for metric_name in ("accuracy", "precision", "recall", "f2", "pr_auc"):
        ci_rows.append(
            bootstrap_metric_ci(
                y[test_idx],
                test_prob,
                metric_name,
            )
        )

    pd.DataFrame(ci_rows).to_csv(
        OUTPUT_DIR / "locked_test_bootstrap_ci.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Test predictions + narrative evidence.
    # ------------------------------------------------------------
    predictions = []

    test_ids = df.iloc[test_idx]["report_id"].astype(str).tolist()
    test_desc = df.iloc[test_idx]["description"].astype(str).tolist()
    test_true = y[test_idx]

    for rid, desc, true_y, prob in zip(
        test_ids,
        test_desc,
        test_true,
        test_prob,
    ):
        evidence = extract_evidence(desc)
        predictions.append({
            "report_id": rid,
            "true_label": int(true_y),
            "sif_precursor_probability": float(prob),
            "prediction_at_0.50": int(prob >= REFERENCE_THRESHOLD),
            **evidence,
        })

    prediction_df = pd.DataFrame(predictions).sort_values(
        "sif_precursor_probability",
        ascending=False,
    )

    prediction_df.to_csv(
        OUTPUT_DIR / "locked_test_predictions_with_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------
    # Final model:
    # train on TRAIN + VALIDATION, test remains untouched.
    # ------------------------------------------------------------
    trainval_idx = np.concatenate([train_idx, val_idx])

    x_trainval, [x_test_final], keep_final = drop_structured_all_missing(
        X[trainval_idx],
        [X[test_idx]],
        embedding_dim,
    )

    final_model = make_model(selected_c)
    final_model.fit(
        x_trainval,
        y[trainval_idx],
    )

    final_test_prob = final_model.predict_proba(x_test_final)[:, 1]
    final_test_metrics = metrics(y[test_idx], final_test_prob)

    # Save one production/research model package.
    package = {
        "model": final_model,
        "model_name": MODEL_NAME,
        "selected_C": selected_c,
        "reference_threshold": REFERENCE_THRESHOLD,
        "embedding_dimension": embedding_dim,
        "feature_names_after_keep": [
            name for name, flag in zip(feature_names, keep_final) if flag
        ],
        "feature_keep_mask": keep_final.tolist(),
        "source_policy": {
            "inputs": ["description"],
            "excluded": [
                "SIF labels",
                "actual outcomes",
                "potential consequences",
                "potential accident level",
                "annotation notes",
                "LSR annotations",
                "near-miss / Hi-Po labels",
            ],
        },
    }

    joblib.dump(
        package,
        OUTPUT_DIR / "rakshak_final_model.joblib",
    )

    # ------------------------------------------------------------
    # Final report.
    # ------------------------------------------------------------
    report = {
        "experiment": "RAKSHAK Final Research Pipeline v1.2",
        "dataset": {
            "records": int(len(df)),
            "yes": int(y.sum()),
            "no": int((y == 0).sum()),
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "model": {
            "encoder": MODEL_NAME,
            "embedding_dimension": int(embedding_dim),
            "pathway_feature_count": int(X.shape[1] - embedding_dim),
            "selected_C": float(selected_c),
            "reference_threshold": REFERENCE_THRESHOLD,
            "final_training_records": int(len(trainval_idx)),
        },
        "validation": val_metrics,
        "locked_test": test_metrics,
        "locked_test_after_trainval_refit": final_test_metrics,
        "repeated_group_safe_cv": cv_summary,
        "bootstrap_ci_file": "locked_test_bootstrap_ci.csv",
        "model_file": "rakshak_final_model.joblib",
        "interpretation": (
            "Research/triage model only. The locked test has 20 records, so "
            "performance estimates are uncertain. Evidence and pathway fields "
            "are heuristic narrative signals and require human review."
        ),
    }

    (OUTPUT_DIR / "final_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL TRAIN+VALIDATION REFIT — LOCKED TEST")
    print(json.dumps(final_test_metrics, indent=2))
    print("\nSaved outputs to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
