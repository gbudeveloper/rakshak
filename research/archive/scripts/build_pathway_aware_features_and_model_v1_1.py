"""
RAKSHAK Pathway-Aware SIF Model v1.1
====================================

Purpose
-------
The v1.0 conflict audit showed several positive SIF cases with strong
narrative hazard -> exposure -> pathway signals but low model probability.
This experiment adds explicit pathway-level interactions to the existing
narrative-only precursor representation.

Important:
- Uses report_id + description only.
- Does NOT use SIF labels as features.
- Does NOT use actual outcomes, potential consequences, accident level,
  annotation notes, or LSR annotations.
- Locked validation/test split remains unchanged.
- Repeated CV remains duplicate-group-safe.

Compared representations:
  A) transformer_only
  B) precursor_structured_v0.2
  C) pathway_aware_structured
  D) transformer_plus_pathway_aware

Pathway features
----------------
- hazard signal count
- exposure signal count
- pathway signal count
- hazard x exposure
- hazard x pathway
- exposure x pathway
- complete pathway (hazard + exposure + pathway)
- mechanism-specific hazard/exposure interactions
- pathway category indicators

This is a research experiment. It does not produce a production safety
decision by itself.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import random

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
BASE_FEATURE_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "sif_precursor_feature_builder_v0.2"
    / "sif_precursor_features.csv"
)
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "sif_splits_v0.2"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "sif_pathway_aware_model_v1.1"
)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

RANDOM_STATE = 42
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 256
N_SPLITS = 5
N_REPEATS = 10
C_VALUES = (0.10, 0.25, 0.50, 1.00)


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

    # Mechanism-consistency interactions.
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


def metrics(y_true, prob, threshold=0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
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


def load() -> tuple[pd.DataFrame, dict[str, set[str]]]:
    labels = pd.read_csv(LABEL_PATH)
    labels["report_id"] = labels["report_id"].astype(str)
    labels["final_sif_potential"] = (
        labels["final_sif_potential"].astype(str).str.strip().str.upper()
    )
    labels = labels[labels["final_sif_potential"].isin(["YES", "NO"])].copy()
    labels["label"] = (labels["final_sif_potential"] == "YES").astype(int)

    features = pd.read_csv(BASE_FEATURE_PATH)
    features["report_id"] = features["report_id"].astype(str)

    merged = labels[["report_id", "description", "label"]].merge(
        features,
        on="report_id",
        how="inner",
        validate="one_to_one",
    )

    # Build new pathway features directly from narrative.
    path_rows = [
        build_pathway_features(text)
        for text in merged["description"].fillna("").astype(str)
    ]
    pathway = pd.DataFrame(path_rows)
    merged = pd.concat(
        [merged.reset_index(drop=True), pathway.reset_index(drop=True)],
        axis=1,
    )

    splits = {}
    for name in ("train", "validation", "test"):
        frame = pd.read_csv(SPLIT_DIR / f"{name}.csv")
        splits[name] = set(frame["report_id"].astype(str))

    return merged.reset_index(drop=True), splits


def embed(texts: list[str]) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH

    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
        device=device,
    ).astype(np.float32)


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


def split_indices(df: pd.DataFrame, splits: dict[str, set[str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup = {rid: i for i, rid in enumerate(df["report_id"].astype(str))}
    tr = np.array([lookup[r] for r in sorted(splits["train"])])
    va = np.array([lookup[r] for r in sorted(splits["validation"])])
    te = np.array([lookup[r] for r in sorted(splits["test"])])
    return tr, va, te


def group_ids_from_splits(
    df: pd.DataFrame,
    splits: dict[str, set[str]],
) -> np.ndarray:
    groups = np.empty(len(df), dtype=object)

    for name in ("train", "validation", "test"):
        frame = pd.read_csv(SPLIT_DIR / f"{name}.csv")
        lookup = dict(
            zip(
                frame["report_id"].astype(str),
                frame["duplicate_group_id"].fillna("").astype(str),
            )
        )
        for i, rid in enumerate(df["report_id"].astype(str)):
            if rid in lookup:
                groups[i] = lookup[rid]

    if np.any(pd.isna(groups)):
        raise ValueError("Some rows are missing duplicate_group_id.")

    return groups


def run_cv(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C: float,
) -> pd.DataFrame:
    rows = []

    for repeat in range(N_REPEATS):
        cv = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + repeat,
        )

        for fold, (tr, te) in enumerate(cv.split(X, y, groups), start=1):
            model = make_model(C)
            model.fit(X[tr], y[tr])

            prob = model.predict_proba(X[te])[:, 1]
            m = metrics(y[te], prob, 0.50)

            rows.append({
                "representation": name,
                "C": C,
                "repeat": repeat + 1,
                "fold": fold,
                **m,
            })

    return pd.DataFrame(rows)


def main() -> None:
    seed_everything()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, splits = load()
    y = df["label"].to_numpy(int)

    tr, va, te = split_indices(df, splits)

    print(f"Records: {len(df)} | YES={int(y.sum())} | NO={int((y==0).sum())}")
    print(
        f"Train/Val/Test: {len(tr)}/{len(va)}/{len(te)}"
    )

    # Base structured features.
    feature_cols = [
        c for c in pd.read_csv(BASE_FEATURE_PATH, nrows=1).columns
        if c != "report_id"
    ]
    base_struct = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )

    path_cols = [c for c in df.columns if c.startswith("path_") or c.startswith("pathway_")]
    path_struct = (
        df[path_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )

    pathway_only = path_struct
    pathway_aware = np.hstack([base_struct, path_struct])

    print(f"Base precursor features : {base_struct.shape[1]}")
    print(f"Pathway features        : {path_struct.shape[1]}")
    print(f"Pathway-aware total     : {pathway_aware.shape[1]}")

    texts = df["description"].fillna("").astype(str).tolist()
    embeddings = embed(texts)
    np.save(OUTPUT_DIR / "transformer_embeddings.npy", embeddings)

    representations = {
        "precursor_structured": base_struct,
        "pathway_aware_structured": pathway_aware,
        "transformer_plus_pathway": np.hstack([embeddings, pathway_aware]),
    }

    # Locked split: select C using validation only.
    locked = []

    for name, X in representations.items():
        best = None

        for C in C_VALUES:
            x_train = X[tr]
            x_val = X[va]
            x_test = X[te]

            if "structured" in name:
                # Drop all-missing structured columns using training partition.
                # Embeddings have no missing values.
                if name == "transformer_plus_pathway":
                    struct_part = x_train[:, embeddings.shape[1]:]
                    keep_struct = ~np.all(np.isnan(struct_part), axis=0)
                    keep = np.concatenate([
                        np.ones(embeddings.shape[1], dtype=bool),
                        keep_struct,
                    ])
                else:
                    keep = ~np.all(np.isnan(x_train), axis=0)

                x_train = x_train[:, keep]
                x_val = x_val[:, keep]
                x_test = x_test[:, keep]

            model = make_model(C)
            model.fit(x_train, y[tr])

            val_prob = model.predict_proba(x_val)[:, 1]
            val_metrics = metrics(y[va], val_prob, 0.50)

            candidate = {
                "representation": name,
                "C": C,
                "validation_f2": val_metrics["f2"],
                "validation_pr_auc": val_metrics["pr_auc"],
                "validation_recall": val_metrics["recall"],
            }

            if best is None or (
                candidate["validation_f2"],
                candidate["validation_recall"],
                candidate["validation_pr_auc"],
            ) > (
                best["validation_f2"],
                best["validation_recall"],
                best["validation_pr_auc"],
            ):
                best = candidate

        C = best["C"]
        x_train = X[tr]
        x_val = X[va]
        x_test = X[te]

        if "structured" in name:
            if name == "transformer_plus_pathway":
                struct_part = x_train[:, embeddings.shape[1]:]
                keep_struct = ~np.all(np.isnan(struct_part), axis=0)
                keep = np.concatenate([
                    np.ones(embeddings.shape[1], dtype=bool),
                    keep_struct,
                ])
            else:
                keep = ~np.all(np.isnan(x_train), axis=0)
            x_train = x_train[:, keep]
            x_val = x_val[:, keep]
            x_test = x_test[:, keep]

        model = make_model(C)
        model.fit(x_train, y[tr])

        test_prob = model.predict_proba(x_test)[:, 1]

        locked.append({
            **best,
            "locked_test": metrics(y[te], test_prob, 0.50),
            "features_used": int(x_train.shape[1]),
        })

        joblib.dump(
            {"model": model, "representation": name, "C": C},
            OUTPUT_DIR / f"{name}_model.joblib",
        )

    # Group-safe repeated CV for selected C=.50 baseline and pathway models.
    groups = group_ids_from_splits(df, splits)

    cv_frames = []
    for item in locked:
        name = item["representation"]
        C = item["C"]
        X = (
            representations[name]
            if name in representations
            else representations["pathway_aware_structured"]
        )

        print(f"Running group-safe repeated CV: {name} (C={C})")
        cv_frames.append(run_cv(name, X, y, groups, C))

    cv_results = pd.concat(cv_frames, ignore_index=True)

    summary_rows = []
    for name, g in cv_results.groupby("representation"):
        row = {"representation": name, "C": float(g["C"].iloc[0]), "folds": len(g)}
        for metric_name in ("accuracy", "precision", "recall", "f2", "pr_auc", "fp", "fn"):
            row[f"{metric_name}_mean"] = float(g[metric_name].mean())
            row[f"{metric_name}_std"] = float(g[metric_name].std(ddof=1))
        summary_rows.append(row)

    cv_summary = pd.DataFrame(summary_rows)

    pd.DataFrame(locked).to_json(
        OUTPUT_DIR / "locked_comparison.json",
        orient="records",
        indent=2,
    )
    pd.DataFrame(locked).to_csv(
        OUTPUT_DIR / "locked_comparison.csv",
        index=False,
    )
    cv_results.to_csv(
        OUTPUT_DIR / "repeated_group_cv_results.csv",
        index=False,
    )
    cv_summary.to_csv(
        OUTPUT_DIR / "repeated_group_cv_summary.csv",
        index=False,
    )

    feature_manifest = {
        "base_feature_count": int(base_struct.shape[1]),
        "pathway_feature_count": int(path_struct.shape[1]),
        "pathway_feature_names": path_cols,
        "source_policy": "report_id + description only",
        "excluded": [
            "sif labels",
            "actual outcomes",
            "potential consequences",
            "potential accident level",
            "annotation notes",
            "LSR labels",
        ],
    }
    (OUTPUT_DIR / "feature_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2),
        encoding="utf-8",
    )

    report = {
        "experiment": "RAKSHAK Pathway-Aware SIF Model v1.1",
        "dataset_records": int(len(df)),
        "train_validation_test": [int(len(tr)), int(len(va)), int(len(te))],
        "representations": [x["representation"] for x in locked],
        "locked_results": locked,
        "repeated_group_cv_summary": summary_rows,
        "interpretation_note": (
            "Pathway features are narrative-only heuristic signals. "
            "They do not establish SIF status and do not replace human adjudication."
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("LOCKED COMPARISON")
    print("=" * 78)
    for item in locked:
        lt = item["locked_test"]
        print(
            f"{item['representation']:<28} "
            f"C={item['C']:<4} "
            f"Val F2={item['validation_f2']:.3f} "
            f"Test F2={lt['f2']:.3f} "
            f"Test PR-AUC={lt['pr_auc']:.3f} "
            f"Test Recall={lt['recall']:.3f}"
        )

    print()
    print("REPEATED GROUP-SAFE CV")
    print(cv_summary.to_string(index=False))
    print()
    print(f"Saved outputs to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
