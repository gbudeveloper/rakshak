"""
RAKSHAK Evidence-Aware SIF Precursor Ranking v0.4 FIXED FIXED
=================================================

Purpose
-------
Turn the current narrative-only SIF classifier into an operational ranking
pipeline that returns:
  1. SIF precursor likelihood
  2. precursor mechanism evidence
  3. observed barrier/barrier-gap evidence
  4. candidate IOGP Life-Saving Rule (LSR) mapping
  5. reviewer-oriented confidence

Important
---------
This is NOT a replacement for HSSE judgment.
The model ranks reports for review.

Inputs
------
- data/annotations/resolved_annotations_v0.2.csv
- experiments/sif_precursor_feature_builder_v0.2/sif_precursor_features.csv
- experiments/sif_hybrid_structured_transformer_v0.3/transformer_embeddings.npy
- data/processed/sif_splits_v0.2/*.csv

The structured features are narrative-only because the v0.2 feature builder
consumes report_id + description only.

Outputs
-------
experiments/sif_evidence_aware_ranking_v0.4/
  ranked_reports.csv
  evidence_spans.csv
  precursor_summary.csv
  lsr_candidates.csv
  reviewer_queue.csv
  model_scores.csv
  summary.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABEL_PATH = PROJECT_ROOT / "data" / "annotations" / "resolved_annotations_v0.2.csv"
FEATURE_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "sif_precursor_feature_builder_v0.2"
    / "sif_precursor_features.csv"
)
EMBEDDING_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "sif_hybrid_structured_transformer_v0.3"
    / "transformer_embeddings.npy"
)
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "sif_splits_v0.2"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "sif_evidence_aware_ranking_v0.4"

RANDOM_STATE = 42
C = 0.50


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def sentence_spans(text: str) -> list[str]:
    text = normalize(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?;])\s+|\n+", text)
    return [p.strip(" -•\t") for p in parts if p.strip()]


# These are intentionally mechanism / precursor terms.
# Avoid terms that directly encode injury, fatality, hospitalization, etc.
PRECURSOR_PATTERNS: dict[str, list[str]] = {
    "Electrical energy": [
        r"\belectric",
        r"\benergized\b",
        r"\bvoltage\b",
        r"\bswitchboard\b",
        r"\bpanel\b",
        r"\bcable\b",
    ],
    "Moving equipment / mechanical energy": [
        r"\bmachin",
        r"\bconveyor\b",
        r"\broller\b",
        r"\brotat",
        r"\bmoving part",
        r"\bpress\b",
    ],
    "Pressure / stored energy": [
        r"\bpressur",
        r"\bcompressed\b",
        r"\bstored energy\b",
        r"\bpressure\b",
    ],
    "Chemical / hazardous release": [
        r"\bammonia\b",
        r"\bchlorine\b",
        r"\bchemical\b",
        r"\bsolvent\b",
        r"\bacid\b",
        r"\bcaustic\b",
        r"\btoxic\b",
        r"\bcorrosive\b",
        r"\bleak(?:age|ed|ing)?\b",
        r"\bspill(?:ed|ing)?\b",
        r"\brelease(?:d)?\b",
    ],
    "Thermal energy": [
        r"\bsteam\b",
        r"\bhot\b",
        r"\bheat\b",
        r"\bthermal\b",
    ],
    "Fire / explosion energy": [
        r"\bfire\b",
        r"\bexplos",
        r"\bflammab",
        r"\bignition\b",
    ],
    "Vehicle interaction": [
        r"\bvehicle\b",
        r"\btruck\b",
        r"\bforklift\b",
        r"\bcrane\b",
        r"\bexcavat",
        r"\bloader\b",
    ],
    "Fall from height": [
        r"\bladder\b",
        r"\bscaffold",
        r"\bplatform\b",
        r"\bheight\b",
        r"\belevated\b",
        r"\bfell\b",
        r"\bfall\b",
    ],
    "Caught-between / pinch mechanism": [
        r"\bpinch\b",
        r"\bcaught\b",
        r"\btrapped\b",
        r"\bbetween\b",
    ],
    "Dropped / falling object": [
        r"\bdropped\b",
        r"\bfalling object\b",
        r"\btool\b.*\bfall",
        r"\bmaterial\b.*\bfall",
    ],
    "Line-of-fire exposure": [
        r"\bline of fire\b",
        r"\bin the path\b",
        r"\btrajectory\b",
        r"\bstruck by\b",
        r"\bstruck\b",
    ],
}


BARRIER_PATTERNS: dict[str, list[str]] = {
    "PPE gap": [
        r"\bwithout\b.*\b(?:gloves?|helmet|goggles?|face shield|ppe)\b",
        r"\bno\b.*\b(?:gloves?|helmet|goggles?|face shield|ppe)\b",
        r"\bdid not\b.*\bwear\b",
    ],
    "Guard gap": [
        r"\bwithout guard\b",
        r"\bguard\b.*\b(?:removed|missing|open)\b",
        r"\b(?:bonnet|cover)\b.*\bopen\b",
    ],
    "Isolation gap": [
        r"\bnot isolated\b",
        r"\bnot lock(?:ed)? out\b",
        r"\bwithout isolation\b",
        r"\benergized\b.*\bwork\b",
    ],
    "Fall-protection gap": [
        r"\bwithout\b.*\b(?:harness|fall protection|lifeline)\b",
        r"\bno\b.*\b(?:harness|fall protection|lifeline)\b",
    ],
    "Exclusion / barricade gap": [
        r"\bnot barricaded\b",
        r"\bnot blocked\b",
        r"\bno exclusion\b",
    ],
    "Procedure / permit signal": [
        r"\bpermit\b",
        r"\bjsa\b",
        r"\brisk assessment\b",
        r"\bprocedure\b",
        r"\bchecklist\b",
    ],
}


LSR_PATTERNS: dict[str, list[str]] = {
    "Isolation": [
        r"\benergized\b",
        r"\bnot isolated\b",
        r"\bisolat(?:e|ed|ion)\b",
        r"\block(?:ed)? out\b",
        r"\blockout\b",
    ],
    "Line of Fire": [
        r"\bline of fire\b",
        r"\bstruck by\b",
        r"\btrajectory\b",
        r"\bin the path\b",
        r"\bbetween\b",
        r"\bpinch\b",
    ],
    "Work at Height": [
        r"\bladder\b",
        r"\bscaffold",
        r"\bplatform\b",
        r"\bheight\b",
        r"\bharness\b",
        r"\blifeline\b",
    ],
    "Driving": [
        r"\bdriv(?:e|er|ing)\b",
        r"\bvehicle\b",
        r"\btruck\b",
        r"\bforklift\b",
        r"\bseat belt\b",
    ],
    "Safe Mechanical Lifting": [
        r"\bcrane\b",
        r"\blifting\b",
        r"\bload\b",
        r"\bsling\b",
        r"\blifting tool\b",
    ],
    "Hot Work": [
        r"\bhot work\b",
        r"\bwelding\b",
        r"\bcutting\b",
        r"\bignition\b",
        r"\bflammab",
    ],
    "Confined Space": [
        r"\bconfined space\b",
        r"\bvessel\b",
        r"\btank\b",
        r"\binside\b.*\b(?:tank|vessel)\b",
    ],
    "Chemical Exposure": [
        r"\bammonia\b",
        r"\bchlorine\b",
        r"\btoxic\b",
        r"\bchemical\b",
        r"\bcorrosive\b",
        r"\bgas\b.*\bexpos",
    ],
}


def matched_terms(text: str, patterns: list[str]) -> list[str]:
    hits = []
    for p in patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            hits.append(m.group(0))
    return hits


def extract_evidence(description: str) -> tuple[list[dict], list[dict], list[dict]]:
    sentences = sentence_spans(description)
    precursor_rows = []
    barrier_rows = []
    lsr_rows = []

    for idx, sent in enumerate(sentences):
        low = sent.lower()

        for mechanism, patterns in PRECURSOR_PATTERNS.items():
            hits = matched_terms(low, patterns)
            if hits:
                precursor_rows.append(
                    {
                        "sentence_index": idx,
                        "mechanism": mechanism,
                        "evidence_text": sent,
                        "matched_terms": " | ".join(sorted(set(hits))),
                    }
                )

        for barrier, patterns in BARRIER_PATTERNS.items():
            hits = matched_terms(low, patterns)
            if hits:
                barrier_rows.append(
                    {
                        "sentence_index": idx,
                        "barrier_signal": barrier,
                        "evidence_text": sent,
                        "matched_terms": " | ".join(sorted(set(hits))),
                    }
                )

        for lsr, patterns in LSR_PATTERNS.items():
            hits = matched_terms(low, patterns)
            if hits:
                lsr_rows.append(
                    {
                        "sentence_index": idx,
                        "lsr_candidate": lsr,
                        "evidence_text": sent,
                        "matched_terms": " | ".join(sorted(set(hits))),
                    }
                )

    return precursor_rows, barrier_rows, lsr_rows


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="saga",
                    l1_ratio=0.0,
                    C=C,
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def load() -> tuple[pd.DataFrame, np.ndarray, dict[str, set[str]]]:
    labels = pd.read_csv(LABEL_PATH)
    features = pd.read_csv(FEATURE_PATH)
    embeddings = np.load(EMBEDDING_PATH)

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

    if len(embeddings) != len(merged):
        # v0.3 embedding file follows the merged table order.
        # This check prevents silently pairing the wrong embeddings.
        raise ValueError(
            f"Embedding rows ({len(embeddings)}) != merged rows ({len(merged)})."
        )

    splits = {}
    for split in ("train", "validation", "test"):
        frame = pd.read_csv(SPLIT_DIR / f"{split}.csv")
        splits[split] = set(frame["report_id"].astype(str))

    return merged.reset_index(drop=True), embeddings, splits


def score_model(
    merged: pd.DataFrame,
    embeddings: np.ndarray,
    splits: dict[str, set[str]],
) -> tuple[np.ndarray, float, float]:
    feature_cols = [
        c for c in pd.read_csv(FEATURE_PATH, nrows=1).columns if c != "report_id"
    ]
    structured = (
        merged[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    )

    x_hybrid = np.hstack([embeddings, structured])

    id_to_idx = {rid: i for i, rid in enumerate(merged["report_id"])}
    tr = np.array([id_to_idx[r] for r in sorted(splits["train"])])
    va = np.array([id_to_idx[r] for r in sorted(splits["validation"])])
    te = np.array([id_to_idx[r] for r in sorted(splits["test"])])

    x_train = x_hybrid[tr]
    x_val = x_hybrid[va]
    x_test = x_hybrid[te]
    y_train = merged["label"].to_numpy()[tr]
    y_val = merged["label"].to_numpy()[va]
    y_test = merged["label"].to_numpy()[te]

    keep = ~np.all(np.isnan(x_train[:, embeddings.shape[1] :]), axis=0)
    full_keep = np.concatenate(
        [
            np.ones(embeddings.shape[1], dtype=bool),
            keep,
        ]
    )
    x_train = x_train[:, full_keep]
    x_val = x_val[:, full_keep]
    x_test = x_test[:, full_keep]

    model = build_model()
    model.fit(x_train, y_train)

    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    # Operating point is intentionally not optimized aggressively here.
    # 0.50 is used as a transparent ranking classifier threshold; ranking
    # itself is based on continuous probability.
    val_pr_auc = average_precision_score(y_val, val_prob)
    test_pr_auc = average_precision_score(y_test, test_prob)

    # Persist fitted model and metadata.
    import joblib

    joblib.dump(
        {
            "model": model,
            "feature_keep_mask": full_keep.tolist(),
            "embedding_dimension": int(embeddings.shape[1]),
            "feature_columns": feature_cols,
            "threshold": 0.50,
        },
        OUTPUT_DIR / "hybrid_ranker_model.joblib",
    )

    probabilities = np.full(len(merged), np.nan, dtype=float)
    probabilities[te] = test_prob

    return probabilities, float(val_pr_auc), float(test_pr_auc)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    merged, embeddings, splits = load()

    # Fit a ranking model once using train, then report locked validation/test
    # ranking scores. This does NOT tune on the test set.
    probabilities, val_pr_auc, test_pr_auc = score_model(
        merged,
        embeddings,
        splits,
    )

    evidence_rows = []
    ranked_rows = []
    lsr_rows = []
    precursor_summary_rows = []

    probability_lookup = {
        str(rid): float(prob)
        for rid, prob in zip(
            merged["report_id"].astype(str),
            probabilities,
        )
        if np.isfinite(prob)
    }

    for _, row in merged.iterrows():
        rid = str(row["report_id"])
        desc = str(row["description"])

        precursor, barriers, lsr = extract_evidence(desc)

        for item in precursor:
            evidence_rows.append(
                {
                    "report_id": rid,
                    "evidence_type": "precursor",
                    **item,
                }
            )

        for item in barriers:
            evidence_rows.append(
                {
                    "report_id": rid,
                    "evidence_type": "barrier",
                    **item,
                }
            )

        for item in lsr:
            evidence_rows.append(
                {
                    "report_id": rid,
                    "evidence_type": "lsr",
                    **item,
                }
            )

        mechanism_counts = {}
        for item in precursor:
            mechanism_counts[item["mechanism"]] = (
                mechanism_counts.get(item["mechanism"], 0) + 1
            )

        barrier_counts = {}
        for item in barriers:
            barrier_counts[item["barrier_signal"]] = (
                barrier_counts.get(item["barrier_signal"], 0) + 1
            )

        lsr_counts = {}
        for item in lsr:
            lsr_counts[item["lsr_candidate"]] = (
                lsr_counts.get(item["lsr_candidate"], 0) + 1
            )

        ranked_rows.append(
            {
                "report_id": rid,
                "description": desc,
                "sif_label": ("YES" if int(row["label"]) == 1 else "NO"),
                "split": (
                    "train"
                    if rid in splits["train"]
                    else (
                        "validation"
                        if rid in splits["validation"]
                        else "test" if rid in splits["test"] else "unknown"
                    )
                ),
                "sif_precursor_probability": probability_lookup.get(rid, np.nan),
                "precursor_evidence_count": len(precursor),
                "barrier_evidence_count": len(barriers),
                "lsr_candidate_count": len(lsr_counts),
                "top_precursor_mechanisms": " | ".join(
                    m
                    for m, _ in sorted(
                        mechanism_counts.items(),
                        key=lambda kv: (-kv[1], kv[0]),
                    )[:3]
                ),
                "barrier_signals": " | ".join(
                    b
                    for b, _ in sorted(
                        barrier_counts.items(),
                        key=lambda kv: (-kv[1], kv[0]),
                    )[:3]
                ),
                "lsr_candidates": " | ".join(
                    x
                    for x, _ in sorted(
                        lsr_counts.items(),
                        key=lambda kv: (-kv[1], kv[0]),
                    )[:4]
                ),
            }
        )

        for lsr_name, count in lsr_counts.items():
            lsr_rows.append(
                {
                    "report_id": rid,
                    "lsr_candidate": lsr_name,
                    "evidence_count": count,
                }
            )

        precursor_summary_rows.append(
            {
                "report_id": rid,
                "mechanism_count": len(mechanism_counts),
                "exposure_evidence_count": len(precursor),
                "barrier_signal_count": len(barrier_counts),
                "lsr_count": len(lsr_counts),
            }
        )

    ranked = pd.DataFrame(ranked_rows)

    # Build a reviewer queue from the locked test reports only.
    # Ranking remains continuous; the queue is not a safety determination.
    reviewer = ranked[ranked["split"] == "test"].copy()
    reviewer["review_priority_score"] = (
        reviewer["sif_precursor_probability"].fillna(0.0)
        + 0.02 * reviewer["barrier_evidence_count"]
        + 0.01 * reviewer["lsr_candidate_count"]
    )
    ranked_order = reviewer["review_priority_score"].rank(
        method="first",
        ascending=False,
    )
    # Use rank quartiles only when the queue has at least four reports.
    if len(reviewer) >= 4:
        reviewer["review_priority"] = pd.qcut(
            ranked_order,
            q=4,
            labels=["P1", "P2", "P3", "P4"],
        )
    else:
        reviewer["review_priority"] = [f"P{i}" for i in range(1, len(reviewer) + 1)]
    reviewer = reviewer.sort_values(
        ["review_priority", "review_priority_score"],
        ascending=[True, False],
    )

    ranked.sort_values(
        "sif_precursor_probability",
        ascending=False,
        na_position="last",
    ).to_csv(
        OUTPUT_DIR / "ranked_reports.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(evidence_rows).to_csv(
        OUTPUT_DIR / "evidence_spans.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(precursor_summary_rows).to_csv(
        OUTPUT_DIR / "precursor_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(lsr_rows).to_csv(
        OUTPUT_DIR / "lsr_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reviewer.to_csv(
        OUTPUT_DIR / "reviewer_queue.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_scores = reviewer["sif_precursor_probability"].dropna().to_numpy()

    pd.DataFrame(
        {
            "report_id": reviewer["report_id"],
            "sif_precursor_probability": reviewer["sif_precursor_probability"],
        }
    ).sort_values(
        "sif_precursor_probability",
        ascending=False,
    ).to_csv(
        OUTPUT_DIR / "model_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "experiment": "RAKSHAK Evidence-Aware SIF Precursor Ranking v0.4 FIXED FIXED",
        "records": int(len(merged)),
        "train_records": int(len(splits["train"])),
        "validation_records": int(len(splits["validation"])),
        "test_records": int(len(splits["test"])),
        "embedding_dimension": int(embeddings.shape[1]),
        "structured_feature_count": int(
            len(
                [
                    c
                    for c in pd.read_csv(FEATURE_PATH, nrows=1).columns
                    if c != "report_id"
                ]
            )
        ),
        "validation_pr_auc": val_pr_auc,
        "locked_test_pr_auc": test_pr_auc,
        "ranking_only": True,
        "operating_threshold": 0.50,
        "note": (
            "This component ranks reports for human review. Evidence spans and "
            "LSR candidates are heuristic text matches, not adjudicated safety facts."
        ),
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("RAKSHAK EVIDENCE-AWARE SIF PRECURSOR RANKING v0.4")
    print("=" * 78)
    print(f"Records              : {len(merged)}")
    print(f"Embedding dimension   : {embeddings.shape[1]}")
    print(f"Validation PR-AUC     : {val_pr_auc:.4f}")
    print(f"Locked test PR-AUC    : {test_pr_auc:.4f}")
    print(f"Test reviewer queue   : {len(reviewer)} reports")
    print()
    print(f"Saved outputs to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
