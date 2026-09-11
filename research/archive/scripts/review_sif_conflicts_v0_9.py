"""
RAKSHAK SIF Conflict Review v0.9
================================

Purpose:
- Inspect model/label conflicts from the current locked-test ranking.
- Create a human-review sheet for the most important false positives,
  false negatives, and evidence-sparse high-score reports.
- Do NOT retrain or alter the model.
- Do NOT infer that the existing binary label is wrong.
- Preserve the original narrative for adjudication.

Inputs:
  experiments/sif_evidence_aware_ranking_v0.4/ranked_reports.csv
  experiments/sif_barrier_inspection_v0.7/narrative_barrier_inspection.csv
  data/annotations/resolved_annotations_v0.2.csv

Outputs:
  experiments/sif_conflict_review_v0.9/
    conflict_review.csv
    high_score_false_positives.csv
    low_score_false_negatives.csv
    evidence_sparse_high_score.csv
    review_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RANKING_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "sif_evidence_aware_ranking_v0.4"
    / "ranked_reports.csv"
)

BARRIER_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "sif_barrier_inspection_v0.7"
    / "narrative_barrier_inspection.csv"
)

LABEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "resolved_annotations_v0.2.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "sif_conflict_review_v0.9"
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in (RANKING_PATH, BARRIER_PATH, LABEL_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    ranking = pd.read_csv(RANKING_PATH)
    barrier = pd.read_csv(BARRIER_PATH)
    labels = pd.read_csv(LABEL_PATH)

    ranking = ranking[
        ranking["split"].astype(str).str.lower().eq("test")
    ].copy()

    ranking["probability"] = pd.to_numeric(
        ranking["sif_precursor_probability"], errors="coerce"
    )

    ranking["true_label"] = (
        ranking["sif_label"].astype(str).str.upper().eq("YES").astype(int)
    )

    barrier_cols = [
        "report_id",
        "control_category_count",
        "barrier_gap_category_count",
        "control_categories_mentioned",
        "barrier_gap_categories_mentioned",
    ]
    barrier_cols = [c for c in barrier_cols if c in barrier.columns]

    merged = ranking.merge(
        barrier[barrier_cols],
        on="report_id",
        how="left",
        suffixes=("", "_barrier"),
    )

    # Prefer the ranking narrative; fall back to adjudication source if needed.
    if "description" not in merged.columns:
        merged = merged.merge(
            labels[["report_id", "description"]],
            on="report_id",
            how="left",
        )

    for c in (
        "precursor_evidence_count",
        "barrier_evidence_count",
        "lsr_candidate_count",
        "control_category_count",
        "barrier_gap_category_count",
    ):
        if c in merged.columns:
            merged[c] = pd.to_numeric(
                merged[c], errors="coerce"
            ).fillna(0).astype(int)

    merged["review_type"] = "NORMAL"

    # Strict diagnostics:
    # High-score NO = false positive against current adjudicated label.
    merged.loc[
        (merged["probability"] >= 0.80) & (merged["true_label"] == 0),
        "review_type",
    ] = "HIGH_SCORE_FALSE_POSITIVE"

    # Low-score YES = false negative at the operational 0.50 reference point.
    merged.loc[
        (merged["probability"] < 0.50) & (merged["true_label"] == 1),
        "review_type",
    ] = "LOW_SCORE_FALSE_NEGATIVE"

    # High score but no precursor evidence = evidence-sparse model signal.
    merged.loc[
        (merged["probability"] >= 0.80)
        & (merged["precursor_evidence_count"] == 0),
        "review_type",
    ] = "HIGH_SCORE_NO_PRECURSOR_EVIDENCE"

    # Priority order for human review.
    priority = {
        "HIGH_SCORE_FALSE_POSITIVE": 1,
        "HIGH_SCORE_NO_PRECURSOR_EVIDENCE": 2,
        "LOW_SCORE_FALSE_NEGATIVE": 3,
        "NORMAL": 4,
    }
    merged["review_priority"] = (
        merged["review_type"].map(priority).fillna(4).astype(int)
    )

    # Suggested adjudication questions, not conclusions.
    merged["review_questions"] = merged["review_type"].map({
        "HIGH_SCORE_FALSE_POSITIVE": (
            "Check whether the narrative contains a credible SIF pathway. "
            "Identify which mechanism phrase drove the score and whether it "
            "is actually hazardous in context."
        ),
        "HIGH_SCORE_NO_PRECURSOR_EVIDENCE": (
            "Check why the model scored this report highly despite no detected "
            "precursor span. Inspect narrative context manually before action."
        ),
        "LOW_SCORE_FALSE_NEGATIVE": (
            "Check whether the narrative contains a subtle precursor mechanism "
            "that the model failed to rank highly. Record the missing mechanism."
        ),
        "NORMAL": "",
    }).fillna("")

    # Empty fields for the actual human adjudicator to complete.
    merged["human_precursor_decision"] = ""
    merged["human_mechanism"] = ""
    merged["human_evidence_span"] = ""
    merged["human_barrier_state"] = ""
    merged["human_lsr_candidate"] = ""
    merged["human_notes"] = ""

    merged = merged.sort_values(
        ["review_priority", "probability", "report_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    review_columns = [
        "review_priority",
        "review_type",
        "report_id",
        "probability",
        "sif_label",
        "description",
        "precursor_evidence_count",
        "top_precursor_mechanisms",
        "lsr_candidates",
        "control_category_count",
        "barrier_gap_category_count",
        "control_categories_mentioned",
        "barrier_gap_categories_mentioned",
        "review_questions",
        "human_precursor_decision",
        "human_mechanism",
        "human_evidence_span",
        "human_barrier_state",
        "human_lsr_candidate",
        "human_notes",
    ]
    review_columns = [c for c in review_columns if c in merged.columns]

    review = merged[review_columns].copy()

    high_fp = review[
        review["review_type"] == "HIGH_SCORE_FALSE_POSITIVE"
    ].copy()

    high_no_evidence = review[
        review["review_type"] == "HIGH_SCORE_NO_PRECURSOR_EVIDENCE"
    ].copy()

    low_fn = review[
        review["review_type"] == "LOW_SCORE_FALSE_NEGATIVE"
    ].copy()

    review.to_csv(
        OUTPUT_DIR / "conflict_review.csv",
        index=False,
        encoding="utf-8-sig",
    )
    high_fp.to_csv(
        OUTPUT_DIR / "high_score_false_positives.csv",
        index=False,
        encoding="utf-8-sig",
    )
    high_no_evidence.to_csv(
        OUTPUT_DIR / "evidence_sparse_high_score.csv",
        index=False,
        encoding="utf-8-sig",
    )
    low_fn.to_csv(
        OUTPUT_DIR / "low_score_false_negatives.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "experiment": "RAKSHAK SIF Conflict Review v0.9",
        "test_records": int(len(review)),
        "high_score_false_positives": int(len(high_fp)),
        "evidence_sparse_high_score": int(len(high_no_evidence)),
        "low_score_false_negatives": int(len(low_fn)),
        "note": (
            "These are review candidates against the current adjudicated labels. "
            "They are not automatic corrections to labels or proof of model error."
        ),
    }

    (OUTPUT_DIR / "review_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("RAKSHAK SIF CONFLICT REVIEW v0.9")
    print("=" * 78)
    print(f"Test records                 : {len(review)}")
    print(f"High-score false positives  : {len(high_fp)}")
    print(f"High-score/no-evidence      : {len(high_no_evidence)}")
    print(f"Low-score false negatives   : {len(low_fn)}")
    print()

    if len(high_fp):
        print("HIGH-SCORE FALSE POSITIVES")
        print("-" * 78)
        print(
            high_fp[
                [
                    "report_id",
                    "probability",
                    "sif_label",
                    "precursor_evidence_count",
                    "top_precursor_mechanisms",
                    "lsr_candidates",
                ]
            ].to_string(index=False)
        )
        print()

    if len(low_fn):
        print("LOW-SCORE FALSE NEGATIVES")
        print("-" * 78)
        print(
            low_fn[
                [
                    "report_id",
                    "probability",
                    "sif_label",
                    "precursor_evidence_count",
                    "top_precursor_mechanisms",
                    "lsr_candidates",
                ]
            ].to_string(index=False)
        )
        print()

    print(f"Saved outputs to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
