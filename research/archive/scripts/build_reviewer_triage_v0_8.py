"""
RAKSHAK Reviewer Triage Policy v0.8 FIXED
===================================

Builds an operational triage layer on top of the v0.4/v0.7 outputs.

Important:
- Does NOT change the trained model.
- Does NOT infer missing barriers.
- Does NOT suppress a high model score merely because evidence is sparse.
- Uses evidence coverage to decide how much human review is required.
- Labels LSR mappings as "candidate", not verified.

Triage tiers
------------
P1  HIGH_MODEL + narrative precursor evidence
P2  HIGH_MODEL + no precursor evidence (model-only; mandatory review)
P3  MODERATE_MODEL + precursor evidence
P4  LOW_MODEL or weak evidence

Additional flags
----------------
MODEL_LABEL_CONFLICT
NO_PRECURSOR_EVIDENCE
NO_LSR_EVIDENCE
NO_CONTROL_EVIDENCE
CANDIDATE_LSR_PRESENT

Outputs
-------
experiments/sif_reviewer_triage_v0.8/
  reviewer_triage.csv
  p1_cases.csv
  p2_model_only_cases.csv
  model_label_conflicts.csv
  triage_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIT_DIR = PROJECT_ROOT / "experiments" / "sif_barrier_inspection_v0.7"
RANK_DIR = PROJECT_ROOT / "experiments" / "sif_evidence_aware_ranking_v0.4"

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "sif_reviewer_triage_v0.8"

BARRIER_PATH = AUDIT_DIR / "narrative_barrier_inspection.csv"
RANK_PATH = RANK_DIR / "ranked_reports.csv"


def load_inputs() -> pd.DataFrame:
    if not BARRIER_PATH.exists():
        raise FileNotFoundError(BARRIER_PATH)
    if not RANK_PATH.exists():
        raise FileNotFoundError(RANK_PATH)

    barrier = pd.read_csv(BARRIER_PATH)
    rank = pd.read_csv(RANK_PATH)

    rank = rank[rank["split"].astype(str).str.lower().eq("test")].copy()

    rank["probability"] = pd.to_numeric(
        rank["sif_precursor_probability"], errors="coerce"
    ).fillna(0.0)

    rank["true_label_yes"] = rank["sif_label"].astype(str).str.upper().eq("YES")

    merged = rank.merge(
        barrier[
            [
                "report_id",
                "control_category_count",
                "barrier_gap_category_count",
                "control_categories_mentioned",
                "barrier_gap_categories_mentioned",
            ]
        ],
        on="report_id",
        how="left",
    )

    merged["control_category_count"] = (
        pd.to_numeric(merged["control_category_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    merged["barrier_gap_category_count"] = (
        pd.to_numeric(merged["barrier_gap_category_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    merged["precursor_evidence_count"] = (
        pd.to_numeric(merged["precursor_evidence_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    merged["lsr_candidate_count"] = (
        pd.to_numeric(merged["lsr_candidate_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    return merged


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_inputs()

    # Thresholds are intentionally broad and transparent.
    # This is a triage policy, not a tuned clinical/safety classifier.
    df["high_model"] = df["probability"] >= 0.80
    df["moderate_model"] = df["probability"] >= 0.50

    df["has_precursor_evidence"] = df["precursor_evidence_count"] > 0
    df["has_control_language"] = df["control_category_count"] > 0
    df["has_gap_language"] = df["barrier_gap_category_count"] > 0
    df["has_lsr_candidate"] = df["lsr_candidate_count"] > 0

    df["flag_no_precursor_evidence"] = ~df["has_precursor_evidence"]
    df["flag_no_control_evidence"] = ~df["has_control_language"]
    df["flag_no_lsr_evidence"] = df["has_lsr_candidate"] == False
    df["flag_model_label_conflict"] = (
        (df["probability"] >= 0.80) & (~df["true_label_yes"])
    ) | ((df["probability"] < 0.30) & (df["true_label_yes"]))

    # Safety-first triage:
    # A model-only high score remains high priority and is NOT downgraded.
    df["triage_tier"] = "P4"

    p1 = df["high_model"] & df["has_precursor_evidence"]
    p2 = df["high_model"] & ~df["has_precursor_evidence"]
    p3 = df["moderate_model"] & ~df["high_model"] & df["has_precursor_evidence"]

    df.loc[p1, "triage_tier"] = "P1"
    df.loc[p2, "triage_tier"] = "P2"
    df.loc[p3, "triage_tier"] = "P3"

    df["triage_reason"] = "low_or_weak_model_score"

    df.loc[p1, "triage_reason"] = "high_model_score_with_narrative_precursor_evidence"
    df.loc[p2, "triage_reason"] = (
        "high_model_score_but_no_precursor_evidence_mandatory_review"
    )
    df.loc[p3, "triage_reason"] = (
        "moderate_model_score_with_narrative_precursor_evidence"
    )

    # Human-review instruction.
    df["review_instruction"] = (
        "Review narrative and validate precursor mechanism before operational action."
    )
    df.loc[p2, "review_instruction"] = (
        "Mandatory manual review: model is high but narrative precursor evidence "
        "was not detected. Do not treat the score as a confirmed SIF precursor."
    )

    df = df.sort_values(
        ["triage_tier", "probability", "precursor_evidence_count"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    # Save queue and useful subsets.
    df.to_csv(
        OUTPUT_DIR / "reviewer_triage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df[df["triage_tier"] == "P1"].to_csv(
        OUTPUT_DIR / "p1_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df[df["triage_tier"] == "P2"].to_csv(
        OUTPUT_DIR / "p2_model_only_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df[df["flag_model_label_conflict"]].to_csv(
        OUTPUT_DIR / "model_label_conflicts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "experiment": "RAKSHAK Reviewer Triage Policy v0.8 FIXED",
        "test_records": int(len(df)),
        "tiers": {
            "P1_high_model_with_evidence": int((df["triage_tier"] == "P1").sum()),
            "P2_high_model_no_evidence": int((df["triage_tier"] == "P2").sum()),
            "P3_moderate_with_evidence": int((df["triage_tier"] == "P3").sum()),
            "P4_other": int((df["triage_tier"] == "P4").sum()),
        },
        "flags": {
            "model_label_conflict": int(df["flag_model_label_conflict"].sum()),
            "no_precursor_evidence": int(df["flag_no_precursor_evidence"].sum()),
            "no_control_language": int(df["flag_no_control_evidence"].sum()),
            "no_lsr_candidate": int(df["flag_no_lsr_evidence"].sum()),
        },
        "policy": {
            "high_model_threshold": 0.80,
            "moderate_model_threshold": 0.50,
            "safety_behavior": (
                "A high model score is never suppressed because evidence extraction "
                "is sparse; evidence sparsity instead increases mandatory human review."
            ),
        },
    }

    (OUTPUT_DIR / "triage_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("RAKSHAK REVIEWER TRIAGE POLICY v0.8 FIXED")
    print("=" * 78)
    print(f"Test reports                         : {len(df)}")
    print(f"P1 high + evidence                   : {(df['triage_tier'] == 'P1').sum()}")
    print(f"P2 high + NO evidence                : {(df['triage_tier'] == 'P2').sum()}")
    print(f"P3 moderate + evidence               : {(df['triage_tier'] == 'P3').sum()}")
    print(f"P4 other                             : {(df['triage_tier'] == 'P4').sum()}")
    print(
        f"Model/label conflicts                : {df['flag_model_label_conflict'].sum()}"
    )
    print(
        f"No precursor evidence               : {df['flag_no_precursor_evidence'].sum()}"
    )
    print()
    print("TOP REVIEW QUEUE")
    print("-" * 78)

    show = [
        "report_id",
        "probability",
        "sif_label",
        "triage_tier",
        "triage_reason",
        "precursor_evidence_count",
        "lsr_candidates",
        "top_precursor_mechanisms",
    ]
    show = [c for c in show if c in df.columns]
    print(df[show].head(15).to_string(index=False))

    print()
    print(f"Saved outputs to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
