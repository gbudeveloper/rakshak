"""
RAKSHAK reviewer-priority policy v1.3

Purpose:
Replace the over-aggressive v1.2 batch priority rule.

Problem in v1.2:
    probability >= 0.75 + UNKNOWN/CONTEXT_LIMITED -> P1

On the 411-record batch this produced 148 P1s (36.01%).
UNKNOWN is common in free-text reports and should not, by itself, be treated
as a high-severity conflict.

Policy v1.3:
P1 = high model score + explicit precursor evidence
     OR high-score model conflict with explicit evidence gap requiring HSSE review
     OR low-score model + explicit complete pathway (model-miss escalation)

P2 = high model score but weak/missing precursor evidence

P3 = moderate model score + explicit precursor evidence
     OR complete pathway regardless of score

P4 = moderate/low score with some hazard/exposure/pathway signal

P5 = low score with no useful precursor evidence

This is a reviewer queue, not a safety verdict.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def truthy_signal(value) -> bool:
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return False


def assign_priority(row) -> tuple[str, str]:
    p = float(row["sif_precursor_probability"])
    hazard = truthy_signal(row.get("hazard_signal", 0))
    exposure = truthy_signal(row.get("exposure_signal", 0))
    pathway = truthy_signal(row.get("pathway_signal", 0))
    complete = truthy_signal(row.get("complete_pathway", 0))
    strength = int(row.get("pathway_signal_strength", 0) or 0)

    explicit_evidence = hazard or exposure or pathway
    strong_evidence = exposure or pathway or complete or strength >= 2

    # P1: highest-value human review.
    # A high model score is not enough. We require narrative precursor evidence.
    if p >= 0.80 and strong_evidence:
        return "P1", "HIGH_MODEL_PLUS_EXPLICIT_PRECURSOR_EVIDENCE"

    # A complete pathway with a low model score is a model-miss escalation.
    if p < 0.50 and complete:
        return "P1", "LOW_MODEL_PLUS_COMPLETE_PATHWAY_REVIEW"

    # P2: high score but weak or missing narrative evidence.
    if p >= 0.75:
        return "P2", "HIGH_MODEL_WEAK_OR_MISSING_EVIDENCE"

    # P3: moderate score with meaningful evidence, or complete pathway.
    if complete:
        return "P3", "COMPLETE_PATHWAY_REVIEW"

    if p >= 0.50 and explicit_evidence:
        return "P3", "MODERATE_MODEL_PLUS_EXPLICIT_EVIDENCE"

    # P4: some signal exists but the model score is low/moderate.
    if explicit_evidence:
        return "P4", "LOW_MODEL_PLUS_SOME_PRECURSOR_SIGNAL"

    return "P5", "LOW_MODEL_NO_PRECURSOR_EVIDENCE"


def main():
    project_root = Path(__file__).resolve().parents[1]
    input_file = (
        project_root
        / "experiments"
        / "industrial_safety_predictions_v1.2.csv"
    )
    output_file = (
        project_root
        / "experiments"
        / "industrial_safety_predictions_v1.3.csv"
    )

    if not input_file.exists():
        raise FileNotFoundError(input_file)

    df = pd.read_csv(input_file)

    priorities = []
    reasons = []

    for _, row in df.iterrows():
        priority, reason = assign_priority(row)
        priorities.append(priority)
        reasons.append(reason)

    df["review_priority"] = priorities
    df["review_priority_reason"] = reasons

    # Keep the model score untouched.
    # Keep Evidence Engine fields untouched.
    priority_order = {"P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5}

    df["_priority_rank"] = df["review_priority"].map(priority_order)
    df = df.sort_values(
        ["_priority_rank", "sif_precursor_probability"],
        ascending=[True, False],
    ).drop(columns="_priority_rank")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    dist = (
        df["review_priority"]
        .value_counts()
        .reindex(["P1", "P2", "P3", "P4", "P5"], fill_value=0)
    )

    print("=" * 80)
    print("RAKSHAK REVIEWER PRIORITY POLICY v1.3")
    print("=" * 80)
    print(f"Input records : {len(df)}")
    print("\nPriority distribution:")
    for key, value in dist.items():
        print(f"  {key}: {int(value):3d} ({100*value/len(df):5.2f}%)")

    print("\nTop 20:")
    print(
        df[
            [
                "report_id",
                "sif_precursor_probability",
                "review_priority",
                "review_priority_reason",
                "hazards",
                "exposures",
                "pathways",
            ]
        ].head(20).to_string(index=False)
    )

    print("\nSaved:")
    print(output_file)


if __name__ == "__main__":
    main()
