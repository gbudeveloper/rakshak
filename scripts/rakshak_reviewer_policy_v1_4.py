"""
RAKSHAK reviewer-priority policy v1.4

Design correction from v1.3:
- The classifier score is the primary ranking signal.
- Evidence is a modifier / review flag, not a gate that can demote a high-score
  record. The current evidence engine has only 18% exposure coverage and 15.82%
  pathway coverage on the 411-record batch, so using missing evidence as a hard
  priority gate would hide true positives.

Policy:
P1 = probability >= 0.90
     OR probability >= 0.75 AND complete pathway
     OR probability < 0.50 AND complete pathway (model-miss escalation)

P2 = probability >= 0.75
     OR probability >= 0.50 AND explicit exposure/pathway evidence

P3 = probability >= 0.50
     OR probability >= 0.30 AND explicit precursor evidence

P4 = probability < 0.30 with some precursor signal

P5 = probability < 0.30 with no precursor signal

Evidence is retained in separate columns and does not override the model.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def sig(v) -> bool:
    try:
        return int(v) == 1
    except (TypeError, ValueError):
        return False


def policy(row):
    p = float(row["sif_precursor_probability"])
    exposure = sig(row.get("exposure_signal", 0))
    pathway = sig(row.get("pathway_signal", 0))
    complete = sig(row.get("complete_pathway", 0))
    hazard = sig(row.get("hazard_signal", 0))
    explicit = hazard or exposure or pathway

    if p >= 0.90:
        return "P1", "VERY_HIGH_MODEL_SCORE"

    if p >= 0.75 and complete:
        return "P1", "HIGH_MODEL_PLUS_COMPLETE_PATHWAY"

    if p < 0.50 and complete:
        return "P1", "MODEL_MISS_PLUS_COMPLETE_PATHWAY"

    if p >= 0.75:
        return "P2", "HIGH_MODEL_SCORE_REVIEW"

    if p >= 0.50 and (exposure or pathway):
        return "P2", "MODERATE_HIGH_MODEL_PLUS_EXPLICIT_MECHANISM"

    if p >= 0.50:
        return "P3", "MODERATE_MODEL_SCORE"

    if p >= 0.30 and explicit:
        return "P3", "LOWER_MODEL_PLUS_PRECURSOR_SIGNAL"

    if explicit:
        return "P4", "LOW_MODEL_PLUS_PRECURSOR_SIGNAL"

    return "P5", "LOW_MODEL_NO_PRECURSOR_SIGNAL"


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "experiments" / "industrial_safety_predictions_v1.2.csv"
    output = root / "experiments" / "industrial_safety_predictions_v1.4.csv"

    if not source.exists():
        raise FileNotFoundError(source)

    df = pd.read_csv(source)

    values = [policy(row) for _, row in df.iterrows()]
    df["review_priority"] = [x[0] for x in values]
    df["review_priority_reason"] = [x[1] for x in values]

    order = {"P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5}
    df["_rank"] = df["review_priority"].map(order)
    df = df.sort_values(
        ["_rank", "sif_precursor_probability"],
        ascending=[True, False],
    ).drop(columns="_rank")

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")

    counts = (
        df["review_priority"]
        .value_counts()
        .reindex(["P1", "P2", "P3", "P4", "P5"], fill_value=0)
    )

    print("=" * 80)
    print("RAKSHAK REVIEWER PRIORITY POLICY v1.4")
    print("=" * 80)
    print(f"Input records: {len(df)}")
    print("\nPriority distribution:")
    for k, v in counts.items():
        print(f"  {k}: {int(v):3d} ({100*v/len(df):5.2f}%)")

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
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(output)


if __name__ == "__main__":
    main()
