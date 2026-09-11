"""
RAKSHAK Narrative Barrier Inspection v0.7
==========================================

Reads:
  experiments/sif_evidence_aware_ranking_v0.6_audit/
      narrative_barrier_audit.csv
      test_ranking_audit.csv

Prints:
  - actual control-mentioned rate
  - actual gap-mentioned rate
  - reports with both controls and gaps
  - reports with no barrier-related language
  - top ranked cases with barrier evidence
  - high-score cases needing barrier review

This is diagnostic only. It never treats a mentioned control as an
effective control and never infers a barrier gap from absence of words.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "experiments" / "sif_evidence_aware_ranking_v0.6_audit"

NARRATIVE_PATH = AUDIT_DIR / "narrative_barrier_audit.csv"
RANKING_PATH = AUDIT_DIR / "test_ranking_audit.csv"

OUT_DIR = PROJECT_ROOT / "experiments" / "sif_barrier_inspection_v0.7"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not NARRATIVE_PATH.exists():
        raise FileNotFoundError(NARRATIVE_PATH)
    if not RANKING_PATH.exists():
        raise FileNotFoundError(RANKING_PATH)

    barriers = pd.read_csv(NARRATIVE_PATH)
    ranking = pd.read_csv(RANKING_PATH)

    barriers["control_category_count"] = pd.to_numeric(
        barriers["control_category_count"], errors="coerce"
    ).fillna(0)
    barriers["barrier_gap_category_count"] = pd.to_numeric(
        barriers["barrier_gap_category_count"], errors="coerce"
    ).fillna(0)
    barriers["probability"] = pd.to_numeric(
        barriers["probability"], errors="coerce"
    )

    barriers["has_control"] = barriers["control_category_count"] > 0
    barriers["has_gap"] = barriers["barrier_gap_category_count"] > 0
    barriers["has_both"] = barriers["has_control"] & barriers["has_gap"]
    barriers["has_neither"] = ~barriers["has_control"] & ~barriers["has_gap"]

    total = len(barriers)

    summary = {
        "test_records": int(total),
        "control_mentioned_rate": float(barriers["has_control"].mean()) if total else None,
        "gap_mentioned_rate": float(barriers["has_gap"].mean()) if total else None,
        "both_control_and_gap_rate": float(barriers["has_both"].mean()) if total else None,
        "neither_rate": float(barriers["has_neither"].mean()) if total else None,
        "control_only_count": int(
            (barriers["has_control"] & ~barriers["has_gap"]).sum()
        ),
        "gap_only_count": int(
            (~barriers["has_control"] & barriers["has_gap"]).sum()
        ),
        "both_count": int(barriers["has_both"].sum()),
        "neither_count": int(barriers["has_neither"].sum()),
    }

    # Merge ranking details.
    columns = [
        "report_id",
        "description",
        "probability",
        "sif_label",
        "top_precursor_mechanisms",
        "lsr_candidates",
    ]
    columns = [c for c in columns if c in ranking.columns]

    merged = barriers.merge(
        ranking[columns],
        on="report_id",
        how="left",
        suffixes=("", "_ranking"),
    )

    merged = merged.sort_values(
        ["probability", "report_id"],
        ascending=[False, True],
        na_position="last",
    )

    high_priority = merged[
        merged["probability"].fillna(0.0) >= 0.50
    ].copy()

    no_barrier_cases = merged[merged["has_neither"]].copy()
    both_cases = merged[merged["has_both"]].copy()

    barriers.to_csv(
        OUT_DIR / "narrative_barrier_inspection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    high_priority.to_csv(
        OUT_DIR / "high_priority_barrier_review.csv",
        index=False,
        encoding="utf-8-sig",
    )
    no_barrier_cases.to_csv(
        OUT_DIR / "no_barrier_language_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )
    both_cases.to_csv(
        OUT_DIR / "control_and_gap_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    import json
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("RAKSHAK NARRATIVE BARRIER INSPECTION v0.7")
    print("=" * 78)
    print(f"Test records                    : {total}")
    print(f"Control mentioned rate         : {summary['control_mentioned_rate']:.1%}")
    print(f"Gap mentioned rate             : {summary['gap_mentioned_rate']:.1%}")
    print(f"Both control + gap rate        : {summary['both_control_and_gap_rate']:.1%}")
    print(f"Neither control nor gap        : {summary['neither_rate']:.1%}")
    print(f"High-priority reports (>=0.50): {len(high_priority)}")
    print()

    print("TOP HIGH-PRIORITY REPORTS")
    print("-" * 78)
    cols = [
        c for c in [
            "report_id",
            "probability",
            "sif_label",
            "control_category_count",
            "barrier_gap_category_count",
            "control_categories_mentioned",
            "barrier_gap_categories_mentioned",
            "top_precursor_mechanisms",
        ]
        if c in high_priority.columns
    ]
    print(high_priority[cols].head(10).to_string(index=False))

    print()
    print(f"Saved outputs to:\n{OUT_DIR}")


if __name__ == "__main__":
    main()
