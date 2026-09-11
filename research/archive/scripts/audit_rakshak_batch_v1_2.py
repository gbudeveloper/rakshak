"""
Audit RAKSHAK batch inference v1.2.

Input:
  experiments\industrial_safety_predictions_v1.2.csv

Outputs:
  experiments\rakshak_batch_audit_v1.2\
    batch_summary.json
    priority_distribution.csv
    conflict_distribution.csv
    evidence_coverage.csv
    top20_high_score.csv
    top20_no_evidence.csv

Purpose:
- Quantify reviewer queue composition.
- Detect whether P1 is over-triggering.
- Measure hazard/exposure/pathway evidence coverage.
- Separate high model score from evidence status.
- No model retraining or threshold changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "experiments" / "industrial_safety_predictions_v1.2.csv"
OUT = PROJECT_ROOT / "experiments" / "rakshak_batch_audit_v1.2"


def norm(s):
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def pct(n, d):
    return round(100.0 * n / d, 2) if d else 0.0


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)

    required = [
        "report_id",
        "sif_precursor_probability",
        "prediction_at_0_50",
        "review_priority",
        "model_evidence_conflicts",
        "hazards",
        "exposures",
        "pathways",
        "hazard_signal",
        "exposure_signal",
        "pathway_signal",
        "pathway_signal_strength",
        "complete_pathway",
        "sif_context_status",
        "pathway_assessment",
        "description",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    n = len(df)
    prob = pd.to_numeric(
        df["sif_precursor_probability"], errors="coerce"
    )

    has_hazard = df["hazards"].fillna("").astype(str).str.strip().ne("")
    has_exposure = df["exposures"].fillna("").astype(str).str.strip().ne("")
    has_pathway = df["pathways"].fillna("").astype(str).str.strip().ne("")
    complete = df["complete_pathway"].fillna(0).astype(int).eq(1)

    high = prob.ge(0.75)
    high80 = prob.ge(0.80)
    moderate = prob.ge(0.50) & prob.lt(0.75)
    low = prob.lt(0.50)

    unknown = norm(df["sif_context_status"]).eq("unknown")
    limited = norm(df["sif_context_status"]).eq("context_limited")
    sufficient = norm(df["sif_context_status"]).eq("sufficient_context")

    conflict = df["model_evidence_conflicts"].fillna("").astype(str).str.strip().ne("")
    p1 = df["review_priority"].eq("P1")
    p2 = df["review_priority"].eq("P2")
    p3 = df["review_priority"].eq("P3")
    p4 = df["review_priority"].eq("P4")
    p5 = df["review_priority"].eq("P5")

    summary = {
        "records": int(n),
        "predicted_yes_at_0_50": int(df["prediction_at_0_50"].sum()),
        "predicted_no_at_0_50": int(n - df["prediction_at_0_50"].sum()),

        "score_distribution": {
            "high_ge_0_75": int(high.sum()),
            "high_ge_0_80": int(high80.sum()),
            "moderate_0_50_to_0_75": int(moderate.sum()),
            "low_lt_0_50": int(low.sum()),
            "mean_probability": round(float(prob.mean()), 6),
            "median_probability": round(float(prob.median()), 6),
        },

        "priority_distribution": {
            "P1": int(p1.sum()),
            "P2": int(p2.sum()),
            "P3": int(p3.sum()),
            "P4": int(p4.sum()),
            "P5": int(p5.sum()),
        },

        "evidence_coverage": {
            "hazard_signal_pct": pct(int(has_hazard.sum()), n),
            "exposure_signal_pct": pct(int(has_exposure.sum()), n),
            "pathway_signal_pct": pct(int(has_pathway.sum()), n),
            "complete_pathway_pct": pct(int(complete.sum()), n),
            "no_hazard_evidence_pct": pct(int((~has_hazard).sum()), n),
            "no_exposure_evidence_pct": pct(int((~has_exposure).sum()), n),
            "no_pathway_evidence_pct": pct(int((~has_pathway).sum()), n),
        },

        "context_distribution": {
            "UNKNOWN": int(unknown.sum()),
            "CONTEXT_LIMITED": int(limited.sum()),
            "SUFFICIENT_CONTEXT": int(sufficient.sum()),
        },

        "conflicts": {
            "any_conflict": int(conflict.sum()),
            "any_conflict_pct": pct(int(conflict.sum()), n),
            "high_score_and_unknown_or_limited": int(
                (high & (unknown | limited)).sum()
            ),
            "high_score_and_no_pathway": int(
                (high & ~has_pathway).sum()
            ),
            "high_score_and_complete_pathway": int(
                (high & complete).sum()
            ),
        },

        "queue_sanity": {
            "p1_pct": pct(int(p1.sum()), n),
            "p2_pct": pct(int(p2.sum()), n),
            "p3_pct": pct(int(p3.sum()), n),
            "p4_pct": pct(int(p4.sum()), n),
            "p5_pct": pct(int(p5.sum()), n),
        },

        "interpretation": (
            "Descriptive batch audit only. Do not infer true SIF prevalence "
            "from model predictions on unlabeled records."
        ),
    }

    (OUT / "batch_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    def distribution(col):
        return (
            df[col]
            .fillna("")
            .astype(str)
            .replace("", "EMPTY")
            .value_counts(dropna=False)
            .rename_axis(col)
            .reset_index(name="records")
        )

    for col, filename in [
        ("review_priority", "priority_distribution.csv"),
        ("model_evidence_conflicts", "conflict_distribution.csv"),
        ("sif_context_status", "context_distribution.csv"),
        ("pathway_assessment", "pathway_assessment_distribution.csv"),
    ]:
        distribution(col).to_csv(
            OUT / filename,
            index=False,
            encoding="utf-8-sig",
        )

    evidence = pd.DataFrame([
        {
            "signal": "hazard",
            "records": int(has_hazard.sum()),
            "pct": pct(int(has_hazard.sum()), n),
        },
        {
            "signal": "exposure",
            "records": int(has_exposure.sum()),
            "pct": pct(int(has_exposure.sum()), n),
        },
        {
            "signal": "pathway",
            "records": int(has_pathway.sum()),
            "pct": pct(int(has_pathway.sum()), n),
        },
        {
            "signal": "complete_pathway",
            "records": int(complete.sum()),
            "pct": pct(int(complete.sum()), n),
        },
    ])
    evidence.to_csv(
        OUT / "evidence_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cols = [
        "report_id",
        "sif_precursor_probability",
        "review_priority",
        "model_evidence_conflicts",
        "hazards",
        "exposures",
        "pathways",
        "sif_context_status",
        "pathway_assessment",
        "description",
    ]

    df.sort_values(
        "sif_precursor_probability",
        ascending=False,
    ).head(20)[cols].to_csv(
        OUT / "top20_high_score.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df[
        high
        & ~has_pathway
    ].sort_values(
        "sif_precursor_probability",
        ascending=False,
    ).head(20)[cols].to_csv(
        OUT / "top20_high_score_no_pathway.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df[
        low
        & (has_hazard | has_exposure | has_pathway)
    ].sort_values(
        "sif_precursor_probability",
        ascending=True,
    ).head(20)[cols].to_csv(
        OUT / "top20_low_score_with_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("RAKSHAK BATCH AUDIT v1.2")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print("\nFiles saved to:")
    print(OUT)


if __name__ == "__main__":
    main()
