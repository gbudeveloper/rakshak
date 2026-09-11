"""
Evaluate RAKSHAK reviewer-priority policy v1.4 on the current 70 adjudicated
binary-labelled records.

Audit only:
- no model retraining
- no threshold fitting
- no use of the locked test to tune the policy

The goal is to verify that v1.4 does not recreate the v1.3 problem where P2
was much more enriched than P1, while also checking whether true YES records
are being pushed into P4/P5.
"""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "experiments" / "industrial_safety_predictions_v1.4.csv"
LABEL = ROOT / "data" / "annotations" / "resolved_annotations_v0.2.csv"
OUT = ROOT / "experiments" / "reviewer_policy_eval_v1.4"

PRIORITIES = ["P1", "P2", "P3", "P4", "P5"]


def main():
    if not PRED.exists():
        raise FileNotFoundError(PRED)
    if not LABEL.exists():
        raise FileNotFoundError(LABEL)

    pred = pd.read_csv(PRED)
    lab = pd.read_csv(LABEL)

    pred["report_id"] = pred["report_id"].astype(str)
    lab["report_id"] = lab["report_id"].astype(str)
    lab["final_sif_potential"] = (
        lab["final_sif_potential"].astype(str).str.strip().str.upper()
    )
    lab = lab[lab["final_sif_potential"].isin(["YES", "NO"])].copy()
    lab["gold_label"] = (lab["final_sif_potential"] == "YES").astype(int)

    cols = [
        "report_id",
        "sif_precursor_probability",
        "review_priority",
        "review_priority_reason",
        "hazards",
        "exposures",
        "pathways",
        "complete_pathway",
        "description",
    ]

    df = pred[cols].merge(
        lab[["report_id", "gold_label", "final_sif_potential"]],
        on="report_id",
        how="inner",
        validate="one_to_one",
    )

    if len(df) != 70:
        raise ValueError(f"Expected 70 binary-labelled records, got {len(df)}")

    rows = []
    for p in PRIORITIES:
        s = df[df["review_priority"] == p]
        rows.append(
            {
                "priority": p,
                "records": len(s),
                "gold_yes": int(s["gold_label"].sum()),
                "gold_no": int((1 - s["gold_label"]).sum()),
                "gold_yes_rate": float(s["gold_label"].mean()) if len(s) else None,
                "mean_probability": float(s["sif_precursor_probability"].mean()) if len(s) else None,
                "median_probability": float(s["sif_precursor_probability"].median()) if len(s) else None,
            }
        )
    tiers = pd.DataFrame(rows)

    cumulative = []
    for k in range(1, 6):
        s = df[df["review_priority"].isin(PRIORITIES[:k])]
        cumulative.append(
            {
                "queue": f"P1-P{k}",
                "records": len(s),
                "gold_yes": int(s["gold_label"].sum()),
                "gold_yes_rate": float(s["gold_label"].mean()) if len(s) else None,
            }
        )
    cumulative = pd.DataFrame(cumulative)

    p1 = df["review_priority"].eq("P1")
    p2 = df["review_priority"].isin(["P1", "P2"])
    p3 = df["review_priority"].isin(["P1", "P2", "P3"])

    yes_total = int(df["gold_label"].sum())

    high_no = df[
        (df["sif_precursor_probability"] >= 0.75)
        & (df["gold_label"] == 0)
    ].sort_values("sif_precursor_probability", ascending=False)

    yes_low = df[
        (df["gold_label"] == 1)
        & df["review_priority"].isin(["P4", "P5"])
    ].sort_values("sif_precursor_probability", ascending=True)

    output = {
        "labelled_records": len(df),
        "gold_yes": yes_total,
        "gold_no": int(len(df) - yes_total),
        "priority_distribution": tiers[
            ["priority", "records", "gold_yes", "gold_no", "gold_yes_rate"]
        ].to_dict("records"),
        "queue_enrichment": cumulative.to_dict("records"),
        "p1_positive_rate": float(df.loc[p1, "gold_label"].mean()) if p1.any() else None,
        "p1_p2_positive_rate": float(df.loc[p2, "gold_label"].mean()) if p2.any() else None,
        "p1_p2_p3_positive_rate": float(df.loc[p3, "gold_label"].mean()) if p3.any() else None,
        "p1_recall": float(df.loc[p1, "gold_label"].sum() / yes_total),
        "p1_p2_recall": float(df.loc[p2, "gold_label"].sum() / yes_total),
        "p1_p2_p3_recall": float(df.loc[p3, "gold_label"].sum() / yes_total),
        "high_score_gold_no_count": len(high_no),
        "gold_yes_in_p4_p5_count": len(yes_low),
        "interpretation": (
            "Descriptive reviewer-policy audit on the current 70 adjudicated "
            "binary labels; not model performance and not threshold optimization."
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUT / "gold_policy_evaluation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tiers.to_csv(
        OUT / "priority_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cumulative.to_csv(
        OUT / "cumulative_queue_enrichment.csv",
        index=False,
        encoding="utf-8-sig",
    )
    high_no.to_csv(
        OUT / "high_score_gold_no.csv",
        index=False,
        encoding="utf-8-sig",
    )
    yes_low.to_csv(
        OUT / "gold_yes_low_priority.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUT / "summary.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("RAKSHAK REVIEWER POLICY v1.4 — 70-LABEL AUDIT")
    print("=" * 80)
    print(json.dumps(output, indent=2))

    print("\nPriority metrics:")
    print(tiers.to_string(index=False))

    print("\nHigh-score gold NO:")
    if len(high_no):
        print(
            high_no[
                [
                    "report_id",
                    "sif_precursor_probability",
                    "review_priority",
                    "hazards",
                    "exposures",
                    "pathways",
                ]
            ].to_string(index=False)
        )
    else:
        print("None")

    print("\nGold YES in P4/P5:")
    if len(yes_low):
        print(
            yes_low[
                [
                    "report_id",
                    "sif_precursor_probability",
                    "review_priority",
                    "hazards",
                    "exposures",
                    "pathways",
                ]
            ].to_string(index=False)
        )
    else:
        print("None")

    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
