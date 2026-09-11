"""
Evaluate RAKSHAK reviewer-priority policy v1.3 on the 70 adjudicated
binary-labelled records.

This is an audit only:
- Does not retrain the classifier.
- Does not change probabilities.
- Does not optimize the priority thresholds against the test set.
- Uses the existing adjudicated labels only to measure queue enrichment.

Outputs:
  experiments\reviewer_policy_eval_v1.3\
    gold_policy_evaluation.csv
    priority_label_crosstab.csv
    priority_metrics.csv
    summary.json

The 20 locked test cases remain visible for diagnostics, but no threshold
optimization is performed.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PRED=ROOT/"experiments"/"industrial_safety_predictions_v1.3.csv"
LABEL=ROOT/"data"/"annotations"/"resolved_annotations_v0.2.csv"
OUT=ROOT/"experiments"/"reviewer_policy_eval_v1.3"

PRIORITIES=["P1","P2","P3","P4","P5"]


def main():
    if not PRED.exists():
        raise FileNotFoundError(PRED)
    if not LABEL.exists():
        raise FileNotFoundError(LABEL)

    pred=pd.read_csv(PRED)
    lab=pd.read_csv(LABEL)

    pred["report_id"]=pred["report_id"].astype(str)
    lab["report_id"]=lab["report_id"].astype(str)
    lab["final_sif_potential"]=(
        lab["final_sif_potential"].astype(str).str.strip().str.upper()
    )
    lab=lab[lab["final_sif_potential"].isin(["YES","NO"])].copy()
    lab["gold_label"]=(lab["final_sif_potential"]=="YES").astype(int)

    keep_cols=[
        "report_id","sif_precursor_probability","review_priority",
        "review_priority_reason","hazards","exposures","pathways",
        "complete_pathway","sif_context_status","description"
    ]
    merged=pred[keep_cols].merge(
        lab[["report_id","gold_label","final_sif_potential"]],
        on="report_id",how="inner",validate="one_to_one"
    )

    if len(merged)!=70:
        raise ValueError(f"Expected 70 binary-labelled records, got {len(merged)}")

    # Queue enrichment: prevalence in each reviewer tier.
    tier_rows=[]
    for p in PRIORITIES:
        s=merged[merged.review_priority==p]
        positives=int(s.gold_label.sum())
        total=len(s)
        tier_rows.append({
            "priority":p,
            "records":total,
            "gold_yes":positives,
            "gold_no":total-positives,
            "gold_yes_rate":(positives/total if total else None),
            "mean_probability":(float(s.sif_precursor_probability.mean()) if total else None),
            "median_probability":(float(s.sif_precursor_probability.median()) if total else None),
        })
    tier_metrics=pd.DataFrame(tier_rows)

    # Cumulative reviewer queue enrichment:
    cumulative=[]
    for k,p in enumerate(PRIORITIES,1):
        s=merged[merged.review_priority.isin(PRIORITIES[:k])]
        cumulative.append({
            "queue":f"P1-P{k}",
            "records":len(s),
            "gold_yes":int(s.gold_label.sum()),
            "gold_yes_rate":float(s.gold_label.mean()) if len(s) else None,
        })
    cumulative_df=pd.DataFrame(cumulative)

    # Simple ranking metrics, descriptive only.
    order={"P1":1,"P2":2,"P3":3,"P4":4,"P5":5}
    rank=merged.review_priority.map(order).astype(int)
    merged["_priority_rank"]=rank

    # Quality of P1 as a high-value review tier.
    p1=merged.review_priority.eq("P1")
    p2_or_better=merged.review_priority.isin(["P1","P2"])
    p3_or_better=merged.review_priority.isin(["P1","P2","P3"])

    metrics=pd.DataFrame([
        {
            "metric":"P1 positive rate",
            "value":float(merged.loc[p1,"gold_label"].mean()) if p1.any() else None,
            "definition":"Gold YES fraction among P1 records"
        },
        {
            "metric":"P1 recall",
            "value":float(merged.loc[p1,"gold_label"].sum()/merged.gold_label.sum()) if merged.gold_label.sum() else None,
            "definition":"Gold YES captured by P1"
        },
        {
            "metric":"P1-P2 positive rate",
            "value":float(merged.loc[p2_or_better,"gold_label"].mean()) if p2_or_better.any() else None,
            "definition":"Gold YES fraction among P1/P2 records"
        },
        {
            "metric":"P1-P2 recall",
            "value":float(merged.loc[p2_or_better,"gold_label"].sum()/merged.gold_label.sum()) if merged.gold_label.sum() else None,
            "definition":"Gold YES captured by P1/P2"
        },
        {
            "metric":"P1-P3 positive rate",
            "value":float(merged.loc[p3_or_better,"gold_label"].mean()) if p3_or_better.any() else None,
            "definition":"Gold YES fraction among P1-P3 records"
        },
        {
            "metric":"P1-P3 recall",
            "value":float(merged.loc[p3_or_better,"gold_label"].sum()/merged.gold_label.sum()) if merged.gold_label.sum() else None,
            "definition":"Gold YES captured by P1-P3"
        },
    ])

    # Critical review cases: model probability high but gold NO.
    high_fp=merged[(merged.sif_precursor_probability>=0.75)&(merged.gold_label==0)].copy()
    high_fp=high_fp.sort_values("sif_precursor_probability",ascending=False)

    # Gold YES that landed in low tiers.
    missed=merged[(merged.gold_label==1)&merged.review_priority.isin(["P4","P5"])].copy()
    missed=missed.sort_values("sif_precursor_probability",ascending=True)

    crosstab=pd.crosstab(
        merged["review_priority"],
        merged["final_sif_potential"],
        margins=True
    ).reindex(PRIORITIES,fill_value=0)

    out_dir=OUT
    out_dir.mkdir(parents=True,exist_ok=True)

    merged.drop(columns="_priority_rank").sort_values(
        ["review_priority","sif_precursor_probability"],
        key=lambda s: s.map(order) if s.name=="review_priority" else s,
        ascending=[True,False]
    ).to_csv(out_dir/"gold_policy_evaluation.csv",index=False,encoding="utf-8-sig")

    tier_metrics.to_csv(out_dir/"priority_metrics.csv",index=False,encoding="utf-8-sig")
    cumulative_df.to_csv(out_dir/"cumulative_queue_enrichment.csv",index=False,encoding="utf-8-sig")
    crosstab.to_csv(out_dir/"priority_label_crosstab.csv",encoding="utf-8-sig")
    high_fp.to_csv(out_dir/"high_score_gold_no.csv",index=False,encoding="utf-8-sig")
    missed.to_csv(out_dir/"gold_yes_low_priority.csv",index=False,encoding="utf-8-sig")

    summary={
        "labelled_records":len(merged),
        "gold_yes":int(merged.gold_label.sum()),
        "gold_no":int((1-merged.gold_label).sum()),
        "policy_distribution":tier_metrics[["priority","records","gold_yes","gold_no","gold_yes_rate"]].to_dict("records"),
        "queue_enrichment":cumulative_df.to_dict("records"),
        "p1_positive_rate":float(merged.loc[p1,"gold_label"].mean()) if p1.any() else None,
        "p1_p2_positive_rate":float(merged.loc[p2_or_better,"gold_label"].mean()) if p2_or_better.any() else None,
        "p1_recall":float(merged.loc[p1,"gold_label"].sum()/merged.gold_label.sum()),
        "p1_p2_recall":float(merged.loc[p2_or_better,"gold_label"].sum()/merged.gold_label.sum()),
        "high_score_gold_no_count":len(high_fp),
        "gold_yes_in_p4_p5_count":len(missed),
        "interpretation":"Audit of reviewer-tier enrichment on the current 70-record adjudicated set; not threshold tuning and not a deployment performance estimate."
    }
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    print("="*80)
    print("RAKSHAK REVIEWER POLICY v1.3 — 70-LABEL AUDIT")
    print("="*80)
    print(json.dumps(summary,indent=2))
    print("\nPriority metrics:")
    print(tier_metrics.to_string(index=False))
    print("\nCumulative enrichment:")
    print(cumulative_df.to_string(index=False))
    print("\nHigh-score gold NO:")
    if len(high_fp):
        print(high_fp[["report_id","sif_precursor_probability","review_priority","hazards","exposures","pathways"]].to_string(index=False))
    else:
        print("None")
    print("\nGold YES in P4/P5:")
    if len(missed):
        print(missed[["report_id","sif_precursor_probability","review_priority","hazards","exposures","pathways"]].to_string(index=False))
    else:
        print("None")
    print("\nSaved:",out_dir)


if __name__=="__main__":
    main()
