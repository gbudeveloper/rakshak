"""
Build narrative-derived activity metadata for the full 411 unique narratives.

Source:
  data/annotations/industrial_safety_annotation_pool.csv

Uses the existing evidence-first activity ontology from
safety_information_extractor_v0_4.py. This is descriptive metadata enrichment;
it is NOT an ML feature and NOT an authoritative activity field.

Output:
  experiments/safety_information_extraction_v0.5/safety_extraction_flat.csv
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import sys

ROOT=Path(__file__).resolve().parents[1]
POOL=ROOT/"data/annotations/industrial_safety_annotation_pool.csv"
OUT=ROOT/"experiments/safety_information_extraction_v0.5"/"safety_extraction_flat.csv"

sys.path.insert(0,str(ROOT/"scripts"))
try:
    from safety_information_extractor_v0_4 import extract_activities
except Exception as exc:
    raise RuntimeError(
        "Could not import safety_information_extractor_v0_4.py. "
        "Keep that existing script in scripts/. Error: "+str(exc)
    ) from exc

def main():
    if not POOL.exists():
        raise FileNotFoundError(POOL)

    pool=pd.read_csv(POOL)
    required={"report_id","description"}
    missing=required-set(pool.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows=[]
    for rid,text in zip(
        pool["report_id"].astype(str),
        pool["description"].fillna("").astype(str),
    ):
        acts=extract_activities(text)
        labels=[]
        evidence=[]
        for a in acts:
            labels.append(a.get("label",""))
            evidence.append(a.get("evidence",""))
        rows.append({
            "report_id":rid,
            "activities":"; ".join(dict.fromkeys(x for x in labels if x)),
            "activity_evidence":"; ".join(dict.fromkeys(x for x in evidence if x)),
            "activity_count":len(acts),
        })

    out=pd.DataFrame(rows).drop_duplicates("report_id")
    OUT.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(OUT,index=False,encoding="utf-8-sig")

    print("="*80)
    print("BATCH ACTIVITY METADATA v0.5")
    print("="*80)
    print(f"Input records : {len(pool)}")
    print(f"Output rows   : {len(out)}")
    print(f"With activity : {int(out['activities'].fillna('').astype(str).str.strip().ne('').sum())}")
    print(f"Output        : {OUT}")
    print("\nTop activities:")
    vals=[]
    for x in out["activities"].fillna("").astype(str):
        vals += [v.strip() for v in x.split(";") if v.strip()]
    if vals:
        print(pd.Series(vals).value_counts().head(15).to_string())
    else:
        print("None detected")

if __name__=="__main__":
    main()
