"""
RAKSHAK Safety Extraction Error Analysis v0.1
============================================

Purpose:
    Inspect genuine misses from the corrected v0.4 extraction audit before
    changing the extractor.

Important:
    The current "token recall" metric is NOT treated as a quality metric
    here because the extractor emits normalized ontology labels while human
    notes are free-text descriptions. This script focuses on category
    presence, evidence, and the original narrative.

Reads:
    experiments/safety_extraction_audit_v0.4/case_audit.csv
    data/annotations/resolved_annotations_v0.2.csv

Writes:
    experiments/safety_extraction_error_analysis_v0.1/
        category_misses.csv
        category_summary.json

Console:
    Prints all missed cases by category with narrative + human notes +
    extracted values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIT_PATH = (
    PROJECT_ROOT / "experiments" / "safety_extraction_audit_v0.4" / "case_audit.csv"
)

ANNOTATION_PATH = (
    PROJECT_ROOT / "data" / "annotations" / "resolved_annotations_v0.2.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "safety_extraction_error_analysis_v0.1"


CATEGORY_INFO = {
    "hazards": {
        "human": "hazards_human",
        "extracted": "hazards_extracted",
        "evidence": "hazards_evidence",
        "audit": "hazards_audit_hint",
    },
    "exposures": {
        "human": "exposures_human",
        "extracted": "exposures_extracted",
        "evidence": "exposures_evidence",
        "audit": "exposures_audit_hint",
    },
    "consequences": {
        "human": "consequences_human",
        "extracted": "consequences_extracted",
        "evidence": "consequences_evidence",
        "audit": "consequences_audit_hint",
    },
    "barriers": {
        "human": "barriers_human",
        "extracted": "barriers_extracted",
        "evidence": "barriers_evidence",
        "audit": "barriers_audit_hint",
    },
}


def clean(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def main() -> None:
    if not AUDIT_PATH.exists():
        raise FileNotFoundError(f"Audit file not found: {AUDIT_PATH}")

    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(f"Annotation file not found: {ANNOTATION_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(AUDIT_PATH)
    ann = pd.read_csv(ANNOTATION_PATH)

    ann = ann.drop_duplicates("report_id")

    # Prefer narrative from annotation source. Fall back to audit data if
    # the audit was extended to include it later.
    desc_col = None
    for candidate in ["description", "normalized_description"]:
        if candidate in ann.columns:
            desc_col = candidate
            break

    if desc_col is None:
        raise ValueError(
            f"Could not find description column in annotations. "
            f"Available: {ann.columns.tolist()}"
        )

    merged = ann[["report_id", desc_col]].merge(
        audit,
        on="report_id",
        how="inner",
    )

    all_rows: List[dict] = []
    summary: Dict[str, dict] = {}

    for category, cols in CATEGORY_INFO.items():
        required = [cols["human"], cols["extracted"]]
        missing = [c for c in required if c not in merged.columns]
        if missing:
            raise ValueError(
                f"{category}: missing columns {missing}. "
                f"Available: {merged.columns.tolist()}"
            )

        misses = merged[
            merged[cols["human"]].fillna("").astype(str).str.strip().ne("")
            & merged[cols["extracted"]].fillna("").astype(str).str.strip().eq("")
        ].copy()

        summary[category] = {
            "missed_count": int(len(misses)),
            "records": [],
        }

        print()
        print("=" * 90)
        print(f"{category.upper()} MISSES: {len(misses)}")
        print("=" * 90)

        for _, row in misses.iterrows():
            report_id = clean(row["report_id"])
            description = clean(row[desc_col])
            human = clean(row[cols["human"]])
            evidence = (
                clean(row[cols["evidence"]])
                if cols["evidence"] in merged.columns
                else ""
            )

            record = {
                "report_id": report_id,
                "category": category,
                "description": description,
                "human_note": human,
                "extracted": clean(row[cols["extracted"]]),
                "extracted_evidence": evidence,
                "audit_hint": (
                    clean(row[cols["audit"]]) if cols["audit"] in merged.columns else ""
                ),
            }

            all_rows.append(record)
            summary[category]["records"].append(record)

            print(f"\n[{report_id}]")
            print(f"HUMAN NOTE      : {human}")
            print(f"EXTRACTED       : {clean(row[cols['extracted']])}")
            print(f"NARRATIVE       : {description}")

    all_path = OUTPUT_DIR / "category_misses.csv"
    pd.DataFrame(all_rows).to_csv(
        all_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_path = OUTPUT_DIR / "category_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 90)
    print(f"Misses CSV : {all_path}")
    print(f"Summary    : {summary_path}")
    print()
    print(
        "Do not optimize the old token-recall metric from the audit. "
        "The extractor uses normalized ontology labels, while human notes "
        "are free-text; lexical overlap is therefore not a valid primary metric."
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
