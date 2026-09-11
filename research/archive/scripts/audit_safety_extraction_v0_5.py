"""
RAKSHAK Safety Extraction Audit v0.5
====================================

Audits v0.4 extraction against the human annotation notes in
resolved_annotations_v0.2.csv.

The audit is intentionally diagnostic:
- presence/absence coverage
- evidence coverage
- lightweight semantic/token overlap
- category-level hints

It does NOT treat annotation notes as exact gold spans.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_PATH = PROJECT_ROOT / "data" / "annotations" / "resolved_annotations_v0.2.csv"
EXTRACTION_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "safety_information_extraction_v0.4"
    / "safety_extraction_flat.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "safety_extraction_audit_v0.4"


CATEGORY_MAP = {
    "hazards": [
        "hazard_notes", "hazards_notes", "hazard_note", "hazards",
        "hazard", "hazard_notes_v0_3",
    ],
    "exposures": [
        "exposure_notes", "exposures_notes", "exposure_note", "exposures",
        "exposure",
    ],
    "consequences": [
        "consequence_notes", "consequences_notes", "consequence_note",
        "consequences", "consequence",
    ],
    "barriers": [
        "barrier_notes", "barriers_notes", "barrier_note", "barriers",
        "barrier",
    ],
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


def normalize(s: str) -> str:
    s = clean(s).lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s: str) -> set[str]:
    stop = {
        "the", "and", "with", "from", "was", "were", "that", "this",
        "worker", "person", "employee", "operator", "incident", "during",
        "while", "his", "her", "their", "into", "onto", "near", "due",
        "because", "which", "when", "then", "after", "before", "a", "an",
        "of", "to", "in", "on", "for", "at", "by", "as", "is", "are",
    }
    return {t for t in normalize(s).split() if len(t) >= 3 and t not in stop}


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols:
            return cols[candidate.lower()]
    return None


def overlap(human: str, extracted: str) -> float:
    h = tokens(human)
    e = tokens(extracted)
    if not h or not e:
        return 0.0
    return len(h & e) / len(h)


def audit_hint(human: str, extracted: str, evidence: str) -> str:
    hp = bool(clean(human))
    ep = bool(clean(extracted))
    evp = bool(clean(evidence))

    if hp and not ep:
        return "MISSED_CATEGORY"
    if not hp and ep:
        return "POSSIBLE_FALSE_POSITIVE"
    if not hp and not ep:
        return "NO_SIGNAL"
    ov = overlap(human, extracted)
    if ov >= 0.50:
        return "REASONABLE_LEXICAL_OVERLAP"
    if ov > 0:
        return "PARTIAL_COVERAGE"
    if evp:
        return "CATEGORY_PRESENT_BUT_LOW_LEXICAL_OVERLAP"
    return "CATEGORY_PRESENT_NO_OVERLAP"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(ANNOTATION_PATH)
    if not EXTRACTION_PATH.exists():
        raise FileNotFoundError(EXTRACTION_PATH)

    ann = pd.read_csv(ANNOTATION_PATH)
    ext = pd.read_csv(EXTRACTION_PATH)

    if "report_id" not in ann.columns or "report_id" not in ext.columns:
        raise ValueError("Both files must contain report_id")

    ann = ann.drop_duplicates("report_id")
    ext = ext.drop_duplicates("report_id")

    merged = ann.merge(
        ext,
        on="report_id",
        how="inner",
        suffixes=("_human", "_extracted"),
    )

    # Resolve actual human note columns before building the audit.
    human_cols = {
        category: first_existing(ann, candidates)
        for category, candidates in CATEGORY_MAP.items()
    }

    missing_human = [k for k, v in human_cols.items() if v is None]
    if missing_human:
        raise ValueError(
            "Could not identify human note columns for: "
            + ", ".join(missing_human)
            + f"\nAvailable columns: {ann.columns.tolist()}"
        )

    # Extraction column names are explicit in v0.4.
    extraction_cols = {
        "hazards": ("hazards", "hazard_evidence"),
        "exposures": ("exposures", "exposure_evidence"),
        "consequences": ("consequences", "consequence_evidence"),
        "barriers": ("barriers", "barrier_evidence"),
    }

    rows = []
    for _, row in merged.iterrows():
        out = {"report_id": clean(row["report_id"])}

        for category, (ext_col, ev_col) in extraction_cols.items():
            human_col = human_cols[category]

            # Since merge suffixes apply only to overlapping names, extraction
            # columns normally retain their names. Resolve defensively.
            actual_ext = ext_col if ext_col in merged.columns else f"{ext_col}_extracted"
            actual_ev = ev_col if ev_col in merged.columns else f"{ev_col}_extracted"

            human_val = clean(row.get(human_col))
            ext_val = clean(row.get(actual_ext))
            ev_val = clean(row.get(actual_ev))

            ov = overlap(human_val, ext_val)

            out[f"{category}_human"] = human_val
            out[f"{category}_extracted"] = ext_val
            out[f"{category}_evidence"] = ev_val
            out[f"{category}_human_present"] = bool(human_val)
            out[f"{category}_extracted_present"] = bool(ext_val)
            out[f"{category}_evidence_present"] = bool(ev_val)
            out[f"{category}_token_recall"] = round(ov, 4)
            out[f"{category}_audit_hint"] = audit_hint(human_val, ext_val, ev_val)

        rows.append(out)

    audit = pd.DataFrame(rows)
    case_path = OUTPUT_DIR / "case_audit.csv"
    audit.to_csv(case_path, index=False, encoding="utf-8-sig")

    summary = {"version": "0.5", "records_audited": int(len(audit)), "categories": {}}

    for category in extraction_cols:
        hp = audit[f"{category}_human_present"]
        ep = audit[f"{category}_extracted_present"]
        ev = audit[f"{category}_evidence_present"]
        both = hp & ep

        summary["categories"][category] = {
            "human_present": int(hp.sum()),
            "extracted_present": int(ep.sum()),
            "evidence_present": int(ev.sum()),
            "missed_when_human_present": int((hp & ~ep).sum()),
            "possible_false_positive": int((~hp & ep).sum()),
            "partial_or_better": int(
                (both & (audit[f"{category}_token_recall"] > 0)).sum()
            ) if len(audit) else 0,
            "evidence_missing_when_human_present": int((hp & ~ev).sum()),
            "miss_rate_when_human_present_pct": (
                round(100 * int((hp & ~ep).sum()) / int(hp.sum()), 2)
                if int(hp.sum()) else 0.0
            ),
            "mean_token_recall_when_human_present": (
                round(
                    float(audit.loc[hp, f"{category}_token_recall"].mean()),
                    4,
                ) if int(hp.sum()) else 0.0
            ),
        }

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 78)
    print("RAKSHAK SAFETY EXTRACTION AUDIT v0.4")
    print("=" * 78)
    print(f"Records audited : {len(audit)}")
    print()

    for category in extraction_cols:
        c = summary["categories"][category]
        print(category.upper())
        print(f"  Human present                       : {c['human_present']}")
        print(f"  Extracted present                   : {c['extracted_present']}")
        print(f"  Evidence present                    : {c['evidence_present']}")
        print(f"  Missed when human present           : {c['missed_when_human_present']}")
        print(f"  Possible false positive             : {c['possible_false_positive']}")
        print(f"  Partial/reasonable overlap          : {c['partial_or_better']}")
        print(f"  Evidence missing when human present : {c['evidence_missing_when_human_present']}")
        print(f"  Missed rate when human present      : {c['miss_rate_when_human_present_pct']:.1f}%")
        print(f"  Mean token recall                   : {c['mean_token_recall_when_human_present']:.3f}")
        print()

    print("=" * 78)
    print("FILES")
    print("=" * 78)
    print(f"Case audit : {case_path}")
    print(f"Summary    : {summary_path}")
    print("=" * 78)
    print("AUDIT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
