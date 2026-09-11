from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANNOTATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "resolved_annotations_v0.2.csv"
)

EXTRACTION_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "safety_information_extraction_v0.3"
    / "safety_extraction_flat.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "safety_extraction_audit_v0.3"
)

CASE_AUDIT_PATH = OUTPUT_DIR / "case_audit.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


CATEGORY_MAP = {
    "hazards": "hazard_notes",
    "exposures": "exposure_notes",
    "consequences": "consequence_notes",
    "barriers": "barrier_notes",
}

EVIDENCE_COLUMNS = {
    "hazards": "hazard_evidence",
    "exposures": "exposure_evidence",
    "consequences": "consequence_evidence",
    "barriers": "barrier_evidence",
}

STOPWORDS = {
    "the", "and", "that", "with", "from", "this", "were", "was",
    "when", "while", "into", "onto", "near", "worker", "employee",
    "operator", "collaborator", "personnel", "during", "performed",
    "performing", "activity", "work", "working", "area", "event",
    "injury", "injured", "causing", "cause", "caused", "could",
    "would", "also", "used", "use", "made", "make", "at", "of",
    "to", "in", "on", "a", "an", "for", "by", "is", "are", "be",
    "as", "or", "their", "his", "her", "its", "had", "has", "have",
    "not", "no", "one", "two", "three", "all", "very", "more",
}


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("\u00a0", " ").strip()


def tokens(text: object) -> set[str]:
    value = clean_text(text).lower()
    words = re.findall(r"[a-z][a-z0-9-]{2,}", value)
    return {word for word in words if word not in STOPWORDS}


def list_values(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    return [
        chunk.strip()
        for chunk in re.split(r";|\|\|", text)
        if chunk.strip()
    ]


def presence(value: object) -> bool:
    return bool(clean_text(value))


def lexical_overlap(human_text: object, extracted_text: object) -> dict:
    human = tokens(human_text)
    extracted = tokens(extracted_text)

    if not human:
        return {
            "human_present": False,
            "extracted_present": bool(extracted),
            "token_overlap": 0,
            "token_recall": None,
            "jaccard": None,
        }

    overlap = len(human & extracted)
    union = len(human | extracted)

    return {
        "human_present": True,
        "extracted_present": bool(extracted),
        "token_overlap": overlap,
        "token_recall": overlap / max(len(human), 1),
        "jaccard": overlap / max(union, 1),
    }


def audit_hint(human_text: object, extracted_text: object) -> str:
    human = tokens(human_text)
    extracted = tokens(extracted_text)

    if not human and not extracted:
        return "NO_SIGNAL"
    if human and not extracted:
        return "MISSED_CATEGORY"
    if not human and extracted:
        return "POSSIBLE_FALSE_POSITIVE"

    overlap = human & extracted
    recall = len(overlap) / max(len(human), 1)

    if recall >= 0.50:
        return "REASONABLE_LEXICAL_OVERLAP"
    if recall > 0:
        return "PARTIAL_COVERAGE"
    return "LOW_SEMANTIC_OVERLAP"


def evidence_present(value: object) -> bool:
    return bool(list_values(value))


def main() -> None:
    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(f"Missing annotation file:\n{ANNOTATION_PATH}")

    if not EXTRACTION_PATH.exists():
        raise FileNotFoundError(
            f"Missing extraction file:\n{EXTRACTION_PATH}\n"
            "Run safety_information_extractor_v0.3.py --limit 0 first."
        )

    annotations = pd.read_csv(ANNOTATION_PATH)
    extraction = pd.read_csv(EXTRACTION_PATH)

    required_annotation = ["report_id", *CATEGORY_MAP.values()]
    required_extraction = ["report_id", *CATEGORY_MAP.keys(), *EVIDENCE_COLUMNS.values()]

    missing_annotation = [c for c in required_annotation if c not in annotations.columns]
    missing_extraction = [c for c in required_extraction if c not in extraction.columns]

    if missing_annotation:
        raise ValueError(f"Annotations missing columns: {missing_annotation}")
    if missing_extraction:
        raise ValueError(f"Extraction missing columns: {missing_extraction}")

    annotations["report_id"] = annotations["report_id"].astype(str).str.strip()
    extraction["report_id"] = extraction["report_id"].astype(str).str.strip()

    duplicate_ann = annotations["report_id"].duplicated().sum()
    duplicate_ext = extraction["report_id"].duplicated().sum()
    if duplicate_ann or duplicate_ext:
        raise ValueError(
            f"report_id must be unique. Annotation duplicates={duplicate_ann}, "
            f"extraction duplicates={duplicate_ext}."
        )

    merged = annotations.merge(
        extraction,
        on="report_id",
        how="left",
        suffixes=("_human", "_extracted"),
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        missing_ids = merged.loc[merged["_merge"] != "both", "report_id"].tolist()
        raise RuntimeError(
            f"Extraction coverage mismatch: {unmatched} annotation records missing. "
            f"IDs={missing_ids[:10]}"
        )
    merged = merged.drop(columns=["_merge"])

    rows: list[dict] = []
    category_stats: dict[str, dict] = {}

    for category, human_col in CATEGORY_MAP.items():
        extracted_col = category
        evidence_col = EVIDENCE_COLUMNS[category]
        category_stats[category] = {
            "human_present": 0,
            "extracted_present": 0,
            "missed_category": 0,
            "possible_false_positive": 0,
            "partial_or_reasonable": 0,
            "evidence_present": 0,
            "human_present_but_no_evidence": 0,
        }

    for _, row in merged.iterrows():
        case: dict[str, object] = {"report_id": row["report_id"]}

        for category, human_col in CATEGORY_MAP.items():
            extracted_col = (
                f"{category}_extracted"
                if f"{category}_extracted" in merged.columns
                else category
            )
            evidence_col = EVIDENCE_COLUMNS[category]

            human_text = clean_text(row.get(human_col))
            extracted_text = clean_text(row.get(extracted_col))
            evidence_text = clean_text(row.get(evidence_col))

            overlap = lexical_overlap(human_text, extracted_text)
            hint = audit_hint(human_text, extracted_text)
            human_has = presence(human_text)
            extracted_has = presence(extracted_text)
            evidence_has = evidence_present(evidence_text)

            case[f"{category}_human"] = human_text
            case[f"{category}_extracted"] = extracted_text
            case[f"{category}_evidence"] = evidence_text
            case[f"{category}_human_present"] = human_has
            case[f"{category}_extracted_present"] = extracted_has
            case[f"{category}_evidence_present"] = evidence_has
            case[f"{category}_token_overlap"] = overlap["token_overlap"]
            case[f"{category}_token_recall"] = overlap["token_recall"]
            case[f"{category}_jaccard"] = overlap["jaccard"]
            case[f"{category}_audit_hint"] = hint

            stats = category_stats[category]
            if human_has:
                stats["human_present"] += 1
            if extracted_has:
                stats["extracted_present"] += 1
            if evidence_has:
                stats["evidence_present"] += 1
            if human_has and not evidence_has:
                stats["human_present_but_no_evidence"] += 1
            if hint == "MISSED_CATEGORY":
                stats["missed_category"] += 1
            elif hint == "POSSIBLE_FALSE_POSITIVE":
                stats["possible_false_positive"] += 1
            elif hint in {"PARTIAL_COVERAGE", "REASONABLE_LEXICAL_OVERLAP"}:
                stats["partial_or_reasonable"] += 1

        rows.append(case)

    case_audit = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case_audit.to_csv(CASE_AUDIT_PATH, index=False, encoding="utf-8")

    category_summary = {}
    for category, stats in category_stats.items():
        human_present = stats["human_present"]
        total = len(merged)
        category_summary[category] = {
            **stats,
            "human_present_rate": human_present / total if total else 0.0,
            "extracted_present_rate": stats["extracted_present"] / total if total else 0.0,
            "evidence_present_rate": stats["evidence_present"] / total if total else 0.0,
            "missed_when_human_present_rate": (
                stats["missed_category"] / human_present if human_present else 0.0
            ),
            "evidence_missing_when_human_present_rate": (
                stats["human_present_but_no_evidence"] / human_present
                if human_present
                else 0.0
            ),
        }

    # Rank cases by number of missed categories, then by missing evidence.
    ranked = []
    for _, row in case_audit.iterrows():
        missed = sum(row[f"{c}_audit_hint"] == "MISSED_CATEGORY" for c in CATEGORY_MAP)
        weak = sum(
            row[f"{c}_audit_hint"] in {"LOW_SEMANTIC_OVERLAP", "PARTIAL_COVERAGE"}
            for c in CATEGORY_MAP
        )
        no_evidence = sum(
            bool(row[f"{c}_human_present"] and not row[f"{c}_evidence_present"])
            for c in CATEGORY_MAP
        )
        ranked.append(
            {
                "report_id": row["report_id"],
                "missed_category_count": int(missed),
                "weak_or_partial_category_count": int(weak),
                "human_present_without_evidence_count": int(no_evidence),
            }
        )

    ranked.sort(
        key=lambda x: (
            -x["missed_category_count"],
            -x["weak_or_partial_category_count"],
            -x["human_present_without_evidence_count"],
        )
    )

    summary = {
        "version": "safety_extraction_audit_v0.3",
        "records": int(len(merged)),
        "annotation_path": str(ANNOTATION_PATH),
        "extraction_path": str(EXTRACTION_PATH),
        "category_summary": category_summary,
        "most_difficult_records": ranked[:20],
        "interpretation": (
            "Diagnostic coverage audit. Human notes are category-level notes, "
            "not exact span annotations, so lexical overlap is only a proxy. "
            "Extraction columns are read from the post-merge *_extracted fields "
            "created by pandas suffixing, while evidence is audited separately."
        ),
        "next_step": (
            "Use this audit to identify true extraction gaps. Do not use category-level "
            "lexical recall as a production precision/recall metric."
        ),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 78)
    print("RAKSHAK SAFETY EXTRACTION AUDIT v0.2")
    print("=" * 78)
    print(f"Records audited : {len(merged)}")
    print()

    for category, stats in category_summary.items():
        print(category.upper())
        print(f"  Human present                       : {stats['human_present']}")
        print(f"  Extracted present                   : {stats['extracted_present']}")
        print(f"  Evidence present                    : {stats['evidence_present']}")
        print(f"  Missed when human present           : {stats['missed_category']}")
        print(f"  Possible false positive             : {stats['possible_false_positive']}")
        print(f"  Partial/reasonable overlap          : {stats['partial_or_reasonable']}")
        print(f"  Evidence missing when human present : {stats['human_present_but_no_evidence']}")
        print(f"  Missed rate when human present      : {stats['missed_when_human_present_rate']:.1%}")
        print()

    print("=" * 78)
    print("FILES")
    print("=" * 78)
    print(f"Case audit : {CASE_AUDIT_PATH}")
    print(f"Summary    : {SUMMARY_PATH}")
    print()
    print("=" * 78)
    print("AUDIT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
