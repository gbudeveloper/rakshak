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
    / "safety_extraction_audit_v0.1"
)

CASE_AUDIT_PATH = OUTPUT_DIR / "case_audit.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


CATEGORY_MAP = {
    "hazards": [
        "hazard_notes",
        "hazards",
    ],
    "exposures": [
        "exposure_notes",
        "exposures",
    ],
    "consequences": [
        "consequence_notes",
        "consequences",
    ],
    "barriers": [
        "barrier_notes",
        "barriers",
    ],
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


def normalize(text: object) -> str:
    return (
        str(text)
        .lower()
        .replace("\u00a0", " ")
    )


def tokens(text: object) -> set[str]:
    value = normalize(text)
    words = re.findall(
        r"[a-z][a-z0-9-]{2,}",
        value,
    )
    return {
        word
        for word in words
        if word not in STOPWORDS
    }


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    label: str,
) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def list_values(value: object) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    return [
        chunk.strip()
        for chunk in re.split(r";|\|\|", text)
        if chunk.strip()
    ]


def evidence_concepts(
    extraction_row: pd.Series,
    category: str,
) -> list[str]:
    label_column = f"{category}"
    evidence_column = f"{category[:-1]}_evidence"

    values = []

    if label_column in extraction_row:
        values.extend(
            list_values(
                extraction_row[label_column]
            )
        )

    if evidence_column in extraction_row:
        values.extend(
            list_values(
                extraction_row[evidence_column]
            )
        )

    return values


def category_overlap(
    human_text: str,
    extracted_text: str,
) -> dict:
    human = tokens(human_text)
    extracted = tokens(extracted_text)

    if not human:
        return {
            "human_present": False,
            "extracted_present": bool(extracted),
            "token_overlap": 0,
            "token_recall": None,
        }

    overlap = len(human & extracted)

    return {
        "human_present": True,
        "extracted_present": bool(extracted),
        "token_overlap": overlap,
        "token_recall": float(
            overlap / max(len(human), 1)
        ),
    }


def manual_gap_hint(
    human_text: str,
    extracted_text: str,
) -> str:
    human = tokens(human_text)
    extracted = tokens(extracted_text)

    if not human and not extracted:
        return "NO_SIGNAL"

    if human and not extracted:
        return "MISSED_CATEGORY"

    if not human and extracted:
        return "POSSIBLE_FALSE_POSITIVE"

    overlap = human & extracted

    if len(overlap) == 0:
        return "LOW_SEMANTIC_OVERLAP"

    if len(overlap) < max(2, int(0.2 * len(human))):
        return "PARTIAL_COVERAGE"

    return "REASONABLE_LEXICAL_OVERLAP"


def main() -> None:
    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(
            f"Missing annotation file:\n{ANNOTATION_PATH}"
        )

    if not EXTRACTION_PATH.exists():
        raise FileNotFoundError(
            f"Missing extraction file:\n{EXTRACTION_PATH}\n"
            "Run safety_information_extractor_v0.3.py --limit 0 first."
        )

    annotations = pd.read_csv(
        ANNOTATION_PATH
    )

    extraction = pd.read_csv(
        EXTRACTION_PATH
    )

    if "report_id" not in annotations.columns:
        raise ValueError(
            "Annotations missing report_id."
        )

    if "report_id" not in extraction.columns:
        raise ValueError(
            "Extraction missing report_id."
        )

    annotations["report_id"] = (
        annotations["report_id"]
        .astype(str)
        .str.strip()
    )
    extraction["report_id"] = (
        extraction["report_id"]
        .astype(str)
        .str.strip()
    )

    merged = annotations.merge(
        extraction,
        on="report_id",
        how="left",
        suffixes=("_human", "_extracted"),
        validate="one_to_one",
    )

    if merged["report_id"].isna().any():
        raise RuntimeError(
            "Extraction coverage does not match annotation records."
        )

    rows = []

    category_stats = {
        category: {
            "human_present": 0,
            "extracted_present": 0,
            "missed_category": 0,
            "possible_false_positive": 0,
            "partial_or_reasonable": 0,
        }
        for category in CATEGORY_MAP
    }

    for _, row in merged.iterrows():
        case = {
            "report_id": row["report_id"],
        }

        for category, columns in CATEGORY_MAP.items():
            human_col = find_column(
                merged,
                columns[:1],
                f"human {category}",
            )

            # Extraction file uses plural category names.
            extracted_col = category if category in merged.columns else None

            if human_col is None:
                human_text = ""
            else:
                human_text = (
                    "" if pd.isna(row[human_col])
                    else str(row[human_col])
                )

            if extracted_col is None:
                extracted_text = ""
            else:
                extracted_text = (
                    "" if pd.isna(row[extracted_col])
                    else str(row[extracted_col])
                )

            overlap = category_overlap(
                human_text,
                extracted_text,
            )

            hint = manual_gap_hint(
                human_text,
                extracted_text,
            )

            case[f"{category}_human"] = human_text
            case[f"{category}_extracted"] = extracted_text
            case[f"{category}_human_present"] = bool(
                human_text.strip()
            )
            case[f"{category}_extracted_present"] = bool(
                extracted_text.strip()
            )
            case[f"{category}_token_recall"] = overlap[
                "token_recall"
            ]
            case[f"{category}_audit_hint"] = hint

            stats = category_stats[category]

            if human_text.strip():
                stats["human_present"] += 1

            if extracted_text.strip():
                stats["extracted_present"] += 1

            if hint == "MISSED_CATEGORY":
                stats["missed_category"] += 1

            if hint == "POSSIBLE_FALSE_POSITIVE":
                stats["possible_false_positive"] += 1

            if hint in {
                "PARTIAL_COVERAGE",
                "REASONABLE_LEXICAL_OVERLAP",
            }:
                stats["partial_or_reasonable"] += 1

        rows.append(case)

    case_audit = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    case_audit.to_csv(
        CASE_AUDIT_PATH,
        index=False,
        encoding="utf-8",
    )

    # Approximate category coverage summaries.
    # These are diagnostic proxies, not formal precision/recall,
    # because human notes are not gold span annotations.
    category_summary = {}

    for category, stats in category_stats.items():
        human_present = stats["human_present"]

        category_summary[category] = {
            **stats,
            "human_present_rate": float(
                human_present / len(merged)
            ) if len(merged) else 0.0,
            "extracted_present_rate": float(
                stats["extracted_present"] / len(merged)
            ) if len(merged) else 0.0,
            "missed_when_human_present_rate": float(
                stats["missed_category"] / human_present
            ) if human_present else 0.0,
        }

    # Identify records with multiple missed categories.
    missed_counts = []

    for _, row in case_audit.iterrows():
        missed = sum(
            row[f"{category}_audit_hint"]
            == "MISSED_CATEGORY"
            for category in CATEGORY_MAP
        )

        false_positive = sum(
            row[f"{category}_audit_hint"]
            == "POSSIBLE_FALSE_POSITIVE"
            for category in CATEGORY_MAP
        )

        missed_counts.append(
            {
                "report_id": row["report_id"],
                "missed_category_count": int(missed),
                "possible_false_positive_count": int(
                    false_positive
                ),
            }
        )

    difficult_cases = sorted(
        missed_counts,
        key=lambda item: (
            -item["missed_category_count"],
            -item["possible_false_positive_count"],
        ),
    )

    summary = {
        "version": "safety_extraction_audit_v0.1",
        "records": int(len(merged)),
        "category_summary": category_summary,
        "most_missed_records": difficult_cases[:15],
        "interpretation": (
            "This audit compares rule-based extraction against "
            "human annotation notes. It is a diagnostic coverage "
            "proxy, not a formal span-level precision/recall "
            "evaluation because the human notes are not annotated "
            "with exact spans."
        ),
        "recommendation": (
            "Do not connect the extractor directly to SIF or LSR "
            "decisions until the missed-category and false-positive "
            "patterns have been reviewed."
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print("RAKSHAK SAFETY EXTRACTION AUDIT v0.1")
    print("=" * 78)
    print(f"Records audited : {len(merged)}")
    print()

    for category, stats in category_summary.items():
        print(category.upper())
        print(
            f"  Human present                  : "
            f"{stats['human_present']}"
        )
        print(
            f"  Extracted present              : "
            f"{stats['extracted_present']}"
        )
        print(
            f"  Missed when human present      : "
            f"{stats['missed_category']}"
        )
        print(
            f"  Possible false positive        : "
            f"{stats['possible_false_positive']}"
        )
        print(
            f"  Missed rate when human present : "
            f"{stats['missed_when_human_present_rate']:.1%}"
        )
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
