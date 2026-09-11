"""
RAKSHAK Evidence-Aware Ranking Audit v0.6
=========================================

Audits the outputs created by train_evidence_aware_sif_ranking_v0_4.py.

Goals
-----
1. Inspect top-ranked test reports.
2. Quantify ranking quality at K.
3. Check whether evidence spans exist for ranked reports.
4. Check whether LSR candidates have supporting evidence.
5. Flag suspicious mappings:
   - score high but no precursor evidence
   - barrier/LSR claims without corresponding evidence
   - reports whose ranking contradicts the adjudicated binary label
6. Produce a compact reviewer-oriented audit.

This is an audit tool, not a model trainer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANKING_DIR = PROJECT_ROOT / "experiments" / "sif_evidence_aware_ranking_v0.4"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "sif_evidence_aware_ranking_v0.6_audit"

RANKED_PATH = RANKING_DIR / "ranked_reports.csv"
QUEUE_PATH = RANKING_DIR / "reviewer_queue.csv"
EVIDENCE_PATH = RANKING_DIR / "evidence_spans.csv"
LSR_PATH = RANKING_DIR / "lsr_candidates.csv"



# Broader narrative barrier/control vocabulary for diagnostic purposes.
# These are NOT treated as verified barrier states.
CONTROL_PATTERNS = {
    "PPE": [
        r"\bgloves?\b", r"\bhelmet\b", r"\bsafety shoes?\b",
        r"\bgoggles?\b", r"\bface shield\b", r"\brespirator\b",
        r"\bppe\b",
    ],
    "Guarding": [
        r"\bguard\b", r"\bguarding\b", r"\bcover\b", r"\bbarrier\b",
        r"\bbarricad", r"\bfence\b", r"\bhandrail\b",
    ],
    "Isolation": [
        r"\bisolat(?:e|ed|ion)\b", r"\blockout\b", r"\block out\b",
        r"\bLOTO\b", r"\bde-energ", r"\bdeenerg",
    ],
    "Fall protection": [
        r"\bharness\b", r"\blifeline\b", r"\bfall protection\b",
        r"\bscaffold\b", r"\bguardrail\b",
    ],
    "Permit / procedure": [
        r"\bpermit\b", r"\bJSA\b", r"\bjob safety analysis\b",
        r"\brisk assessment\b", r"\bprocedure\b", r"\bchecklist\b",
    ],
    "Exclusion / distance": [
        r"\bsafe distance\b", r"\bexclusion zone\b",
        r"\brestricted area\b", r"\bstand clear\b", r"\bkeep clear\b",
        r"\baway from\b",
    ],
    "Lifting controls": [
        r"\bsling\b", r"\bshackle\b", r"\btag line\b", r"\btagline\b",
        r"\bcertified\b.*\blift", r"\b lifting plan\b",
    ],
}

GAP_PATTERNS = {
    "missing_control": [
        r"\bwithout\b.*\b(?:gloves?|helmet|goggles?|face shield|ppe|guard|harness|lifeline)\b",
        r"\bno\b.*\b(?:guard|harness|lifeline|barricad|isolation|permit|ppe)\b",
        r"\bnot\b.*\b(?:isolated|barricaded|guarded|protected)\b",
        r"\bfailed to\b.*\b(?:isolate|lockout|barricad|wear|use)\b",
        r"\bdid not\b.*\b(?:wear|isolate|lock|secure|barricad|follow)\b",
    ],
    "control_present": [
        r"\bwear(?:ing)?\b", r"\bused\b", r"\bprovided\b", r"\binstalled\b",
        r"\bsecured\b", r"\bisolated\b", r"\bguarded\b", r"\bbarricaded\b",
        r"\bprotected\b", r"\bfollowed\b",
    ],
}

def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required output: {path}")


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (RANKED_PATH, QUEUE_PATH, EVIDENCE_PATH, LSR_PATH):
        require(path)

    ranked = pd.read_csv(RANKED_PATH)
    queue = pd.read_csv(QUEUE_PATH)
    evidence = pd.read_csv(EVIDENCE_PATH)
    lsr = pd.read_csv(LSR_PATH)

    required_ranked = {
        "report_id",
        "description",
        "sif_label",
        "split",
        "sif_precursor_probability",
        "precursor_evidence_count",
        "barrier_evidence_count",
        "lsr_candidate_count",
    }
    missing = required_ranked - set(ranked.columns)
    if missing:
        raise ValueError(f"ranked_reports.csv missing columns: {sorted(missing)}")

    return ranked, queue, evidence, lsr


def precision_at_k(df: pd.DataFrame, k: int) -> float:
    top = df.head(k)
    if len(top) == 0:
        return float("nan")
    return float((top["sif_label"].astype(str).str.upper() == "YES").mean())


def recall_at_k(df: pd.DataFrame, k: int) -> float:
    total_positive = int(
        df["sif_label"].astype(str).str.upper().eq("YES").sum()
    )
    if total_positive == 0:
        return float("nan")
    top = df.head(k)
    hits = int(top["sif_label"].astype(str).str.upper().eq("YES").sum())
    return float(hits / total_positive)


def audit_narrative_barriers(text: str) -> tuple[list[str], list[str], list[str]]:
    import re

    controls = []
    gaps = []
    sentences = sentence_split = [
        s.strip(" -•\t")
        for s in re.split(r"(?<=[.!?;])\s+|\n+", str(text or ""))
        if s.strip()
    ]

    for sent in sentences:
        for name, patterns in CONTROL_PATTERNS.items():
            if any(re.search(p, sent, flags=re.I) for p in patterns):
                controls.append(name)

        for gap_name, patterns in GAP_PATTERNS.items():
            if any(re.search(p, sent, flags=re.I) for p in patterns):
                gaps.append(gap_name)

    return sorted(set(controls)), sorted(set(gaps)), sentences



def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ranked, queue, evidence, lsr = load()

    # Test ranking only.
    test = ranked[
        ranked["split"].astype(str).str.lower().eq("test")
    ].copy()

    test["probability"] = pd.to_numeric(
        test["sif_precursor_probability"], errors="coerce"
    )
    test["true_yes"] = (
        test["sif_label"].astype(str).str.upper() == "YES"
    ).astype(int)

    test = test.sort_values(
        ["probability", "report_id"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    narrative_rows = []
    for _, row in test.iterrows():
        controls, gaps, _ = audit_narrative_barriers(row["description"])
        narrative_rows.append({
            "report_id": str(row["report_id"]),
            "true_label": str(row["sif_label"]),
            "probability": float(row["probability"]),
            "control_categories_mentioned": " | ".join(controls),
            "barrier_gap_categories_mentioned": " | ".join(gaps),
            "control_category_count": len(controls),
            "barrier_gap_category_count": len(gaps),
        })
    narrative_barriers = pd.DataFrame(narrative_rows)

    # Ranking metrics.
    ranking_metrics = {
        "test_records": int(len(test)),
        "test_yes": int(test["true_yes"].sum()),
        "test_no": int((test["true_yes"] == 0).sum()),
        "test_pr_auc": (
            float(average_precision_score(test["true_yes"], test["probability"]))
            if test["true_yes"].nunique() == 2
            else None
        ),
    }

    for k in (5, 10, 15, 20):
        k_eff = min(k, len(test))
        ranking_metrics[f"precision_at_{k}"] = precision_at_k(test, k_eff)
        ranking_metrics[f"recall_at_{k}"] = recall_at_k(test, k_eff)

    # Evidence joins.
    evidence_counts = (
        evidence.groupby(["report_id", "evidence_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    evidence_counts.columns.name = None

    if "precursor" not in evidence_counts.columns:
        evidence_counts["precursor"] = 0
    if "barrier" not in evidence_counts.columns:
        evidence_counts["barrier"] = 0
    if "lsr" not in evidence_counts.columns:
        evidence_counts["lsr"] = 0

    lsr_counts = (
        lsr.groupby("report_id")
        .size()
        .rename("lsr_rows")
        .reset_index()
    )

    audit = test.merge(
        evidence_counts[
            ["report_id", "precursor", "barrier", "lsr"]
        ],
        on="report_id",
        how="left",
    ).merge(
        lsr_counts,
        on="report_id",
        how="left",
    )

    for c in ("precursor", "barrier", "lsr", "lsr_rows"):
        audit[c] = audit[c].fillna(0).astype(int)

    # Suspicion flags.
    audit["flag_no_precursor_evidence"] = audit["precursor"] == 0
    audit["flag_high_score_no_precursor"] = (
        (audit["probability"] >= 0.50)
        & audit["flag_no_precursor_evidence"]
    )
    audit["flag_lsr_without_lsr_evidence"] = (
        (audit["lsr_candidate_count"].fillna(0).astype(float) > 0)
        & (audit["lsr"] == 0)
    )
    audit["flag_barrier_without_barrier_evidence"] = (
        (audit["barrier_evidence_count"].fillna(0).astype(float) > 0)
        & (audit["barrier"] == 0)
    )

    # Contradiction is not necessarily a model failure; it is a review candidate.
    audit["flag_high_score_label_no"] = (
        (audit["probability"] >= 0.50)
        & (audit["true_yes"] == 0)
    )
    audit["flag_low_score_label_yes"] = (
        (audit["probability"] < 0.50)
        & (audit["true_yes"] == 1)
    )

    # Human-review priority:
    # model score dominates, evidence density adds modest weight.
    audit["review_score"] = (
        audit["probability"].fillna(0.0)
        + 0.02 * audit["precursor"]
        + 0.02 * audit["barrier"]
        + 0.01 * audit["lsr"]
    )

    audit = audit.sort_values(
        ["review_score", "probability", "report_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    # Top reviewer list.
    reviewer_columns = [
        "report_id",
        "probability",
        "sif_label",
        "precursor_evidence_count",
        "barrier_evidence_count",
        "lsr_candidate_count",
        "precursor",
        "barrier",
        "lsr",
        "top_precursor_mechanisms",
        "barrier_signals",
        "lsr_candidates",
        "flag_high_score_no_precursor",
        "flag_lsr_without_lsr_evidence",
        "flag_barrier_without_barrier_evidence",
        "flag_high_score_label_no",
        "flag_low_score_label_yes",
    ]
    reviewer_columns = [
        c for c in reviewer_columns if c in audit.columns
    ]

    reviewer_top = audit[reviewer_columns].head(20)

    # Evidence quality summary.
    evidence_quality = {
        "test_with_precursor_evidence_rate": float(
            (audit["precursor"] > 0).mean()
        ),
        "test_with_barrier_evidence_rate": float(
            (audit["barrier"] > 0).mean()
        ),
        "test_with_lsr_evidence_rate": float(
            (audit["lsr"] > 0).mean()
        ),
        "high_score_no_precursor_count": int(
            audit["flag_high_score_no_precursor"].sum()
        ),
        "lsr_without_evidence_count": int(
            audit["flag_lsr_without_lsr_evidence"].sum()
        ),
        "barrier_without_evidence_count": int(
            audit["flag_barrier_without_barrier_evidence"].sum()
        ),
        "high_score_false_positive_count": int(
            audit["flag_high_score_label_no"].sum()
        ),
        "low_score_false_negative_count": int(
            audit["flag_low_score_label_yes"].sum()
        ),
    }

    # Merge broader narrative barrier diagnostics.
    audit = audit.merge(
        narrative_barriers,
        on=["report_id"],
        how="left",
        suffixes=("", "_narrative"),
    )

    # Save exact audit files.
    narrative_barriers.to_csv(
        OUTPUT_DIR / "narrative_barrier_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit.to_csv(
        OUTPUT_DIR / "test_ranking_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reviewer_top.to_csv(
        OUTPUT_DIR / "reviewer_top20_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    flagged = audit[
        audit[
            [
                "flag_high_score_no_precursor",
                "flag_lsr_without_lsr_evidence",
                "flag_barrier_without_barrier_evidence",
                "flag_high_score_label_no",
                "flag_low_score_label_yes",
            ]
        ].any(axis=1)
    ].copy()

    flagged.to_csv(
        OUTPUT_DIR / "flagged_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "experiment": "RAKSHAK Evidence-Aware Ranking Audit v0.6",
        "ranking_metrics": ranking_metrics,
        "evidence_quality": evidence_quality,
        "narrative_barrier_diagnostics": {
            "test_with_any_control_mentioned_rate": float(
                (narrative_barriers["control_category_count"] > 0).mean()
            ),
            "test_with_any_gap_mentioned_rate": float(
                (narrative_barriers["barrier_gap_category_count"] > 0).mean()
            ),
        },
        "files": {
            "test_ranking_audit": "test_ranking_audit.csv",
            "reviewer_top20": "reviewer_top20_audit.csv",
            "flagged_cases": "flagged_cases.csv",
        },
        "interpretation": (
            "This audit evaluates whether the ranking is operationally supported "
            "by narrative evidence. It does not establish safety-critical accuracy "
            "and does not validate heuristic LSR mappings as ground truth."
        ),
    }
    (OUTPUT_DIR / "audit_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("RAKSHAK EVIDENCE-AWARE RANKING AUDIT v0.6")
    print("=" * 78)
    print(f"Test records                    : {ranking_metrics['test_records']}")
    print(f"Test PR-AUC                     : {ranking_metrics['test_pr_auc']:.4f}")
    print(f"Precision@5                     : {ranking_metrics['precision_at_5']:.4f}")
    print(f"Recall@5                        : {ranking_metrics['recall_at_5']:.4f}")
    print(f"Precision@10                    : {ranking_metrics['precision_at_10']:.4f}")
    print(f"Recall@10                       : {ranking_metrics['recall_at_10']:.4f}")
    print()
    print(f"Test precursor evidence rate   : {evidence_quality['test_with_precursor_evidence_rate']:.1%}")
    print(f"Test barrier evidence rate     : {evidence_quality['test_with_barrier_evidence_rate']:.1%}")
    print(f"Test LSR evidence rate         : {evidence_quality['test_with_lsr_evidence_rate']:.1%}")
    print(f"High-score/no-evidence flags   : {evidence_quality['high_score_no_precursor_count']}")
    print(f"LSR-without-evidence flags     : {evidence_quality['lsr_without_evidence_count']}")
    print(f"Barrier-without-evidence flags : {evidence_quality['barrier_without_evidence_count']}")
    print(f"High-score false positives     : {evidence_quality['high_score_false_positive_count']}")
    print(f"Low-score false negatives      : {evidence_quality['low_score_false_negative_count']}")
    print()
    print(f"Saved audit to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
