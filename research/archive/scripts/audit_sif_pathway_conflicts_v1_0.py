"""
RAKSHAK SIF Pathway Conflict Audit v1.0
=======================================

Purpose
-------
Audit the 5 most important current model conflicts using the ORIGINAL
incident narrative, without using annotation-derived outcome/consequence
fields.

Focus:
    hazard/energy
        ->
    human exposure
        ->
    credible SIF pathway

This is diagnostic. It does not overwrite labels and does not retrain the
model.

Cases:
- High-score false positive
- High-score/no-evidence
- Low-score false negatives

Inputs:
  data/annotations/resolved_annotations_v0.2.csv
  experiments/sif_conflict_review_v0.9/conflict_review.csv

Output:
  experiments/sif_pathway_conflict_audit_v1.0/
    pathway_conflict_audit.csv
    conflict_texts.csv
    summary.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "resolved_annotations_v0.2.csv"
)

CONFLICT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "sif_conflict_review_v0.9"
    / "conflict_review.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "sif_pathway_conflict_audit_v1.0"
)


HAZARD_PATTERNS = {
    "electrical_energy": [
        r"\belectric", r"\benergized\b", r"\bvoltage\b", r"\bpanel\b",
        r"\bswitchboard\b", r"\bcable\b",
    ],
    "mechanical_energy": [
        r"\bmachin", r"\bconveyor\b", r"\broller\b", r"\brotat",
        r"\bmoving part", r"\bpress\b",
    ],
    "pressure_stored_energy": [
        r"\bpressur", r"\bcompressed\b", r"\bstored energy\b",
    ],
    "chemical_release": [
        r"\bammonia\b", r"\bchlorine\b", r"\bchemical\b",
        r"\bsolvent\b", r"\bacid\b", r"\bcaustic\b", r"\btoxic\b",
        r"\bcorrosive\b", r"\bleak", r"\bspill", r"\brelease\b",
    ],
    "thermal_energy": [
        r"\bsteam\b", r"\bhot\b", r"\bheat\b", r"\bthermal\b",
    ],
    "fire_explosion_energy": [
        r"\bfire\b", r"\bexplos", r"\bflammab", r"\bignition\b",
    ],
    "vehicle_energy": [
        r"\bvehicle\b", r"\btruck\b", r"\bforklift\b", r"\bcrane\b",
        r"\bexcavat", r"\bloader\b", r"\bdriv(?:e|er|ing)\b",
    ],
    "height_fall_energy": [
        r"\bladder\b", r"\bscaffold", r"\bplatform\b", r"\bheight\b",
        r"\belevated\b", r"\bfell\b", r"\bfall\b",
    ],
    "caught_between_mechanism": [
        r"\bpinch\b", r"\bcaught\b", r"\btrapped\b", r"\bbetween\b",
    ],
    "dropped_object": [
        r"\bdropped\b", r"\bfalling object\b", r"\btool\b.*\bfall",
        r"\bmaterial\b.*\bfall",
    ],
}

EXPOSURE_PATTERNS = {
    "human_present": [
        r"\bworker\b", r"\boperator\b", r"\bemployee\b", r"\btechnician\b",
        r"\bmechanic\b", r"\bdriver\b", r"\bperson\b",
    ],
    "direct_contact": [
        r"\bcontact(?:ed|ing)?\b", r"\btouched\b",
        r"\bhand\b.*\b(?:near|on|against)\b",
    ],
    "line_of_fire": [
        r"\bline of fire\b", r"\bin the path\b", r"\btrajectory\b",
        r"\bstruck by\b", r"\bstruck\b",
    ],
    "caught_between": [
        r"\bpinch\b", r"\bcaught\b", r"\btrapped\b", r"\bbetween\b",
    ],
    "fall_exposure": [
        r"\bfell\b", r"\bfall\b", r"\bdropped from\b",
    ],
    "chemical_exposure": [
        r"\bsplash\b", r"\bexpos(?:e|ed|ure)\b",
        r"\binhal", r"\breath", r"\bfume", r"\bgas\b",
        r"\bvapou?r\b", r"\bleak", r"\bspill",
    ],
    "thermal_exposure": [
        r"\bhot\b", r"\bsteam\b", r"\bheat\b", r"\bburn",
    ],
    "vehicle_exposure": [
        r"\bvehicle\b", r"\btruck\b", r"\bforklift\b", r"\bdriver\b",
        r"\bpedestrian\b",
    ],
}

PATHWAY_PATTERNS = {
    "struck_by_or_projectile": [
        r"\bstruck by\b", r"\bstruck\b", r"\bproject(?:ed|ile)\b",
        r"\beject(?:ed|ion)\b", r"\bthrown\b",
    ],
    "caught_crush": [
        r"\bpinch\b", r"\bcaught\b", r"\btrapped\b", r"\bcrush",
        r"\bamputation\b",
    ],
    "fall_from_height": [
        r"\bladder\b", r"\bscaffold", r"\bplatform\b", r"\bheight\b",
        r"\belevated\b", r"\bfell\b",
    ],
    "electrical_contact": [
        r"\belectric", r"\benergized\b", r"\bvoltage\b",
        r"\bshock\b", r"\bcontact\b",
    ],
    "chemical_release_exposure": [
        r"\bleak", r"\bspill", r"\brelease", r"\bsplash",
        r"\binhal", r"\bfume", r"\bgas\b", r"\bvapou?r\b",
    ],
    "thermal_contact": [
        r"\bhot\b", r"\bsteam\b", r"\bheat\b", r"\bthermal\b",
    ],
    "vehicle_person_collision": [
        r"\bvehicle\b.*\bperson\b", r"\bperson\b.*\bvehicle\b",
        r"\bpedestrian\b", r"\bstruck\b.*\bvehicle\b",
    ],
}

CONTROL_PATTERNS = {
    "ppe": [r"\bgloves?\b", r"\bhelmet\b", r"\bgoggles?\b", r"\bface shield\b", r"\bppe\b"],
    "guarding": [r"\bguard\b", r"\bcover\b", r"\bbarricad", r"\bbarrier\b"],
    "isolation": [r"\bisolat", r"\blockout\b", r"\block out\b", r"\bLOTO\b", r"\bde-energ"],
    "fall_protection": [r"\bharness\b", r"\blifeline\b", r"\bfall protection\b", r"\bguardrail\b"],
    "permit_procedure": [r"\bpermit\b", r"\bJSA\b", r"\bjob safety\b", r"\brisk assessment\b", r"\bprocedure\b"],
    "distance_exclusion": [r"\bsafe distance\b", r"\bexclusion zone\b", r"\bkeep clear\b", r"\bstand clear\b"],
}

NEGATION_PATTERNS = [
    r"\bwithout\b",
    r"\bno\b",
    r"\bnot\b",
    r"\bdid not\b",
    r"\bfailed to\b",
    r"\bmissing\b",
    r"\bremoved\b",
    r"\bopen\b",
]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def hits(text: str, patterns: list[str]) -> list[str]:
    out = []
    for p in patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            out.append(m.group(0))
    return sorted(set(out))


def sentence_split(text: str) -> list[str]:
    text = norm(text)
    return [
        s.strip(" -•\t")
        for s in re.split(r"(?<=[.!?;])\s+|\n+", text)
        if s.strip()
    ]


def inspect_case(description: str) -> dict:
    text = norm(description)
    sentences = sentence_split(text)

    hazard_categories = []
    exposure_categories = []
    pathway_categories = []
    control_categories = []
    gap_signals = []

    hazard_terms = []
    exposure_terms = []
    pathway_terms = []
    control_terms = []

    for name, patterns in HAZARD_PATTERNS.items():
        terms = hits(text, patterns)
        if terms:
            hazard_categories.append(name)
            hazard_terms.extend(terms)

    for name, patterns in EXPOSURE_PATTERNS.items():
        terms = hits(text, patterns)
        if terms:
            exposure_categories.append(name)
            exposure_terms.extend(terms)

    for name, patterns in PATHWAY_PATTERNS.items():
        terms = hits(text, patterns)
        if terms:
            pathway_categories.append(name)
            pathway_terms.extend(terms)

    for name, patterns in CONTROL_PATTERNS.items():
        terms = hits(text, patterns)
        if terms:
            control_categories.append(name)
            control_terms.extend(terms)

    for p in NEGATION_PATTERNS:
        m = re.search(p, text, flags=re.I)
        if m:
            gap_signals.append(m.group(0))

    # Conservative narrative-only pathway heuristic.
    # This is not a SIF classifier; it only indicates whether the narrative
    # contains the three ingredients needed for a manual review:
    # hazard + exposure + mechanism/pathway.
    pathway_strength = 0
    if hazard_categories:
        pathway_strength += 1
    if exposure_categories:
        pathway_strength += 1
    if pathway_categories:
        pathway_strength += 1

    if pathway_strength >= 3:
        pathway_assessment = "SUPPORTED_BY_NARRATIVE_SIGNALS"
    elif pathway_strength == 2:
        pathway_assessment = "PARTIALLY_SUPPORTED"
    elif pathway_strength == 1:
        pathway_assessment = "WEAK_SIGNAL"
    else:
        pathway_assessment = "NO_CLEAR_PATHWAY_SIGNAL"

    return {
        "hazard_categories": " | ".join(sorted(set(hazard_categories))),
        "hazard_terms": " | ".join(sorted(set(hazard_terms))),
        "exposure_categories": " | ".join(sorted(set(exposure_categories))),
        "exposure_terms": " | ".join(sorted(set(exposure_terms))),
        "pathway_categories": " | ".join(sorted(set(pathway_categories))),
        "pathway_terms": " | ".join(sorted(set(pathway_terms))),
        "control_categories": " | ".join(sorted(set(control_categories))),
        "control_terms": " | ".join(sorted(set(control_terms))),
        "negation_or_gap_signals": " | ".join(sorted(set(gap_signals))),
        "hazard_signal": bool(hazard_categories),
        "exposure_signal": bool(exposure_categories),
        "pathway_signal": bool(pathway_categories),
        "pathway_strength": pathway_strength,
        "pathway_assessment": pathway_assessment,
        "sentence_count": len(sentences),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not LABEL_PATH.exists():
        raise FileNotFoundError(LABEL_PATH)
    if not CONFLICT_PATH.exists():
        raise FileNotFoundError(CONFLICT_PATH)

    labels = pd.read_csv(LABEL_PATH)
    conflicts = pd.read_csv(CONFLICT_PATH)

    labels["report_id"] = labels["report_id"].astype(str)
    labels["description"] = labels["description"].fillna("").astype(str)

    conflicts["report_id"] = conflicts["report_id"].astype(str)

    merged = conflicts.merge(
        labels[["report_id", "description"]],
        on="report_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_source"),
    )

    if merged["description"].isna().any():
        missing_ids = merged.loc[
            merged["description"].isna(), "report_id"
        ].tolist()
        raise ValueError(f"Missing narratives for IDs: {missing_ids}")

    rows = []
    for _, row in merged.iterrows():
        result = inspect_case(row["description"])
        rows.append({
            "report_id": row["report_id"],
            "review_type": row.get("review_type", ""),
            "probability": row.get("probability", None),
            "sif_label": row.get("sif_label", ""),
            "description": row["description"],
            **result,
        })

    audit = pd.DataFrame(rows)

    # Add a compact reviewer recommendation:
    # never auto-relabel; identify the exact missing ingredient to check.
    def recommendation(row: pd.Series) -> str:
        if row["pathway_assessment"] == "SUPPORTED_BY_NARRATIVE_SIGNALS":
            return "Manual adjudication: narrative contains hazard + exposure + pathway signals."
        if row["pathway_assessment"] == "PARTIALLY_SUPPORTED":
            return "Manual adjudication: check the missing hazard/exposure/pathway ingredient."
        if row["pathway_assessment"] == "WEAK_SIGNAL":
            return "Manual adjudication: weak narrative signal; inspect context carefully."
        return "Manual adjudication: no clear precursor pathway signal detected."

    audit["review_recommendation"] = audit.apply(recommendation, axis=1)

    # Aggregate diagnosis by conflict class.
    summary = {
        "cases": int(len(audit)),
        "by_review_type": (
            audit["review_type"].value_counts(dropna=False).to_dict()
        ),
        "pathway_assessment_counts": (
            audit["pathway_assessment"].value_counts(dropna=False).to_dict()
        ),
        "hazard_signal_rate": float(audit["hazard_signal"].mean()),
        "exposure_signal_rate": float(audit["exposure_signal"].mean()),
        "pathway_signal_rate": float(audit["pathway_signal"].mean()),
        "fully_supported_pathway_rate": float(
            (audit["pathway_assessment"] == "SUPPORTED_BY_NARRATIVE_SIGNALS").mean()
        ),
        "note": (
            "Narrative signals are heuristic diagnostics. They do not establish "
            "SIF status, barrier effectiveness, or correctness of existing labels."
        ),
    }

    audit.to_csv(
        OUTPUT_DIR / "pathway_conflict_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit[
        [
            "report_id",
            "review_type",
            "probability",
            "sif_label",
            "description",
        ]
    ].to_csv(
        OUTPUT_DIR / "conflict_texts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("RAKSHAK SIF PATHWAY CONFLICT AUDIT v1.0")
    print("=" * 78)
    print(f"Cases audited                    : {len(audit)}")
    print(f"Hazard signal rate              : {summary['hazard_signal_rate']:.1%}")
    print(f"Exposure signal rate            : {summary['exposure_signal_rate']:.1%}")
    print(f"Pathway signal rate             : {summary['pathway_signal_rate']:.1%}")
    print(f"Fully supported pathway rate    : {summary['fully_supported_pathway_rate']:.1%}")
    print()
    print("CASE DIAGNOSIS")
    print("-" * 78)

    show = [
        "report_id",
        "review_type",
        "probability",
        "sif_label",
        "pathway_strength",
        "pathway_assessment",
        "hazard_categories",
        "exposure_categories",
        "pathway_categories",
    ]
    print(audit[show].to_string(index=False))

    print()
    print(f"Saved outputs to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
