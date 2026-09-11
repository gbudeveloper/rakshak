"""
RAKSHAK Evidence Engine v1.3
============================

Purpose
-------
Replace the overly broad v1.2 keyword pathway rules with a stricter,
context-aware evidence layer for SIF precursor triage.

Input
-----
- data/annotations/resolved_annotations_v0.2.csv
- experiments/rakshak_final_v1.2/locked_test_predictions_with_evidence.csv
- Fixed locked split: data/processed/sif_splits_v0.2/test.csv

Design
------
Evidence is separated into:
    hazard signal
    human-exposure signal
    mechanism/pathway signal
    context-consistency signal

Important safety rule:
    A hazard keyword alone does NOT imply a SIF precursor.
    A plausible pathway alone does NOT imply SIF potential.
    The engine never invents missing severity, quantity, distance, energy,
    duration, concentration, number of people, or other counterfactual facts.

The engine is a reviewer-support component, not an autonomous safety decision.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_PATH = PROJECT_ROOT / "data" / "annotations" / "resolved_annotations_v0.2.csv"
TEST_SPLIT = PROJECT_ROOT / "data" / "processed" / "sif_splits_v0.2" / "test.csv"
PREDICTION_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "rakshak_final_v1.2"
    / "locked_test_predictions_with_evidence.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "rakshak_evidence_v1.3"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def first_match(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.I)
        if m:
            return m.group(0)
    return None


def has(text: str, patterns: list[str]) -> bool:
    return first_match(text, patterns) is not None


# ---------------------------------------------------------------------------
# Context-aware lexicons
# ---------------------------------------------------------------------------

HAZARDS = {
    "electrical": [
        r"\belectric(?:al|ity)?\b", r"\benergized\b", r"\bde-energized\b",
        r"\bvoltage\b", r"\bkv\b", r"\bpower\b", r"\bshort circuit\b",
        r"\belectric shock\b",
    ],
    "chemical": [
        r"\bchemical\b", r"\bsolvent\b", r"\bacid\b", r"\bcaustic\b",
        r"\btoxic\b", r"\bcorrosive\b", r"\bthinner\b", r"\bpaint\b",
        r"\bchlorine\b", r"\bammonia\b", r"\bgas\b", r"\bfume\b",
        r"\bvapou?r\b",
    ],
    "thermal": [
        r"\bsteam\b", r"\bhot\b", r"\bheat\b", r"\bthermal\b",
        r"\bburn(?:ed|ing)?\b", r"\bscald",
    ],
    "pressure": [
        r"\bpressur(?:e|ized|ised)\b", r"\bcompressed\b",
        r"\bstored energy\b", r"\bpressure vessel\b",
    ],
    "fire_explosion": [
        r"\bfire\b", r"\bexplos(?:ion|ive|ed)\b", r"\bflammab",
        r"\bignition\b", r"\bcombust",
    ],
    "vehicle": [
        r"\bvehicle\b", r"\btruck\b", r"\bforklift\b", r"\bcrane\b",
        r"\bexcavat(?:or|ing)\b", r"\bloader\b", r"\blocomotive\b",
        r"\bdriver\b", r"\bdriving\b",
    ],
    "fall_height": [
        r"\bheight\b", r"\belevated\b", r"\bscaffold", r"\bladder\b",
        r"\bplatform\b", r"\btower\b", r"\bfell\b", r"\bfall(?:ing)?\b",
        r"\bdropped from\b",
    ],
    "mechanical": [
        r"\bmachin", r"\bconveyor\b", r"\broller\b", r"\brotat",
        r"\bmoving part\b", r"\bpress\b", r"\bmixer\b",
    ],
    "stored_projectile": [
        r"\bproject(?:ile|ed)\b", r"\beject(?:ed|ion)\b",
        r"\bthrown\b", r"\bdropped\b", r"\bfalling object\b",
    ],
}

HUMAN_EXPOSURE = {
    "chemical_exposure": [
        r"\binhal", r"\bfume", r"\bvapou?r", r"\bchemical exposure\b",
        r"\bsplash(?:ed)? .*?(?:skin|eye|face|shoulder|lip|body)\b",
        r"\bcontact(?:ed)? .*?(?:chemical|solvent|acid|thinner|paint)\b",
    ],
    "thermal_exposure": [
        r"\bburn", r"\bscald", r"\bhot\b.*\b(?:contact|skin|body)\b",
        r"\bsteam\b.*\b(?:contact|injur|burn)\b",
        r"\b(?:water|liquid)\b.*\bhot\b",
    ],
    "electrical_exposure": [
        r"\belectric shock\b", r"\b(?:received|suffered|got)\b.*\belectric\b",
        r"\benergized\b.*\b(?:contact|touched|shock)\b",
        r"\bcontact\b.*\benergized\b",
    ],
    "vehicle_exposure": [
        r"\bvehicle\b.*\b(?:hit|struck|collision|collid|ran over|run over)\b",
        r"\b(?:hit|struck|collision|collid)\b.*\bvehicle\b",
        r"\bvehicle exposure\b", r"\bpedestrian\b",
    ],
    "fall_exposure": [
        r"\bfell\b", r"\bfall(?:ing)?\b.*\b(?:from|off)\b",
        r"\b(?:from|off)\b.*\b(?:height|platform|scaffold|tower|ladder)\b",
    ],
    "line_of_fire": [
        r"\bline of fire\b", r"\bstruck by\b", r"\bin the path\b",
        r"\btrajectory\b",
    ],
    "caught_between_exposure": [
        r"\bpinch\b", r"\bcaught\b", r"\btrapped\b", r"\bcrush",
    ],
    "generic_human_present": [
        r"\boperator\b", r"\bemployee\b", r"\bcollaborator\b", r"\bworker\b",
        r"\btechnician\b", r"\bsupervisor\b", r"\bforeman\b",
        r"\boccupant(?:s)?\b", r"\bperson(?:nel)?\b",
    ],
}

# A pathway must have a mechanism-specific context, not merely a generic
# word such as "contact" or "splash".
PATHWAYS = {
    "electrical_contact": {
        "requires": [
            r"\belectric shock\b",
            r"\benergized\b.*\b(?:contact|touch)\b",
            r"\bcontact\b.*\benergized\b",
            r"\bvoltage\b.*\bcontact\b",
        ],
        "compatible_hazards": {"electrical"},
    },
    "chemical_release_exposure": {
        "requires": [
            r"\b(?:chemical|solvent|acid|caustic|thinner|paint|toxic|corrosive)\b"
            r".{0,100}\b(?:spill|leak|release|splash|inhal|fume|vapou?r)\b",
            r"\b(?:spill|leak|release|splash|inhal|fume|vapou?r)\b"
            r".{0,100}\b(?:chemical|solvent|acid|caustic|thinner|paint|toxic|corrosive)\b",
            r"\bchemical exposure\b",
            r"\binhal(?:ed|ation)?\b",
        ],
        "compatible_hazards": {"chemical"},
    },
    "thermal_contact": {
        "requires": [
            r"\bsteam\b.*\b(?:contact|burn|scald|injur)\b",
            r"\bhot\b.*\b(?:contact|burn|scald|injur)\b",
            r"\b(?:burn|scald)\w*\b.*\b(?:hot|steam|heat|thermal)\b",
        ],
        "compatible_hazards": {"thermal"},
    },
    "fall_from_height": {
        "requires": [
            r"\bfell\b.*\b(?:height|platform|scaffold|tower|ladder)\b",
            r"\bfall(?:ing)?\b.*\b(?:height|platform|scaffold|tower|ladder)\b",
            r"\b(?:height|platform|scaffold|tower|ladder)\b.*\bfall\b",
        ],
        "compatible_hazards": {"fall_height"},
    },
    "struck_by_projectile": {
        "requires": [
            r"\bstruck by\b", r"\bstruck\b.*\b(?:object|material|projectile)\b",
            r"\bproject(?:ile|ed)\b.*\b(?:hit|strike|person|worker)\b",
            r"\beject(?:ed|ion)\b.*\b(?:hit|strike|person|worker)\b",
        ],
        "compatible_hazards": {"stored_projectile", "mechanical", "pressure"},
    },
    "caught_crush": {
        "requires": [
            r"\bpinch\b.*\b(?:hand|finger|person|worker)\b",
            r"\bcaught\b.*\b(?:between|in|inside)\b",
            r"\btrapped\b.*\b(?:between|in|inside)\b",
            r"\bcrush(?:ed|ing)?\b",
        ],
        "compatible_hazards": {"mechanical"},
    },
    "vehicle_person_collision": {
        "requires": [
            r"\b(?:vehicle|truck|forklift|crane|loader|excavator)\b"
            r".{0,100}\b(?:hit|struck|collision|collid|run over|ran over)\b",
            r"\b(?:hit|struck|collision|collid|run over|ran over)\b"
            r".{0,100}\b(?:vehicle|truck|forklift|crane|loader|excavator)\b",
            r"\bvehicle\b.*\bpedestrian\b",
        ],
        "compatible_hazards": {"vehicle"},
    },
}

SAFE_CONTEXTS = {
    # These phrases explicitly weaken a generic hazard signal and help avoid
    # overcalling a pathway merely because a hazardous material/equipment term
    # is present.
    "water_mud_only": [
        r"\bwater\b.*\bmud\b", r"\bmud\b.*\bwater\b",
    ],
    "no_person_affected": [
        r"\bno presence of personnel\b",
        r"\bno personnel\b.*\b(?:affected|injur)\b",
        r"\bno one\b.*\b(?:affected|injur)\b",
    ],
    "deenergized_disabled": [
        r"\bde-energized\b", r"\bdeenergized\b", r"\bdisabled\b",
    ],
}


def matched_categories(text: str, lexicon: dict[str, list[str]]) -> dict[str, str]:
    out = {}
    for category, patterns in lexicon.items():
        match = first_match(text, patterns)
        if match:
            out[category] = match
    return out


def pathway_matches(text: str, hazard_categories: set[str]) -> dict[str, str]:
    matches = {}

    for pathway, spec in PATHWAYS.items():
        match = first_match(text, spec["requires"])
        if not match:
            continue

        # Require either a compatible hazard in the narrative OR a very
        # specific pathway phrase that itself establishes the hazard context.
        if hazard_categories.intersection(spec["compatible_hazards"]):
            matches[pathway] = match
            continue

        if pathway == "vehicle_person_collision" and "vehicle" in hazard_categories:
            matches[pathway] = match
        elif pathway == "electrical_contact" and "electrical" in hazard_categories:
            matches[pathway] = match
        elif pathway == "chemical_release_exposure" and "chemical" in hazard_categories:
            matches[pathway] = match
        elif pathway == "thermal_contact" and "thermal" in hazard_categories:
            matches[pathway] = match
        elif pathway == "fall_from_height" and "fall_height" in hazard_categories:
            matches[pathway] = match

    # Strong suppression: water/mud splash is not treated as chemical release
    # unless actual chemical terminology is present.
    if "water_mud_only" in safe_flags(text):
        if "chemical" not in hazard_categories:
            matches.pop("chemical_release_exposure", None)

    return matches


def safe_flags(text: str) -> set[str]:
    return {
        key
        for key, patterns in SAFE_CONTEXTS.items()
        if has(text, patterns)
    }


def assess(text: str) -> dict:
    hazards = matched_categories(text, HAZARDS)
    exposures = matched_categories(text, HUMAN_EXPOSURE)
    paths = pathway_matches(text, set(hazards))
    flags = safe_flags(text)

    # Generic human-present is not itself an exposure. It is context only.
    exposures_real = {
        k: v for k, v in exposures.items()
        if k != "generic_human_present"
    }

    hazard_signal = bool(hazards)
    exposure_signal = bool(exposures_real)

    # A pathway is only called explicit when mechanism-specific language exists.
    pathway_signal = bool(paths)

    if hazard_signal and exposure_signal and pathway_signal:
        assessment = "SUPPORTED"
    elif (hazard_signal and exposure_signal) or (hazard_signal and pathway_signal) or (exposure_signal and pathway_signal):
        assessment = "PARTIAL"
    elif hazard_signal or exposure_signal or pathway_signal:
        assessment = "WEAK"
    else:
        assessment = "NONE"

    # Strong contextual downgrade where the report explicitly says no personnel
    # could have been affected.
    if "no_person_affected" in flags:
        assessment = "WEAK" if assessment in {"SUPPORTED", "PARTIAL"} else assessment

    # A generic chemical hazard without actual exposure is not a supported
    # chemical-release pathway.
    if "chemical" in hazards and not (
        "chemical_exposure" in exposures_real
        or "chemical_release_exposure" in paths
    ):
        paths.pop("chemical_release_exposure", None)

    complete = bool(hazards) and bool(exposures_real) and bool(paths)

    return {
        "hazards": " | ".join(sorted(hazards)),
        "hazard_evidence": " | ".join(f"{k}: {v}" for k, v in sorted(hazards.items())),
        "exposures": " | ".join(sorted(exposures_real)),
        "exposure_evidence": " | ".join(
            f"{k}: {v}" for k, v in sorted(exposures_real.items())
        ),
        "pathways": " | ".join(sorted(paths)),
        "pathway_evidence": " | ".join(f"{k}: {v}" for k, v in sorted(paths.items())),
        "safe_context_flags": " | ".join(sorted(flags)),
        "hazard_signal": int(bool(hazards)),
        "exposure_signal": int(bool(exposures_real)),
        "pathway_signal": int(bool(paths)),
        "pathway_signal_strength": (
            int(bool(hazards)) + int(bool(exposures_real)) + int(bool(paths))
        ),
        "complete_pathway": int(complete),
        "pathway_assessment": assessment,
    }


def reviewer_priority(probability: float, evidence: dict) -> str:
    complete = bool(evidence["complete_pathway"])
    strength = int(evidence["pathway_signal_strength"])

    if probability >= 0.80 and complete:
        return "P1"
    if probability >= 0.80 and strength >= 1:
        return "P2"
    if probability >= 0.50 and complete:
        return "P3"
    if probability >= 0.50 and strength >= 1:
        return "P3"
    if probability >= 0.25:
        return "P4"
    return "P5"


def reviewer_interpretation(probability: float, evidence: dict, true_label: int | None) -> str:
    strength = int(evidence["pathway_signal_strength"])
    assessment = evidence["pathway_assessment"]

    if true_label is not None and true_label == 0 and probability >= 0.75:
        return (
            "High model score but adjudicated NO; mandatory human review. "
            "Do not infer SIF potential from hazard/pathway signals alone."
        )

    if true_label is not None and true_label == 1 and probability < 0.50 and strength >= 2:
        return (
            "Model score is below the reference threshold despite meaningful "
            "hazard/exposure evidence; review for possible missed precursor."
        )

    if assessment == "SUPPORTED":
        return (
            "Narrative contains explicit hazard, human-exposure, and "
            "mechanism-specific pathway evidence. Human HSSE review remains required."
        )
    if assessment == "PARTIAL":
        return (
            "Narrative contains partial pathway evidence. Review missing "
            "mechanism/severity context before disposition."
        )
    if assessment == "WEAK":
        return (
            "Only limited safety-context evidence is present. Avoid inferring "
            "a complete SIF pathway."
        )
    return (
        "No explicit pathway evidence detected by the rules. Model score should "
        "be treated as a ranking signal requiring reviewer context."
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(PREDICTION_PATH)
    labels = pd.read_csv(LABEL_PATH)
    test_ids = pd.read_csv(TEST_SPLIT)

    pred["report_id"] = pred["report_id"].astype(str)
    labels["report_id"] = labels["report_id"].astype(str)
    test_ids["report_id"] = test_ids["report_id"].astype(str)

    merged = (
        test_ids[["report_id"]]
        .merge(labels[["report_id", "description", "final_sif_potential"]], on="report_id", how="left")
        .merge(
            pred[["report_id", "sif_precursor_probability"]],
            on="report_id",
            how="left",
        )
    )

    if merged["description"].isna().any():
        raise ValueError("Missing description for one or more test records.")
    if merged["sif_precursor_probability"].isna().any():
        raise ValueError("Missing model probability for one or more test records.")

    rows = []

    for _, row in merged.iterrows():
        text = norm(row["description"])
        e = assess(text)
        true_label = (
            1 if str(row["final_sif_potential"]).upper() == "YES"
            else 0 if str(row["final_sif_potential"]).upper() == "NO"
            else None
        )
        probability = float(row["sif_precursor_probability"])

        row_out = {
            "report_id": row["report_id"],
            "true_label": true_label,
            "true_sif_potential": row["final_sif_potential"],
            "sif_precursor_probability": probability,
            "prediction_at_0.50": int(probability >= 0.50),
            **e,
            "reviewer_priority": reviewer_priority(probability, e),
            "reviewer_interpretation": reviewer_interpretation(
                probability, e, true_label
            ),
        }
        rows.append(row_out)

    out = pd.DataFrame(rows).sort_values(
        ["reviewer_priority", "sif_precursor_probability"],
        ascending=[True, False],
    )

    out.to_csv(
        OUTPUT_DIR / "locked_test_reviewer_report_v1.3.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Explicit error buckets.
    out["error_type"] = "CORRECT"
    out.loc[
        (out["true_label"] == 0) & (out["prediction_at_0.50"] == 1),
        "error_type",
    ] = "FALSE_POSITIVE"
    out.loc[
        (out["true_label"] == 1) & (out["prediction_at_0.50"] == 0),
        "error_type",
    ] = "FALSE_NEGATIVE"

    conflict = out[out["error_type"] != "CORRECT"].copy()
    conflict.to_csv(
        OUTPUT_DIR / "locked_test_conflicts_v1.3.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "records": int(len(out)),
        "supported_pathway_rate": float(out["complete_pathway"].mean()),
        "hazard_signal_rate": float(out["hazard_signal"].mean()),
        "exposure_signal_rate": float(out["exposure_signal"].mean()),
        "mechanism_pathway_signal_rate": float(out["pathway_signal"].mean()),
        "priority_counts": out["reviewer_priority"].value_counts().sort_index().to_dict(),
        "assessment_counts": out["pathway_assessment"].value_counts().to_dict(),
        "false_positive_count": int((out["error_type"] == "FALSE_POSITIVE").sum()),
        "false_negative_count": int((out["error_type"] == "FALSE_NEGATIVE").sum()),
        "high_score_no_count": int(
            ((out["true_label"] == 0) & (out["sif_precursor_probability"] >= 0.75)).sum()
        ),
        "low_score_yes_with_supported_path_count": int(
            (
                (out["true_label"] == 1)
                & (out["sif_precursor_probability"] < 0.50)
                & (out["complete_pathway"] == 1)
            ).sum()
        ),
    }

    (OUTPUT_DIR / "summary_v1.3.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("RAKSHAK EVIDENCE ENGINE v1.3")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print("\nCONFLICT CASES")
    print(
        conflict[
            [
                "report_id",
                "true_sif_potential",
                "sif_precursor_probability",
                "error_type",
                "hazards",
                "exposures",
                "pathways",
                "pathway_assessment",
                "reviewer_priority",
            ]
        ].to_string(index=False)
    )
    print("\nSaved:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
