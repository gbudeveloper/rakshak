"""
RAKSHAK Evidence Engine v1.5
============================

Final targeted evidence/conflict layer.

What v1.5 adds over v1.4
------------------------
1. Distinguishes "hazard/exposure occurred" from "SIF sufficiency".
2. Extracts localized/minor consequence language.
3. Extracts contextual modifiers:
      - de-energized / disabled
      - protective equipment
      - distance
      - containment / no-personnel-affected language
4. Adds a "sif_context_status" field:
      SUFFICIENT_CONTEXT
      CONTEXT_LIMITED
      CONTEXT_AGAINST_SIF
      UNKNOWN
5. Detects model/evidence conflicts explicitly.
6. Does not change the classifier and does not use adjudication labels
   as model features.

This is reviewer-support logic, not an automatic SIF decision engine.

Inputs
------
experiments/rakshak_final_v1.2/locked_test_predictions_with_evidence.csv
data/annotations/resolved_annotations_v0.2.csv
data/processed/sif_splits_v0.2/test.csv

Outputs
-------
experiments/rakshak_evidence_v1.5/
    locked_test_reviewer_report_v1.5.csv
    locked_test_conflicts_v1.5.csv
    summary_v1.5.json
    evidence_rules_v1.5.json
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

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "rakshak_evidence_v1.5"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def hit(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.I)
        if m:
            return m.group(0)
    return None


def all_hits(text: str, patterns: list[str]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.I)
        if m:
            out.append(m.group(0))
    return out


# ---------------------------------------------------------------------------
# Hazard / exposure / mechanism dictionaries
# ---------------------------------------------------------------------------

HAZARDS = {
    "electrical": [
        r"\belectric(?:al|ity)?\b",
        r"\benergized\b",
        r"\bde[- ]?energized\b",
        r"\bvoltage\b",
        r"\bkv\b",
        r"\bpower\b",
        r"\belectric shock\b",
    ],
    "chemical": [
        r"\bchemical\b",
        r"\bsolvent\b",
        r"\bacid\b",
        r"\bcaustic\b",
        r"\btoxic\b",
        r"\bcorrosive\b",
        r"\bthinner\b",
        r"\bpaint\b",
        r"\bchlorine\b",
        r"\bammonia\b",
        r"\bfume\b",
        r"\bvapou?r\b",
        r"\bgas\b",
        r"\bmixture\b",
    ],
    "thermal": [
        r"\bsteam\b",
        r"\bhot\b",
        r"\bheat\b",
        r"\bthermal\b",
        r"\bburn(?:ed|ing)?\b",
        r"\bscald(?:ed|ing)?\b",
    ],
    "pressure": [
        r"\bpressur(?:e|ized|ised)\b",
        r"\bcompressed\b",
        r"\bstored energy\b",
        r"\bhydraulic\b",
    ],
    "fire_explosion": [
        r"\bfire\b",
        r"\bexplos(?:ion|ive|ed)\b",
        r"\bflammab",
        r"\bignition\b",
        r"\bcombust",
    ],
    "vehicle": [
        r"\bvehicle\b",
        r"\btruck\b",
        r"\bforklift\b",
        r"\bcrane\b",
        r"\bexcavat(?:or|ing)\b",
        r"\bloader\b",
        r"\blocomotive\b",
        r"\bdriver\b",
        r"\bdriving\b",
        r"\boccupants?\b",
        r"\bpedestrian\b",
    ],
    "fall_height": [
        r"\bheight\b",
        r"\belevated\b",
        r"\bscaffold",
        r"\bladder\b",
        r"\bplatform\b",
        r"\btower\b",
        r"\bfell\b",
        r"\bfall(?:ing)?\b",
    ],
    "mechanical": [
        r"\bmachin",
        r"\bconveyor\b",
        r"\broller\b",
        r"\brotat",
        r"\bmoving part\b",
        r"\bpress\b",
        r"\bmixer\b",
    ],
    "stored_projectile": [
        r"\bproject(?:ile|ed)\b",
        r"\beject(?:ed|ion)\b",
        r"\bthrown\b",
        r"\bdropped\b",
        r"\bfalling object\b",
        r"\bsplash(?:ed)?\b",
        r"\bflow\b",
    ],
}

EXPOSURES = {
    "chemical_exposure": [
        r"\binhal",
        r"\bfume",
        r"\bvapou?r",
        r"\bchemical exposure\b",
        r"\bchemical\b.{0,100}\b(?:skin|eye|face|mouth|lip|shoulder|body)\b",
        r"\b(?:skin|eye|face|mouth|lip|shoulder|body)\b.{0,100}\b(?:chemical|solvent|acid|thinner|paint)\b",
        r"\b(?:splash|projected)\b.{0,80}\b(?:chemical|thinner|solvent|acid|paint)\b",
        r"\b(?:chemical|thinner|solvent|acid|paint)\b.{0,80}\b(?:splash|projected)\b",
    ],
    "thermal_exposure": [
        r"\bburn(?:ed|ing)?\b",
        r"\bscald(?:ed|ing)?\b",
        r"\bhot\b.{0,100}\b(?:contact|skin|body|burn)\b",
        r"\bsteam\b.{0,100}\b(?:contact|skin|body|burn)\b",
        r"\bredness\b.{0,80}\bburning\b",
    ],
    "electrical_exposure": [
        r"\belectric shock\b",
        r"\b(?:received|suffered|got)\b.{0,60}\belectric(?:al)?\b",
        r"\benergized\b.{0,80}\b(?:contact|touched|shock)\b",
        r"\bcontact\b.{0,80}\benergized\b",
    ],
    "vehicle_exposure": [
        r"\bvehicle exposure\b",
        r"\bpedestrian\b",
        r"\b(?:hit|struck|collision|collid|ran over|run over)\b.{0,100}\b(?:vehicle|truck|forklift|crane|loader|excavator)\b",
        r"\b(?:vehicle|truck|forklift|crane|loader|excavator)\b.{0,100}\b(?:hit|struck|collision|collid|ran over|run over)\b",
        r"\b(?:vehicle|truck|forklift|crane|loader|excavator)\b.{0,80}\bperson(?:nel)?\b",
    ],
    "fall_exposure": [
        r"\bfell\b",
        r"\bfall(?:ing)?\b.{0,80}\b(?:from|off)\b",
        r"\b(?:from|off)\b.{0,80}\b(?:height|platform|scaffold|tower|ladder)\b",
    ],
    "line_of_fire": [
        r"\bline of fire\b",
        r"\bstruck by\b",
        r"\bin the path\b",
        r"\btrajectory\b",
        r"\bpath of\b.{0,70}\b(?:flow|object|vehicle|material)\b",
    ],
    "caught_between_exposure": [
        r"\bpinch\b",
        r"\bcaught\b.{0,70}\b(?:between|in|inside)\b",
        r"\btrapped\b.{0,70}\b(?:between|in|inside)\b",
        r"\bcrush(?:ed|ing)?\b",
    ],
    "direct_contact": [
        r"\bcontact(?:ed)?\b",
        r"\btouched\b",
        r"\bprojected onto\b",
        r"\bsplashed\b",
    ],
}

PATHWAYS = {
    "electrical_contact": {
        "requires_any": EXPOSURES["electrical_exposure"],
        "compatible_hazards": {"electrical"},
    },
    "chemical_release_exposure": {
        "requires_any": EXPOSURES["chemical_exposure"],
        "compatible_hazards": {"chemical"},
    },
    "thermal_contact": {
        "requires_any": EXPOSURES["thermal_exposure"],
        "compatible_hazards": {"thermal"},
    },
    "fall_from_height": {
        "requires_any": EXPOSURES["fall_exposure"],
        "compatible_hazards": {"fall_height"},
    },
    "struck_by_projectile": {
        "requires_any": [
            r"\bstruck by\b",
            r"\bproject(?:ile|ed)\b.{0,100}\b(?:hit|strike|person|worker)\b",
            r"\beject(?:ed|ion)\b.{0,100}\b(?:hit|strike|person|worker)\b",
            r"\bfalling object\b.{0,100}\b(?:hit|strike|person|worker)\b",
        ],
        "compatible_hazards": {"stored_projectile", "mechanical", "pressure"},
    },
    "caught_crush": {
        "requires_any": EXPOSURES["caught_between_exposure"],
        "compatible_hazards": {"mechanical"},
    },
    "vehicle_person_collision": {
        "requires_any": EXPOSURES["vehicle_exposure"],
        "compatible_hazards": {"vehicle"},
    },
}


# ---------------------------------------------------------------------------
# Context / severity signals
# ---------------------------------------------------------------------------

MINOR_CONTEXT = {
    "localized_reaction": [
        r"\blocalized\b",
        r"\blocalised\b",
        r"\bmild\b",
        r"\bminor\b",
        r"\bslight\b",
    ],
    "redness_swelling": [
        r"\bredness\b",
        r"\bswelling\b",
        r"\birritation\b",
    ],
    "superficial_injury": [
        r"\bsuperficial\b",
        r"\bsmall cut\b",
        r"\bminor cut\b",
        r"\bminor laceration\b",
        r"\bscratch\b",
    ],
    "no_time_loss": [
        r"\bcould continue the activity\b",
        r"\bcontinued the activity\b",
        r"\breturned to work\b",
        r"\bno lost time\b",
        r"\bwithout lost time\b",
    ],
}

SEVERE_CONTEXT = {
    "fatality": [
        r"\bfatal(?:ity)?\b",
        r"\bdied\b",
        r"\bdeath\b",
        r"\bkilled\b",
    ],
    "loss_of_consciousness": [
        r"\b(unconscious|loss of consciousness)\b",
    ],
    "hospitalization": [
        r"\bhospital(?:ized|ised|ization|isation)\b",
        r"\badmitted to hospital\b",
    ],
    "fracture_amputation": [
        r"\bfracture\b",
        r"\bamputation\b",
        r"\bamputated\b",
    ],
    "severe_burn": [
        r"\bsevere burn\b",
        r"\bsecond degree burn\b",
        r"\bthird degree burn\b",
        r"\bextensive burn\b",
    ],
}

CONTEXT = {
    "water_mud": [
        r"\bwater\b",
        r"\bmud\b",
    ],
    "no_person_affected": [
        r"\bno presence of personnel\b",
        r"\bno personnel\b.{0,80}\b(?:affected|injur)\b",
        r"\bno one\b.{0,80}\b(?:affected|injur)\b",
    ],
    "deenergized_disabled": [
        r"\bde[- ]?energized\b",
        r"\bdisabled\b",
    ],
    "protective_equipment": [
        r"\b(?:ppe|ep[pi]s?)\b",
        r"\bsafety belt\b",
        r"\bseat belt\b",
        r"\bsafety harness\b",
    ],
    "distance_context": [
        r"\b\d+\s*(?:m|meter|meters|metres)\b",
        r"\bdistance\b",
    ],
    "containment": [
        r"\bcontained\b",
        r"\bcontainment\b",
        r"\bclosed system\b",
        r"\binside the container\b",
    ],
}


def category_matches(text: str, lexicon: dict[str, list[str]]) -> dict[str, str]:
    result = {}
    for category, patterns in lexicon.items():
        m = hit(text, patterns)
        if m:
            result[category] = m
    return result


def context_matches(text: str) -> set[str]:
    return {
        category
        for category, patterns in CONTEXT.items()
        if hit(text, patterns)
    }


def minor_matches(text: str) -> dict[str, str]:
    result = {}
    for category, patterns in MINOR_CONTEXT.items():
        m = hit(text, patterns)
        if m:
            result[category] = m
    return result


def severe_matches(text: str) -> dict[str, str]:
    result = {}
    for category, patterns in SEVERE_CONTEXT.items():
        m = hit(text, patterns)
        if m:
            result[category] = m
    return result


def contextual_pathways(text: str, hazards: set[str]) -> dict[str, str]:
    paths = {}

    for name, spec in PATHWAYS.items():
        m = hit(text, spec["requires_any"])
        if not m:
            continue
        if hazards.intersection(spec["compatible_hazards"]):
            paths[name] = m

    # Never infer chemical release from generic water/mud splash/projected.
    flags = context_matches(text)
    if (
        "water_mud" in flags
        and "chemical" not in hazards
        and "chemical_exposure" not in category_matches(text, EXPOSURES)
    ):
        paths.pop("chemical_release_exposure", None)

    return paths


def derive_sif_context_status(
    hazards: dict[str, str],
    exposures: dict[str, str],
    pathways: dict[str, str],
    contexts: set[str],
    minor: dict[str, str],
    severe: dict[str, str],
) -> tuple[str, str]:

    # Explicit contextual facts that strongly counter a credible SIF pathway.
    if "no_person_affected" in contexts:
        return (
            "CONTEXT_AGAINST_SIF",
            "Narrative explicitly states that personnel were not affected or at risk.",
        )

    # Strong severe evidence makes the context materially supportive.
    if severe and pathways:
        return (
            "SUFFICIENT_CONTEXT",
            "Mechanism evidence is present together with explicit severe-consequence context.",
        )

    # A complete mechanism plus only minor/localized outcome wording is NOT
    # sufficient evidence for SIF. It is still a real exposure event.
    if pathways and minor and not severe:
        return (
            "CONTEXT_LIMITED",
            "Exposure/mechanism is explicit, but available consequence context is localized/minor; SIF sufficiency is not established.",
        )

    # Explicit pathway with human exposure and no counter-context.
    if pathways and exposures:
        return (
            "SUFFICIENT_CONTEXT",
            "Explicit human exposure and mechanism evidence are present; severity/energy context still requires HSSE review.",
        )

    if hazards and exposures:
        return (
            "CONTEXT_LIMITED",
            "Hazard and exposure are present, but a mechanism-specific pathway is not explicit.",
        )

    if hazards:
        return (
            "UNKNOWN",
            "Hazard is present but human exposure/pathway context is incomplete.",
        )

    if exposures:
        return (
            "UNKNOWN",
            "Exposure signal is present but hazard/pathway context is incomplete.",
        )

    return (
        "UNKNOWN",
        "No sufficient narrative safety context detected by the evidence rules.",
    )


def conflict_type(
    true_label: int | None,
    probability: float,
    e: dict,
) -> str:

    predicted = int(probability >= 0.50)

    if true_label == 0 and predicted == 1:
        if probability >= 0.75:
            return "HIGH_SCORE_FALSE_POSITIVE"
        return "FALSE_POSITIVE"

    if true_label == 1 and predicted == 0:
        if e["pathway_signal"] and e["exposure_signal"]:
            return "FALSE_NEGATIVE_WITH_EVIDENCE"
        return "FALSE_NEGATIVE"

    # Not a binary decision error, but still operationally important.
    if true_label == 0 and probability >= 0.75:
        return "HIGH_SCORE_NO"

    if (
        true_label == 1
        and probability < 0.50
        and e["complete_pathway"]
    ):
        return "LOW_SCORE_YES_WITH_EXPLICIT_PATHWAY"

    return "NONE"


def reviewer_priority(probability: float, e: dict, conflict: str) -> str:

    if conflict in {
        "HIGH_SCORE_FALSE_POSITIVE",
        "HIGH_SCORE_NO",
    }:
        return "P1"

    if conflict in {
        "LOW_SCORE_YES_WITH_EXPLICIT_PATHWAY",
        "FALSE_NEGATIVE_WITH_EVIDENCE",
    }:
        return "P1"

    if probability >= 0.80:
        return "P2"

    if probability >= 0.50:
        return "P3"

    if e["complete_pathway"]:
        return "P3"

    if e["pathway_signal_strength"] >= 1:
        return "P4"

    return "P5"


def interpretation(
    probability: float,
    e: dict,
    true_label: int | None,
    conflict: str,
) -> str:

    if conflict == "HIGH_SCORE_FALSE_POSITIVE":
        return (
            "High model score conflicts with adjudicated NO. Review the narrative "
            "and contextual evidence; do not allow hazard/pathway signals to override adjudication."
        )

    if conflict == "LOW_SCORE_YES_WITH_EXPLICIT_PATHWAY":
        return (
            "Model score is below 0.50 despite explicit mechanism evidence. "
            "Treat as a possible model miss and review manually."
        )

    if conflict == "FALSE_NEGATIVE_WITH_EVIDENCE":
        return (
            "Model prediction is NO at 0.50, but the narrative contains exposure "
            "and mechanism evidence. Review as a possible missed precursor."
        )

    if e["sif_context_status"] == "CONTEXT_AGAINST_SIF":
        return (
            "Narrative explicitly limits personnel exposure. Do not infer SIF potential "
            "from the hazard alone."
        )

    if e["sif_context_status"] == "CONTEXT_LIMITED":
        return (
            "Exposure/mechanism evidence exists, but available consequence context is "
            "limited. Human HSSE review is required before SIF interpretation."
        )

    if e["sif_context_status"] == "SUFFICIENT_CONTEXT":
        return (
            "Narrative contains material hazard/exposure/mechanism evidence. "
            "This is a strong reviewer-priority signal, not an automatic SIF decision."
        )

    if probability >= 0.50:
        return (
            "Model ranks the incident above the reference threshold, but narrative "
            "evidence is incomplete. Review before disposition."
        )

    return (
        "Low model score and limited explicit pathway evidence. Retain as lower-priority "
        "unless other HSSE context indicates otherwise."
    )


def analyze(text: str) -> dict:
    hazards = category_matches(text, HAZARDS)
    exposures = category_matches(text, EXPOSURES)
    contexts = context_matches(text)
    minor = minor_matches(text)
    severe = severe_matches(text)
    pathways = contextual_pathways(text, set(hazards))

    # generic direct contact is not an exposure mechanism by itself.
    real_exposures = {
        k: v for k, v in exposures.items()
        if k != "direct_contact"
    }

    hazard_signal = bool(hazards)
    exposure_signal = bool(real_exposures)
    pathway_signal = bool(pathways)

    strength = (
        int(hazard_signal)
        + int(exposure_signal)
        + int(pathway_signal)
    )

    context_status, context_reason = derive_sif_context_status(
        hazards,
        real_exposures,
        pathways,
        contexts,
        minor,
        severe,
    )

    return {
        "hazards": " | ".join(sorted(hazards)),
        "hazard_evidence": " | ".join(
            f"{k}: {v}" for k, v in sorted(hazards.items())
        ),
        "exposures": " | ".join(sorted(real_exposures)),
        "exposure_evidence": " | ".join(
            f"{k}: {v}" for k, v in sorted(real_exposures.items())
        ),
        "pathways": " | ".join(sorted(pathways)),
        "pathway_evidence": " | ".join(
            f"{k}: {v}" for k, v in sorted(pathways.items())
        ),
        "minor_context": " | ".join(
            f"{k}: {v}" for k, v in sorted(minor.items())
        ),
        "severe_context": " | ".join(
            f"{k}: {v}" for k, v in sorted(severe.items())
        ),
        "context_flags": " | ".join(sorted(contexts)),
        "hazard_signal": int(hazard_signal),
        "exposure_signal": int(exposure_signal),
        "pathway_signal": int(pathway_signal),
        "pathway_signal_strength": strength,
        "complete_pathway": int(bool(pathways) and strength == 3),
        "sif_context_status": context_status,
        "sif_context_reason": context_reason,
        "pathway_assessment": (
            "SUPPORTED" if strength == 3 and context_status != "CONTEXT_AGAINST_SIF"
            else "PARTIAL" if strength == 2
            else "WEAK" if strength == 1
            else "NONE"
        ),
    }


def main() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(PREDICTION_PATH)
    labels = pd.read_csv(LABEL_PATH)
    test = pd.read_csv(TEST_SPLIT)

    for df in (predictions, labels, test):
        df["report_id"] = df["report_id"].astype(str)

    merged = (
        test[["report_id"]]
        .merge(
            labels[
                ["report_id", "description", "final_sif_potential"]
            ],
            on="report_id",
            how="left",
        )
        .merge(
            predictions[
                ["report_id", "sif_precursor_probability"]
            ],
            on="report_id",
            how="left",
        )
    )

    if merged["description"].isna().any():
        raise ValueError("Missing description in locked test set.")

    if merged["sif_precursor_probability"].isna().any():
        raise ValueError("Missing model probability in locked test set.")

    rows = []

    for _, row in merged.iterrows():

        text = norm(row["description"])
        evidence = analyze(text)

        true_label = (
            1
            if str(row["final_sif_potential"]).upper() == "YES"
            else 0
            if str(row["final_sif_potential"]).upper() == "NO"
            else None
        )

        probability = float(row["sif_precursor_probability"])
        conflict = conflict_type(
            true_label,
            probability,
            evidence,
        )

        rows.append({
            "report_id": row["report_id"],
            "true_label": true_label,
            "true_sif_potential": row["final_sif_potential"],
            "sif_precursor_probability": probability,
            "prediction_at_0.50": int(probability >= 0.50),
            **evidence,
            "conflict_type": conflict,
            "reviewer_priority": reviewer_priority(
                probability,
                evidence,
                conflict,
            ),
            "reviewer_interpretation": interpretation(
                probability,
                evidence,
                true_label,
                conflict,
            ),
        })

    out = pd.DataFrame(rows)

    out.to_csv(
        OUTPUT_DIR / "locked_test_reviewer_report_v1.5.csv",
        index=False,
        encoding="utf-8-sig",
    )

    conflicts = out[out["conflict_type"] != "NONE"].copy()

    conflicts.to_csv(
        OUTPUT_DIR / "locked_test_conflicts_v1.5.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "records": int(len(out)),
        "hazard_signal_rate": float(out["hazard_signal"].mean()),
        "exposure_signal_rate": float(out["exposure_signal"].mean()),
        "mechanism_pathway_signal_rate": float(out["pathway_signal"].mean()),
        "complete_pathway_rate": float(out["complete_pathway"].mean()),
        "context_status_counts": (
            out["sif_context_status"].value_counts().to_dict()
        ),
        "assessment_counts": (
            out["pathway_assessment"].value_counts().to_dict()
        ),
        "priority_counts": (
            out["reviewer_priority"].value_counts().sort_index().to_dict()
        ),
        "conflict_counts": (
            out["conflict_type"].value_counts().to_dict()
        ),
        "false_positive_count": int(
            (
                (out["true_label"] == 0)
                & (out["prediction_at_0.50"] == 1)
            ).sum()
        ),
        "false_negative_count": int(
            (
                (out["true_label"] == 1)
                & (out["prediction_at_0.50"] == 0)
            ).sum()
        ),
        "high_score_no_count": int(
            (
                (out["true_label"] == 0)
                & (out["sif_precursor_probability"] >= 0.75)
            ).sum()
        ),
        "low_score_yes_with_explicit_pathway": int(
            (
                (out["true_label"] == 1)
                & (out["sif_precursor_probability"] < 0.50)
                & (out["complete_pathway"] == 1)
            ).sum()
        ),
    }

    (OUTPUT_DIR / "summary_v1.5.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    rules = {
        "version": "1.5",
        "principles": [
            "Hazard is not equivalent to SIF potential.",
            "Exposure is not equivalent to SIF potential.",
            "Mechanism/pathway evidence is separated from consequence context.",
            "Localized/minor wording limits SIF sufficiency unless stronger evidence exists.",
            "Explicit no-personnel-affected wording is surfaced as contextual evidence.",
            "De-energized/disabled and protective-equipment wording is preserved as context, not automatic NO.",
            "High-score conflicts and low-score incidents with explicit pathways are separately flagged.",
            "Annotation labels are not model features.",
            "This engine supports HSSE review and does not make autonomous safety decisions.",
        ],
    }

    (OUTPUT_DIR / "evidence_rules_v1.5.json").write_text(
        json.dumps(rules, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("RAKSHAK EVIDENCE ENGINE v1.5")
    print("=" * 80)
    print(json.dumps(summary, indent=2))

    print("\nCONFLICT CASES")
    cols = [
        "report_id",
        "true_sif_potential",
        "sif_precursor_probability",
        "conflict_type",
        "hazards",
        "exposures",
        "pathways",
        "minor_context",
        "severe_context",
        "context_flags",
        "sif_context_status",
        "reviewer_priority",
    ]
    print(conflicts[cols].to_string(index=False))

    print("\nKEY REVIEW QUEUE")
    queue = out[out["reviewer_priority"] == "P1"].sort_values(
        "sif_precursor_probability",
        ascending=False,
    )
    print(
        queue[
            [
                "report_id",
                "true_sif_potential",
                "sif_precursor_probability",
                "conflict_type",
                "pathways",
                "sif_context_status",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
