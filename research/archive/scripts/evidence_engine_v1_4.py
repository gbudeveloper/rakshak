"""
RAKSHAK Evidence Engine v1.4
============================

Purpose
-------
Fix v1.3 over-suppression by separating:
- explicit narrative evidence
- context signals
- pathway compatibility
- reviewer confidence

v1.3 was too strict because it required multiple tightly coupled regex
patterns to appear literally in the same narrative. v1.4 instead uses
phrase-level scoring and mechanism-specific evidence without using
annotation labels as features.

It does NOT retrain the SIF model.

Outputs:
    experiments/rakshak_evidence_v1.4/
        locked_test_reviewer_report_v1.4.csv
        locked_test_conflicts_v1.4.csv
        summary_v1.4.json
        pathway_rules_v1.4.json
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
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "rakshak_evidence_v1.4"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def hit(text: str, patterns: list[str]) -> str | None:
    for p in patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            return m.group(0)
    return None


def hits(text: str, patterns: list[str]) -> list[str]:
    out = []
    for p in patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            out.append(m.group(0))
    return out


# -------------------------------------------------------------------------
# Evidence dictionaries
# -------------------------------------------------------------------------

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
        r"\bchemical\b.{0,80}\b(?:skin|eye|face|mouth|lip|shoulder|body)\b",
        r"\b(?:skin|eye|face|mouth|lip|shoulder|body)\b.{0,80}\b(?:chemical|solvent|acid|thinner|paint)\b",
    ],
    "thermal_exposure": [
        r"\bburn(?:ed|ing)?\b",
        r"\bscald(?:ed|ing)?\b",
        r"\bhot\b.{0,80}\b(?:contact|skin|body|burn)\b",
        r"\bsteam\b.{0,80}\b(?:contact|skin|body|burn)\b",
        r"\bredness\b.{0,60}\bburning\b",
    ],
    "electrical_exposure": [
        r"\belectric shock\b",
        r"\b(?:received|suffered|got)\b.{0,40}\belectric(?:al)?\b",
        r"\benergized\b.{0,60}\b(?:contact|touched|shock)\b",
        r"\bcontact\b.{0,60}\benergized\b",
    ],
    "vehicle_exposure": [
        r"\bvehicle exposure\b",
        r"\bpedestrian\b",
        r"\b(?:hit|struck|collision|collid|ran over|run over)\b.{0,80}\b(?:vehicle|truck|forklift|crane|loader|excavator)\b",
        r"\b(?:vehicle|truck|forklift|crane|loader|excavator)\b.{0,80}\b(?:hit|struck|collision|collid|ran over|run over)\b",
        r"\b(?:vehicle|truck|forklift|crane|loader|excavator)\b.{0,60}\bperson(?:nel)?\b",
    ],
    "fall_exposure": [
        r"\bfell\b",
        r"\bfall(?:ing)?\b.{0,60}\b(?:from|off)\b",
        r"\b(?:from|off)\b.{0,60}\b(?:height|platform|scaffold|tower|ladder)\b",
    ],
    "line_of_fire": [
        r"\bline of fire\b",
        r"\bstruck by\b",
        r"\bin the path\b",
        r"\btrajectory\b",
        r"\bpath of\b.{0,50}\b(?:flow|object|vehicle|material)\b",
    ],
    "caught_between_exposure": [
        r"\bpinch\b",
        r"\bcaught\b.{0,50}\b(?:between|in|inside)\b",
        r"\btrapped\b.{0,50}\b(?:between|in|inside)\b",
        r"\bcrush(?:ed|ing)?\b",
    ],
    "direct_contact": [
        r"\bcontact(?:ed)?\b",
        r"\btouched\b",
        r"\bprojected onto\b",
        r"\bsplashed\b",
    ],
}

# A pathway is emitted when at least one mechanism-specific phrase is present,
# plus compatible context where necessary.
PATHWAYS = {
    "electrical_contact": {
        "requires_any": EXPOSURES["electrical_exposure"],
        "compatible_hazards": {"electrical"},
    },
    "chemical_release_exposure": {
        "requires_any": EXPOSURES["chemical_exposure"]
        + [
            r"\b(?:chemical|solvent|acid|caustic|thinner|paint)\b.{0,100}\b(?:spill|leak|release|splash|projected)\b",
            r"\b(?:spill|leak|release|splash|projected)\b.{0,100}\b(?:chemical|solvent|acid|caustic|thinner|paint)\b",
        ],
        "compatible_hazards": {"chemical"},
    },
    "thermal_contact": {
        "requires_any": EXPOSURES["thermal_exposure"],
        "compatible_hazards": {"thermal"},
    },
    "fall_from_height": {
        "requires_any": EXPOSURES["fall_exposure"]
        + [
            r"\bfell\b",
            r"\bfall\b.{0,60}\b(?:height|platform|scaffold|tower|ladder)\b",
        ],
        "compatible_hazards": {"fall_height"},
    },
    "struck_by_projectile": {
        "requires_any": [
            r"\bstruck by\b",
            r"\bproject(?:ile|ed)\b.{0,80}\b(?:hit|strike|person|worker)\b",
            r"\beject(?:ed|ion)\b.{0,80}\b(?:hit|strike|person|worker)\b",
            r"\bfalling object\b.{0,80}\b(?:hit|strike|person|worker)\b",
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

# Context modifiers
CONTEXT = {
    "water_mud": [
        r"\bwater\b",
        r"\bmud\b",
    ],
    "no_person_affected": [
        r"\bno presence of personnel\b",
        r"\bno personnel\b.{0,60}\b(?:affected|injur)\b",
        r"\bno one\b.{0,60}\b(?:affected|injur)\b",
    ],
    "deenergized_disabled": [
        r"\bde[- ]?energized\b",
        r"\bdisabled\b",
    ],
    "protective_equipment": [
        r"\b(?:ppe|ep[pi]s?)\b",
        r"\bsafety belt\b",
        r"\bseat belt\b",
    ],
    "distance_context": [
        r"\b\d+\s*(?:m|meter|meters)\b",
        r"\bdistance\b",
    ],
}


def category_matches(text: str, lexicon: dict[str, list[str]]) -> dict[str, str]:
    result = {}
    for category, patterns in lexicon.items():
        m = hit(text, patterns)
        if m:
            result[category] = m
    return result


def contextual_pathways(
    text: str,
    hazards: set[str],
) -> dict[str, str]:

    paths = {}

    for name, spec in PATHWAYS.items():
        m = hit(text, spec["requires_any"])
        if not m:
            continue
        if hazards.intersection(spec["compatible_hazards"]):
            paths[name] = m

    # Special water/mud rule:
    # generic splash/projected/flow is not chemical unless chemical evidence
    # is explicit in the narrative.
    if (
        "water_mud" in context_flags(text)
        and "chemical" not in hazards
        and "chemical_exposure" not in category_matches(text, EXPOSURES)
    ):
        paths.pop("chemical_release_exposure", None)

    return paths


def context_flags(text: str) -> set[str]:
    return {
        key for key, patterns in CONTEXT.items()
        if hit(text, patterns)
    }


def assessment(
    text: str,
) -> dict:

    hazards = category_matches(text, HAZARDS)
    exposures = category_matches(text, EXPOSURES)
    flags = context_flags(text)
    paths = contextual_pathways(text, set(hazards))

    # generic direct contact is context, not a pathway.
    real_exposures = {
        k: v for k, v in exposures.items()
        if k != "direct_contact"
    }

    hazard_signal = bool(hazards)
    exposure_signal = bool(real_exposures)
    pathway_signal = bool(paths)

    support_count = int(hazard_signal) + int(exposure_signal) + int(pathway_signal)

    if support_count == 3:
        status = "SUPPORTED"
    elif support_count == 2:
        status = "PARTIAL"
    elif support_count == 1:
        status = "WEAK"
    else:
        status = "NONE"

    # Explicitly documented absence of affected personnel is a strong
    # contextual downgrade.
    if "no_person_affected" in flags:
        status = "WEAK" if status in {"SUPPORTED", "PARTIAL"} else status

    # De-energized/disabled is contextual evidence, not an automatic NO.
    # It can matter to adjudication, so we expose it without forcing a label.
    return {
        "hazards": " | ".join(sorted(hazards)),
        "hazard_evidence": " | ".join(
            f"{k}: {v}" for k, v in sorted(hazards.items())
        ),
        "exposures": " | ".join(sorted(real_exposures)),
        "exposure_evidence": " | ".join(
            f"{k}: {v}" for k, v in sorted(real_exposures.items())
        ),
        "pathways": " | ".join(sorted(paths)),
        "pathway_evidence": " | ".join(
            f"{k}: {v}" for k, v in sorted(paths.items())
        ),
        "context_flags": " | ".join(sorted(flags)),
        "hazard_signal": int(hazard_signal),
        "exposure_signal": int(exposure_signal),
        "pathway_signal": int(pathway_signal),
        "pathway_signal_strength": support_count,
        "complete_pathway": int(support_count == 3 and bool(paths)),
        "pathway_assessment": status,
    }


def priority(prob: float, e: dict) -> str:
    # Priority is deliberately not equivalent to SIF truth.
    if prob >= 0.80 and e["complete_pathway"]:
        return "P1"
    if prob >= 0.80 and e["pathway_signal_strength"] >= 1:
        return "P2"
    if prob >= 0.50:
        return "P3"
    if prob >= 0.25:
        return "P4"
    return "P5"


def interpretation(prob: float, e: dict, true_label: int | None) -> str:
    if true_label == 0 and prob >= 0.75:
        return (
            "High model score conflicts with adjudicated NO. Mandatory human "
            "review; evidence signals must not override the adjudication."
        )

    if true_label == 1 and prob < 0.50 and e["pathway_signal_strength"] >= 2:
        return (
            "Model score is below the reference threshold despite meaningful "
            "narrative evidence. Review for a possible missed precursor."
        )

    if e["pathway_assessment"] == "SUPPORTED":
        return (
            "Explicit hazard, human-exposure and mechanism evidence detected. "
            "Use as a reviewer-priority signal, not an automatic SIF declaration."
        )

    if e["pathway_assessment"] == "PARTIAL":
        return (
            "Partial hazard/exposure/mechanism evidence detected. Verify missing "
            "context such as severity, energy, proximity, or actual pathway."
        )

    if e["pathway_assessment"] == "WEAK":
        return (
            "Limited safety-context evidence detected. Do not infer a complete "
            "SIF pathway from the available narrative."
        )

    return (
        "No explicit mechanism pathway detected. Treat model output as a "
        "ranking signal requiring human context."
    )


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
            labels[["report_id", "description", "final_sif_potential"]],
            on="report_id",
            how="left",
        )
        .merge(
            predictions[["report_id", "sif_precursor_probability"]],
            on="report_id",
            how="left",
        )
    )

    if merged["description"].isna().any():
        raise ValueError("Missing description in locked test set.")
    if merged["sif_precursor_probability"].isna().any():
        raise ValueError("Missing model probabilities in locked test set.")

    rows = []

    for _, row in merged.iterrows():
        text = norm(row["description"])
        e = assessment(text)

        label = (
            1
            if str(row["final_sif_potential"]).upper() == "YES"
            else 0
            if str(row["final_sif_potential"]).upper() == "NO"
            else None
        )

        prob = float(row["sif_precursor_probability"])

        rows.append({
            "report_id": row["report_id"],
            "true_label": label,
            "true_sif_potential": row["final_sif_potential"],
            "sif_precursor_probability": prob,
            "prediction_at_0.50": int(prob >= 0.50),
            **e,
            "reviewer_priority": priority(prob, e),
            "reviewer_interpretation": interpretation(prob, e, label),
        })

    out = pd.DataFrame(rows)

    out["error_type"] = "CORRECT"
    out.loc[
        (out["true_label"] == 0) & (out["prediction_at_0.50"] == 1),
        "error_type",
    ] = "FALSE_POSITIVE"
    out.loc[
        (out["true_label"] == 1) & (out["prediction_at_0.50"] == 0),
        "error_type",
    ] = "FALSE_NEGATIVE"

    out = out.sort_values(
        ["reviewer_priority", "sif_precursor_probability"],
        ascending=[True, False],
    )

    out.to_csv(
        OUTPUT_DIR / "locked_test_reviewer_report_v1.4.csv",
        index=False,
        encoding="utf-8-sig",
    )

    conflicts = out[out["error_type"] != "CORRECT"].copy()
    conflicts.to_csv(
        OUTPUT_DIR / "locked_test_conflicts_v1.4.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "records": int(len(out)),
        "hazard_signal_rate": float(out["hazard_signal"].mean()),
        "exposure_signal_rate": float(out["exposure_signal"].mean()),
        "mechanism_pathway_signal_rate": float(out["pathway_signal"].mean()),
        "complete_pathway_rate": float(out["complete_pathway"].mean()),
        "priority_counts": out["reviewer_priority"].value_counts().sort_index().to_dict(),
        "assessment_counts": out["pathway_assessment"].value_counts().to_dict(),
        "false_positive_count": int((out["error_type"] == "FALSE_POSITIVE").sum()),
        "false_negative_count": int((out["error_type"] == "FALSE_NEGATIVE").sum()),
        "high_score_no_count": int(
            (
                (out["true_label"] == 0)
                & (out["sif_precursor_probability"] >= 0.75)
            ).sum()
        ),
        "low_score_yes_with_supported_path_count": int(
            (
                (out["true_label"] == 1)
                & (out["sif_precursor_probability"] < 0.50)
                & (out["complete_pathway"] == 1)
            ).sum()
        ),
    }

    (OUTPUT_DIR / "summary_v1.4.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    rule_manifest = {
        "version": "1.4",
        "principles": [
            "Generic contact does not imply electrical contact.",
            "Generic splash/projected/flow does not imply chemical release.",
            "Vehicle pathway requires vehicle/person interaction evidence.",
            "Hazard, exposure and pathway remain separate signals.",
            "Evidence never overrides adjudicated SIF labels.",
            "Missing severity or energy information is not invented.",
            "De-energized/disabled wording is exposed as context only.",
        ],
        "paths": list(PATHWAYS.keys()),
    }

    (OUTPUT_DIR / "pathway_rules_v1.4.json").write_text(
        json.dumps(rule_manifest, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("RAKSHAK EVIDENCE ENGINE v1.4")
    print("=" * 80)
    print(json.dumps(summary, indent=2))

    print("\nCONFLICT CASES")
    print(
        conflicts[
            [
                "report_id",
                "true_sif_potential",
                "sif_precursor_probability",
                "error_type",
                "hazards",
                "exposures",
                "pathways",
                "context_flags",
                "pathway_assessment",
                "reviewer_priority",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
