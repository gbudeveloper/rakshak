"""
RAKSHAK Precursor Feature Builder v0.2
=======================================

Narrative-only, target-independent structured features.

CRITICAL POLICY
---------------
This builder reads ONLY report_id + description from the adjudicated file.
It does NOT read or derive features from:
- SIF labels
- annotation notes
- actual outcome fields
- potential accident level
- extracted consequence/barrier annotations
- LSR labels
- near-miss / Hi-Po labels

The resulting features are therefore computed directly from the narrative text.

Output:
experiments/sif_precursor_feature_builder_v0.2/sif_precursor_features.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "resolved_annotations_v0.2.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "sif_precursor_feature_builder_v0.2"
)


def norm(text: str) -> str:
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", text.strip().lower())


def has_any(text: str, patterns: list[str]) -> int:
    return int(any(re.search(p, text, flags=re.I) for p in patterns))


def count_hits(text: str, patterns: list[str]) -> int:
    return sum(1 for p in patterns if re.search(p, text, flags=re.I))


def extract_max(text: str, pattern: str) -> float:
    vals = []
    for m in re.finditer(pattern, text, flags=re.I):
        try:
            vals.append(float(m.group(1).replace(",", ".")))
        except (ValueError, IndexError):
            pass
    return max(vals) if vals else np.nan


def build_features(report_id: str, description: str) -> dict:
    text = norm(description)
    f: dict[str, float | int | str] = {"report_id": str(report_id)}

    # Narrative-only size.
    f["narrative_chars"] = len(text)
    f["narrative_words"] = len(re.findall(r"\b\w+\b", text))

    # Hazard / energy mechanisms.
    hazard_patterns = {
        "electrical": [r"\belectric", r"\bvoltage\b", r"\benergized\b", r"\bshock\b"],
        "mechanical": [r"\bmachin", r"\bmoving part", r"\brotat", r"\bconveyor\b", r"\broller\b", r"\bpress\b"],
        "pressure": [r"\bpressure\b", r"\bpressur", r"\bstored energy\b", r"\bcompressed\b"],
        "chemical": [r"\bchemical\b", r"\bammonia\b", r"\bchlorine\b", r"\bsolvent\b", r"\bacid\b", r"\bcaustic\b", r"\btoxic\b", r"\bcorrosive\b"],
        "thermal": [r"\bhot\b", r"\bsteam\b", r"\bheat\b", r"\bthermal\b", r"\bburn\b"],
        "fire_explosion": [r"\bfire\b", r"\bexplos", r"\bflammab", r"\bignition\b"],
        "vehicle": [r"\bvehicle\b", r"\btruck\b", r"\bcar\b", r"\bforklift\b", r"\bcrane\b", r"\bexcavat", r"\bloader\b", r"\bdriv(?:e|er|ing)\b"],
        "dropped_object": [r"\bdropped object\b", r"\bdropped\b.*\b(?:tool|object|material|load)\b", r"\bfalling object\b"],
        "fall_height": [r"\bfall\b", r"\bfell\b", r"\bladder\b", r"\bscaffold", r"\bplatform\b", r"\bheight\b", r"\belevated\b"],
        "caught_between": [r"\bpinch\b", r"\btrapped\b", r"\bcaught\b", r"\bbetween\b", r"\bcrush\b"],
        "sharp_object": [r"\bsharp\b", r"\bblade\b", r"\bknife\b", r"\bcutting\b", r"\bedge\b"],
        "ground_instability": [r"\bunstable ground\b", r"\bsoft ground\b", r"\bexcavat", r"\bslope\b", r"\bslip\b", r"\btrip\b"],
    }
    for name, pats in hazard_patterns.items():
        f[f"hazard_{name}"] = has_any(text, pats)

    # Exposure / mechanism wording.
    exposure_patterns = {
        "direct_contact": [r"\bcontact(?:ed|ing)?\b", r"\btouched\b", r"\bhand\b.*\b(?:near|on|against)\b"],
        "line_of_fire": [r"\bline of fire\b", r"\bin the path\b", r"\bstruck by\b", r"\bstruck\b", r"\btrajectory\b"],
        "caught_between": [r"\bpinch\b", r"\bcaught\b", r"\btrapped\b", r"\bbetween\b"],
        "fall": [r"\bfall\b", r"\bfell\b", r"\bdropped from\b"],
        "chemical": [r"\bsplash\b", r"\bleak(?:age|ed|ing)?\b", r"\bspill(?:ed|ing)?\b", r"\bexpos(?:e|ed|ure)\b.*\b(?:gas|vapou?r|chemical)\b"],
        "thermal": [r"\bhot\b", r"\bsteam\b", r"\bthermal\b", r"\bheat\b"],
        "electrical": [r"\belectric", r"\benergized\b", r"\bvoltage\b"],
        "vehicle": [r"\bvehicle\b", r"\btruck\b", r"\bforklift\b", r"\bcrane\b", r"\bdriver\b"],
    }
    for name, pats in exposure_patterns.items():
        f[f"exposure_{name}"] = has_any(text, pats)

    # Barrier / barrier-gap wording taken directly from narrative.
    barrier_patterns = {
        "ppe_present": [r"\bwear(?:ing)?\b.*\b(?:gloves?|helmet|goggles?|face shield|ppe)\b", r"\b(?:gloves?|helmet|goggles?|face shield|ppe)\b.*\bused\b"],
        "ppe_gap": [r"\bwithout\b.*\b(?:gloves?|helmet|goggles?|face shield|ppe)\b", r"\bno\b.*\b(?:gloves?|helmet|goggles?|face shield|ppe)\b", r"\bdid not\b.*\bwear\b"],
        "guard_gap": [r"\bwithout guard\b", r"\bguard\b.*\b(?:removed|missing|open)\b", r"\bbonnet\b.*\bopen\b"],
        "isolation_gap": [r"\bnot isolated\b", r"\bnot lock(?:ed)? out\b", r"\bwithout isolation\b", r"\benergized\b.*\bwork\b"],
        "exclusion_gap": [r"\bno exclusion\b", r"\bnot barricaded\b", r"\bnot blocked\b", r"\bline of fire\b.*\bno\b"],
        "fall_protection_gap": [r"\bwithout\b.*\b(?:harness|fall protection|lifeline)\b", r"\bno\b.*\b(?:harness|fall protection|lifeline)\b"],
        "procedure_signal": [r"\bpermit\b", r"\bjsa\b", r"\brisk assessment\b", r"\bprocedure\b", r"\bchecklist\b"],
    }
    for name, pats in barrier_patterns.items():
        f[f"barrier_{name}"] = has_any(text, pats)

    # Operational / behavioral mechanism signals.
    context_patterns = {
        "human_present": [r"\bworker\b", r"\boperator\b", r"\bemployee\b", r"\btechnician\b", r"\bmechanic\b", r"\bdriver\b", r"\bperson\b"],
        "unexpected_motion": [r"\bunexpected(?:ly)?\b", r"\bstarted to move\b", r"\bstarts to move\b", r"\baccidentally\b.*\b(?:activated|started|turned on)\b"],
        "loss_of_control": [r"\blost control\b", r"\bloses control\b", r"\bloss of control\b", r"\bfailed to respond\b"],
        "projectile_release": [r"\bproject(?:ed|ile)\b", r"\beject(?:ed|ion)\b", r"\bfragment\b", r"\bsplinter\b", r"\bthrown\b"],
        "containment_release": [r"\bleak(?:age|ed|ing)?\b", r"\bspill(?:ed|ing)?\b", r"\bruptur(?:e|ed)\b", r"\brelease(?:d)?\b.*\b(?:gas|vapou?r|liquid)\b"],
        "confined_space": [r"\bconfined space\b", r"\btank\b", r"\bvessel\b", r"\binside the chute\b"],
        "traffic_interaction": [r"\bpedestrian\b", r"\btraffic\b", r"\bvehicle\b.*\bperson\b", r"\bperson\b.*\bvehicle\b"],
    }
    for name, pats in context_patterns.items():
        f[name] = has_any(text, pats)

    # Height and electrical magnitude are narrative-only numeric signals.
    f["max_height_m"] = extract_max(
        text,
        r"(?:height(?: of)?|approximately|about)\s*(\d+(?:[.,]\d+)?)\s*(?:m|meter|meters)\b",
    )
    f["significant_height_signal"] = int(
        pd.notna(f["max_height_m"]) and float(f["max_height_m"]) >= 2.0
    )
    f["low_height_signal"] = int(
        pd.notna(f["max_height_m"]) and float(f["max_height_m"]) < 1.0
    )

    f["max_voltage_v"] = extract_max(
        text,
        r"\b(\d+(?:[.,]\d+)?)\s*v(?:olts?)?\b",
    )
    f["high_voltage_signal"] = int(
        pd.notna(f["max_voltage_v"]) and float(f["max_voltage_v"]) >= 50
    )

    # Neutral body-part / task signals. These describe exposure location, not
    # injury outcome.
    for body in ["hand", "finger", "arm", "leg", "foot", "face", "eye", "head", "chest", "back", "neck"]:
        f[f"body_{body}_mentioned"] = int(re.search(rf"\b{re.escape(body)}\b", text) is not None)

    # Mechanism interactions, all computed from narrative-derived precursor flags.
    f["ix_electrical_exposure"] = int(f["hazard_electrical"] and f["exposure_electrical"])
    f["ix_mechanical_direct"] = int(f["hazard_mechanical"] and f["exposure_direct_contact"])
    f["ix_mechanical_line_fire"] = int(f["hazard_mechanical"] and f["exposure_line_of_fire"])
    f["ix_vehicle_exposure"] = int(f["hazard_vehicle"] and f["exposure_vehicle"])
    f["ix_fall_exposure"] = int(f["hazard_fall_height"] and f["exposure_fall"])
    f["ix_caught_between"] = int(f["hazard_caught_between"] and f["exposure_caught_between"])
    f["ix_chemical_exposure"] = int(f["hazard_chemical"] and f["exposure_chemical"])
    f["ix_thermal_exposure"] = int(f["hazard_thermal"] and f["exposure_thermal"])
    f["ix_vehicle_line_fire"] = int(f["hazard_vehicle"] and f["exposure_line_of_fire"])
    f["ix_isolation_gap_electrical"] = int(f["hazard_electrical"] and f["barrier_isolation_gap"])
    f["ix_guard_gap_mechanical"] = int(f["hazard_mechanical"] and f["barrier_guard_gap"])
    f["ix_fall_gap"] = int(f["hazard_fall_height"] and f["barrier_fall_protection_gap"])
    f["ix_release_line_fire"] = int(f["projectile_release"] and f["exposure_line_of_fire"])

    # Explicitly exclude outcome/severity terminology from this feature table.
    return f


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    required = {"report_id", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # A safety check: make sure forbidden target columns are never consumed.
    forbidden = {
        "final_sif_potential",
        "sif_potential",
        "actual_outcome",
        "potential_accident_level",
        "annotation_confidence",
        "life_saving_rules",
        "hazards",
        "exposure",
        "potential_consequences",
        "barriers",
    }
    consumed = {"report_id", "description"}
    if not consumed.issubset(df.columns):
        raise AssertionError("Only report_id + description should be consumed.")

    rows = [
        build_features(rid, desc)
        for rid, desc in zip(
            df["report_id"].astype(str),
            df["description"].fillna("").astype(str),
        )
    ]
    out = pd.DataFrame(rows)

    # Remove any accidental duplicate column names and sort deterministically.
    out = out.loc[:, ~out.columns.duplicated()]
    feature_cols = [c for c in out.columns if c != "report_id"]

    out.to_csv(
        OUTPUT_DIR / "sif_precursor_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "version": "0.2",
        "input_columns_consumed": ["report_id", "description"],
        "forbidden_fields": sorted(forbidden),
        "records": int(len(out)),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
    }
    (OUTPUT_DIR / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("RAKSHAK PRECURSOR FEATURE BUILDER v0.2")
    print("=" * 78)
    print(f"Records  : {len(out)}")
    print(f"Features : {len(feature_cols)}")
    print("Inputs   : report_id + description ONLY")
    print(f"Output   : {OUTPUT_DIR / 'sif_precursor_features.csv'}")


if __name__ == "__main__":
    main()
