"""
RAKSHAK SIF Feature Builder v0.1
================================

Converts v0.6 safety extraction + original narrative into a leakage-safe,
machine-learning-ready feature table.

Purpose
-------
Build structured features for the final SIF model without consuming:
- sif_potential
- potential_accident_level
- annotation notes
- human adjudication fields

The output combines:
1. ontology indicators from v0.6
2. exposure / consequence / barrier indicators
3. mechanism combinations
4. quantitative/context signals parsed from the narrative
5. evidence-count features
6. LSR candidate indicators

This is a feature-engineering layer, not a classifier.

Inputs
------
experiments/safety_information_extraction_v0.6/safety_extraction_flat.csv
data/annotations/resolved_annotations_v0.2.csv

Outputs
-------
experiments/sif_feature_builder_v0.1/
    sif_features.csv
    feature_manifest.json
    feature_summary.json
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXTRACTION_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "safety_information_extraction_v0.6"
    / "safety_extraction_flat.csv"
)

ANNOTATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "resolved_annotations_v0.2.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "sif_feature_builder_v0.1"


def clean(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def labels(v) -> List[str]:
    text = clean(v)
    if not text:
        return []
    return [x.strip() for x in text.split(";") if x.strip()]


def lower(text: str) -> str:
    return re.sub(r"\s+", " ", clean(text).lower())


def any_term(text: str, terms: List[str]) -> bool:
    value = lower(text)
    return any(t.lower() in value for t in terms)


def any_regex(text: str, patterns: List[str]) -> bool:
    value = lower(text)
    return any(re.search(p, value, flags=re.I) for p in patterns)


def safe_count(value) -> int:
    return len(labels(value))


def max_number_near(text: str, patterns: List[str]) -> float:
    """
    Find numbers associated with one of the supplied context patterns.
    Returns the maximum numeric value found, or NaN.
    """
    value = lower(text)
    nums = []

    for pattern in patterns:
        for m in re.finditer(pattern, value, flags=re.I):
            start = max(0, m.start() - 45)
            end = min(len(value), m.end() + 45)
            window = value[start:end]

            for n in re.findall(r"(?<![a-z])(\d+(?:[.,]\d+)?)", window):
                try:
                    nums.append(float(n.replace(",", ".")))
                except ValueError:
                    continue

    return max(nums) if nums else np.nan


# ---------------------------------------------------------------------------
# Canonical ontology flags
# ---------------------------------------------------------------------------

HAZARD_FEATURES = {
    "electrical": "Electrical energy",
    "mechanical": "Mechanical / moving equipment energy",
    "pressure": "Pressure / stored energy",
    "chemical": "Chemical energy / hazardous substance",
    "thermal": "Thermal energy",
    "fire_explosion": "Fire / explosion energy",
    "vehicle": "Vehicle / mobile equipment energy",
    "dropped_object": "Falling object / dropped object energy",
    "fall_height": "Fall-from-height energy",
    "caught_between": "Caught-between / pinch energy",
    "sharp_object": "Sharp-object energy",
    "biological": "Biological / animal hazard",
    "ground": "Ground / unstable terrain",
    "slip_trip": "Slip / trip surface hazard",
}

EXPOSURE_FEATURES = {
    "direct_contact": "Direct physical contact",
    "line_of_fire": "Line-of-fire exposure",
    "caught_between": "Caught-between / pinch exposure",
    "fall": "Fall exposure",
    "chemical": "Chemical exposure",
    "thermal": "Thermal exposure",
    "electrical": "Electrical exposure",
    "vehicle": "Vehicle exposure",
}

ACTUAL_FEATURES = {
    "fatality": "Fatality",
    "electrical_injury": "Electrical injury",
    "crush_amputation": "Crushing / amputation",
    "burn_thermal": "Burn / thermal injury",
    "chemical_injury": "Chemical injury",
    "cut_laceration": "Cut / laceration",
    "impact": "Struck-by / impact injury",
    "fall_injury": "Fall injury",
    "facial_injury": "Eye / facial injury",
    "musculoskeletal": "Musculoskeletal injury",
    "minor_irritation": "Minor irritation / swelling",
}

POTENTIAL_FEATURES = {
    "severe_electrical": "Potential severe electrical injury",
    "severe_caught_between": "Potential severe caught-between injury",
    "severe_fall": "Potential severe fall injury",
    "severe_vehicle": "Potential severe vehicle injury",
    "severe_chemical": "Potential severe chemical injury",
    "severe_thermal": "Potential severe thermal injury",
    "severe_projectile": "Potential severe projectile injury",
    "severe_entanglement": "Potential severe entanglement / machine injury",
}

BARRIER_LABELS = {
    "ppe": "PPE",
    "guard": "Guard / physical protection",
    "isolation": "Isolation / LOTO",
    "deenergization": "De-energization",
    "fall_protection": "Fall protection",
    "exclusion": "Exclusion / traffic control",
    "procedure": "Procedure / JSA / risk assessment",
    "emergency_detection": "Emergency / detection control",
}


def build_flags(row: pd.Series) -> Dict[str, int]:
    hazards = set(labels(row.get("hazards")))
    exposures = set(labels(row.get("exposures")))
    actual = set(labels(row.get("actual_consequences")))
    potential = set(labels(row.get("potential_consequences")))
    barriers = lower(row.get("observed_barriers"))

    features: Dict[str, int] = {}

    for name, label in HAZARD_FEATURES.items():
        features[f"hazard_{name}"] = int(label in hazards)

    for name, label in EXPOSURE_FEATURES.items():
        features[f"exposure_{name}"] = int(label in exposures)

    for name, label in ACTUAL_FEATURES.items():
        features[f"actual_{name}"] = int(label in actual)

    for name, label in POTENTIAL_FEATURES.items():
        features[f"potential_{name}"] = int(label in potential)

    for name, label in BARRIER_LABELS.items():
        features[f"barrier_{name}_present"] = int(
            f"{label.lower()} [present]" in barriers
            or any(
                item.lower().startswith(label.lower())
                and "[present]" in item.lower()
                for item in labels(row.get("observed_barriers"))
            )
        )
        features[f"barrier_{name}_failed"] = int(
            any(
                item.lower().startswith(label.lower())
                and "[failed_or_absent]" in item.lower()
                for item in labels(row.get("observed_barriers"))
            )
        )

    return features


def build_context_features(row: pd.Series, narrative: str) -> Dict[str, object]:
    text = lower(narrative)

    out: Dict[str, object] = {}

    # Narrative size / evidence density.
    out["narrative_chars"] = len(narrative)
    out["narrative_words"] = len(re.findall(r"\b\w+\b", narrative))

    out["hazard_count"] = safe_count(row.get("hazards"))
    out["exposure_count"] = safe_count(row.get("exposures"))
    out["actual_consequence_count"] = safe_count(row.get("actual_consequences"))
    out["potential_consequence_count"] = safe_count(row.get("potential_consequences"))
    out["barrier_count"] = safe_count(row.get("observed_barriers"))
    out["activity_count"] = safe_count(row.get("activities"))

    # Explicit human involvement.
    out["human_actor_present"] = int(any_regex(text, [
        r"\bworker\b", r"\boperator\b", r"\bemployee\b", r"\bmechanic\b",
        r"\btechnician\b", r"\bcollaborator\b", r"\bdriver\b", r"\bperson\b",
    ]))

    # Concrete severity/event language.
    out["fatality_language"] = int(any_regex(text, [
        r"\bfatal\b", r"\bfatality\b", r"\bdied\b", r"\bdeath\b", r"\bkilled\b",
    ]))

    out["hospital_or_medical_escalation"] = int(any_regex(text, [
        r"\bhospital\b", r"\bmedical center\b", r"\bmedical post\b",
        r"\btransferred to\b", r"\bemergency response\b",
    ]))

    out["loss_of_control"] = int(any_regex(text, [
        r"\bloses? control\b", r"\bloss of control\b",
        r"\bcontrol.*(?:failed|did not respond)\b",
        r"\bstarts to reverse\b", r"\bunexpectedly activated\b",
    ]))

    out["unexpected_activation"] = int(any_regex(text, [
        r"\baccidentally activates?\b", r"\bunexpectedly\b.*\bactivated\b",
        r"\bstarts to move\b", r"\bturned on\b",
        r"\bmotor.*running\b", r"\bequipment.*operating\b",
    ]))

    out["projectile_release"] = int(any_regex(text, [
        r"\bproject(?:ed|ion)\b", r"\bexpell(?:ed|ing)?\b",
        r"\breleased\b.*\b(?:fragment|metal|rock|liquid|gas)\b",
        r"\bsplinter\b", r"\bfragment.*impact",
    ]))

    out["containment_release"] = int(any_regex(text, [
        r"\bleak\b", r"\bleakage\b", r"\bspill(?:ed)?\b",
        r"\bruptur(?:e|ed)\b", r"\bprojection of\b", r"\bhot pulp\b",
    ]))

    out["restricted_area_signal"] = int(any_regex(text, [
        r"\binside the chute\b", r"\bconfined space\b",
        r"\brestricted area\b", r"\bfenced area\b",
        r"\belectrical compartment\b",
    ]))

    out["line_of_fire_distance_m"] = max_number_near(
        text,
        [r"\bmeters?\b.*\bline of fire\b", r"\bline of fire\b.*\bmeters?\b"],
    )

    # Height features.
    heights = []
    for m in re.finditer(
        r"(\d+(?:[.,]\d+)?)\s*(?:m|meter|meters)\b.*?(?:height|above|fall|ladder|platform|elevated)",
        text,
        flags=re.I,
    ):
        try:
            heights.append(float(m.group(1).replace(",", ".")))
        except ValueError:
            pass

    # Also accept "... height of 2.98m" / "2.5 meters".
    for m in re.finditer(
        r"(?:height|height of|at an approximate height of)\s*(?:approximately\s*)?"
        r"(\d+(?:[.,]\d+)?)\s*(?:m|meter|meters)\b",
        text,
        flags=re.I,
    ):
        try:
            heights.append(float(m.group(1).replace(",", ".")))
        except ValueError:
            pass

    out["max_height_m"] = max(heights) if heights else np.nan

    out["low_height_event"] = int(
        not np.isnan(out["max_height_m"]) and out["max_height_m"] < 1.0
    )
    out["significant_height_event"] = int(
        not np.isnan(out["max_height_m"]) and out["max_height_m"] >= 2.0
    )

    # Electrical energy quantities.
    volts = []
    for m in re.finditer(r"\b(\d+(?:[.,]\d+)?)\s*v(?:olts?)?\b", text):
        try:
            volts.append(float(m.group(1).replace(",", ".")))
        except ValueError:
            pass

    amps = []
    for m in re.finditer(r"\b(\d+(?:[.,]\d+)?)\s*a(?:mps?)?\b", text):
        try:
            amps.append(float(m.group(1).replace(",", ".")))
        except ValueError:
            pass

    out["voltage_max_v"] = max(volts) if volts else np.nan
    out["current_max_a"] = max(amps) if amps else np.nan

    out["high_voltage_signal"] = int(
        bool(volts) and max(volts) >= 50
    )

    # Vehicle severity context.
    out["vehicle_collision_signal"] = int(any_term(text, [
        "crash", "collision", "hits the", "impact", "turns to its side",
    ]))
    out["vehicle_rollover_signal"] = int(any_term(text, [
        "rollover", "roll over", "turns to its side", "overturn",
    ]))
    out["vehicle_ejection_signal"] = int(any_term(text, [
        "ejection", "ejected", "thrown from", "jumped out of the cabin",
    ]))
    out["vehicle_occupant_signal"] = int(any_term(text, [
        "occupant", "occupants", "driver", "operator", "co-pilot", "copilot",
        "passenger",
    ]))
    out["seat_belt_signal"] = int(any_term(text, [
        "seat belt", "safety belt",
    ]))

    # Exposure-body-part indicators.
    for body in [
        "hand", "finger", "arm", "leg", "foot", "face", "eye",
        "head", "chest", "back", "neck",
    ]:
        out[f"body_{body}_in_narrative"] = int(
            re.search(rf"\b{re.escape(body)}\b", text, flags=re.I) is not None
        )

    # Barrier-gap evidence independent of annotation notes.
    for name, patterns in {
        "ppe_absence_signal": [
            r"\bwithout\b.*\b(?:gloves?|helmet|goggles?|ppe|epps)\b",
            r"\bno\b.*\b(?:gloves?|helmet|goggles?|ppe|epps)\b",
            r"\bdid not\b.*\b(?:use|wear)\b.*\b(?:gloves?|helmet|ppe|epps)\b",
        ],
        "isolation_gap_signal": [
            r"\bnot isolated\b", r"\bnot lock(?:ed)? out\b",
            r"\bwithout isolation\b", r"\benergized\b.*\bwork\b",
        ],
        "guard_gap_signal": [
            r"\bbonnet open\b", r"\bwithout guard\b",
            r"\bguard.*(?:removed|missing)\b",
        ],
        "exclusion_gap_signal": [
            r"\bline of fire\b", r"\bno exclusion\b",
            r"\bnot blocked\b", r"\bnot barricaded\b",
        ],
    }.items():
        out[name] = int(any_regex(text, patterns))

    return out


def build_interactions(row: pd.Series) -> Dict[str, int]:
    """
    Mechanism combinations are often more informative than isolated
    one-hot labels.
    """
    hazards = set(labels(row.get("hazards")))
    exposures = set(labels(row.get("exposures")))
    potential = set(labels(row.get("potential_consequences")))

    def h(x): return x in hazards
    def e(x): return x in exposures
    def p(x): return x in potential

    return {
        # Core exposure-mechanism combinations.
        "ix_electrical_exposure": int(h("Electrical energy") and (
            e("Electrical exposure") or e("Direct physical contact")
        )),
        "ix_mechanical_direct": int(h("Mechanical / moving equipment energy") and (
            e("Direct physical contact") or e("Caught-between / pinch exposure")
        )),
        "ix_mechanical_linefire": int(h("Mechanical / moving equipment energy") and e("Line-of-fire exposure")),
        "ix_vehicle_exposure": int(h("Vehicle / mobile equipment energy") and e("Vehicle exposure")),
        "ix_vehicle_linefire": int(h("Vehicle / mobile equipment energy") and e("Line-of-fire exposure")),
        "ix_fall_exposure": int(h("Fall-from-height energy") and e("Fall exposure")),
        "ix_caught_between": int(h("Caught-between / pinch energy") and e("Caught-between / pinch exposure")),
        "ix_projectile_linefire": int(h("Falling object / dropped object energy") and e("Line-of-fire exposure")),
        "ix_thermal_exposure": int(h("Thermal energy") and e("Thermal exposure")),
        "ix_chemical_exposure": int(h("Chemical energy / hazardous substance") and e("Chemical exposure")),

        # Potential severity combinations.
        "ix_severe_electrical": int(p("Potential severe electrical injury") and h("Electrical energy")),
        "ix_severe_vehicle": int(p("Potential severe vehicle injury") and h("Vehicle / mobile equipment energy")),
        "ix_severe_fall": int(p("Potential severe fall injury") and h("Fall-from-height energy")),
        "ix_severe_caught_between": int(p("Potential severe caught-between injury") and h("Caught-between / pinch energy")),
        "ix_severe_chemical": int(p("Potential severe chemical injury") and h("Chemical energy / hazardous substance")),
        "ix_severe_thermal": int(p("Potential severe thermal injury") and h("Thermal energy")),
        "ix_severe_projectile": int(p("Potential severe projectile injury") and e("Line-of-fire exposure")),

        # Barrier + mechanism.
        "ix_electrical_isolation_gap": int(
            h("Electrical energy")
            and any(
                "Isolation / LOTO [failed_or_absent]".lower() in x.lower()
                or "De-energization [failed_or_absent]".lower() in x.lower()
                for x in labels(row.get("observed_barriers"))
            )
        ),
        "ix_mechanical_guard_gap": int(
            h("Mechanical / moving equipment energy")
            and any(
                "Guard / physical protection [failed_or_absent]".lower() in x.lower()
                for x in labels(row.get("observed_barriers"))
            )
        ),
    }


def build_row(row_ext: pd.Series, narrative: str) -> Dict:
    features: Dict[str, object] = {
        "report_id": clean(row_ext.get("report_id")),
    }

    features.update(build_flags(row_ext))
    features.update(build_context_features(row_ext, narrative))
    features.update(build_interactions(row_ext))

    return features


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not EXTRACTION_PATH.exists():
        raise FileNotFoundError(EXTRACTION_PATH)

    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(ANNOTATION_PATH)

    ext = pd.read_csv(EXTRACTION_PATH)
    ann = pd.read_csv(ANNOTATION_PATH)

    if "report_id" not in ext.columns:
        raise ValueError("Extraction file lacks report_id")

    if "report_id" not in ann.columns or "description" not in ann.columns:
        raise ValueError("Annotation file must contain report_id and description")

    ann_small = ann[
        ["report_id", "description"]
    ].drop_duplicates("report_id")

    merged = ext.merge(
        ann_small,
        on="report_id",
        how="inner",
    )

    rows = []
    for _, row in merged.iterrows():
        rows.append(build_row(row, clean(row.get("description"))))

    feature_df = pd.DataFrame(rows)

    # Ensure booleans become numeric and NaN numeric fields stay NaN.
    for col in feature_df.columns:
        if col == "report_id":
            continue
        if feature_df[col].dtype == bool:
            feature_df[col] = feature_df[col].astype(int)

    csv_path = OUTPUT_DIR / "sif_features.csv"
    feature_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    feature_manifest = {
        "version": "0.1",
        "id_column": "report_id",
        "feature_count": int(feature_df.shape[1] - 1),
        "feature_columns": [
            {
                "name": c,
                "dtype": str(feature_df[c].dtype),
                "missing": int(feature_df[c].isna().sum()),
                "unique": int(feature_df[c].nunique(dropna=True)),
            }
            for c in feature_df.columns
            if c != "report_id"
        ],
        "forbidden_inputs": [
            "sif_potential",
            "potential_accident_level",
            "annotation_confidence",
            "hazard_notes",
            "exposure_notes",
            "consequence_notes",
            "barrier_notes",
        ],
    }

    manifest_path = OUTPUT_DIR / "feature_manifest.json"
    manifest_path.write_text(
        json.dumps(feature_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    numeric = feature_df.drop(columns=["report_id"])
    binary_cols = [
        c for c in numeric.columns
        if set(numeric[c].dropna().unique()).issubset({0, 1})
    ]

    summary = {
        "version": "0.1",
        "records": int(len(feature_df)),
        "features": int(feature_df.shape[1] - 1),
        "binary_features": int(len(binary_cols)),
        "numeric_features": int(feature_df.shape[1] - 1 - len(binary_cols)),
        "feature_groups": {
            "hazards": int(sum(c.startswith("hazard_") for c in feature_df.columns)),
            "exposures": int(sum(c.startswith("exposure_") for c in feature_df.columns)),
            "actual_consequences": int(sum(c.startswith("actual_") for c in feature_df.columns)),
            "potential_consequences": int(sum(c.startswith("potential_") for c in feature_df.columns)),
            "barriers": int(sum(c.startswith("barrier_") for c in feature_df.columns)),
            "interactions": int(sum(c.startswith("ix_") for c in feature_df.columns)),
            "context": int(
                len(feature_df.columns)
                - 1
                - sum(c.startswith("hazard_") for c in feature_df.columns)
                - sum(c.startswith("exposure_") for c in feature_df.columns)
                - sum(c.startswith("actual_") for c in feature_df.columns)
                - sum(c.startswith("potential_") for c in feature_df.columns)
                - sum(c.startswith("barrier_") for c in feature_df.columns)
                - sum(c.startswith("ix_") for c in feature_df.columns)
            ),
        },
    }

    summary_path = OUTPUT_DIR / "feature_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 78)
    print("RAKSHAK SIF FEATURE BUILDER v0.1")
    print("=" * 78)
    print(f"Records        : {summary['records']}")
    print(f"Features       : {summary['features']}")
    print(f"Binary         : {summary['binary_features']}")
    print(f"Numeric        : {summary['numeric_features']}")
    print()
    print("FEATURE GROUPS")
    for group, count in summary["feature_groups"].items():
        print(f"  {group:<22}: {count}")
    print()
    print(f"CSV      : {csv_path}")
    print(f"Manifest : {manifest_path}")
    print(f"Summary  : {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
