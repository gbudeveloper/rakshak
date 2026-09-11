"""
RAKSHAK Safety Information Extractor v0.5
=========================================

Major architecture change from v0.4:
- Separates observed/explicit consequence from inferred credible potential consequence.
- Separates observed barrier evidence from inferred barrier gap.
- Uses event/mechanism relations rather than treating human annotation notes as
  exact extraction gold.
- Keeps evidence with every extracted item.

Input:
    data/annotations/resolved_annotations_v0.2.csv

Output:
    experiments/safety_information_extraction_v0.5/
        safety_extraction_flat.csv
        safety_extraction_detailed.jsonl
        summary.json

Diagnostic engineering baseline only. Not a production safety decision system.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "annotations" / "resolved_annotations_v0.2.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "safety_information_extraction_v0.5"


def clean(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", clean(text)).strip()


def hits(text: str, patterns: List[Tuple[str, str]], window: int = 80, limit: int = 8):
    low = text.lower()
    out = []
    seen = set()

    for label, pattern in patterns:
        for m in re.finditer(pattern, low, flags=re.I):
            key = (label.lower(), m.group(0).lower())
            if key in seen:
                continue
            seen.add(key)
            start = max(0, m.start() - window)
            end = min(len(text), m.end() + window)
            out.append({
                "label": label,
                "evidence": text[m.start():m.end()].strip(),
                "context": norm(text[start:end]),
            })
            if len(out) >= limit:
                return out
    return out


HAZARDS = [
    ("Electrical", r"\b(?:\d{2,4}\s*v|\d{2,5}\s*a|electrical|electric|energized|live|phase[- ]to[- ]phase|phase[- ]to[- ]ground|arc|thermomagnetic)\b"),
    ("Mechanical / moving equipment", r"\b(?:blade|cutter|fan|conveyor|rotating|rotation|moving equipment|moving part|shaft|roller|belt|pulley|gear|motor|propeller)\b"),
    ("Pressure / stored pressure", r"\b(?:pressure|pressurized|compressed air|hydraulic|pneumatic|steam|pressure line)\b"),
    ("Chemical", r"\b(?:acid|sulfuric|sulphuric|nitric|hydrochloric|caustic|ammonia|chemical|solvent|chlorine|corrosive|thinner)\b"),
    ("Thermal", r"\b(?:hot gas|hot metal|molten|liquid metal|steam|heat|heated|high temperature|thermal|scald)\b"),
    ("Fire / explosion", r"\b(?:fire|flame|explosion|ignition|combustible|flammable|flash fire)\b"),
    ("Vehicle / mobile equipment", r"\b(?:vehicle|truck|car|bus|forklift|loader|excavator|mobile equipment|reversing|reversed|backing|tanker)\b"),
    ("Falling object / rockfall", r"\b(?:falling object|dropped object|rock block|rock slab|stone slab|stone|metal fragment|splinter|falling material|detachment|detached)\b"),
    ("Fall from height", r"\b(?:fall from height|fell from|falling from|ladder|scaffold|roof|platform|height|reel)\b"),
    ("Caught-between / pinch", r"\b(?:caught[- ]between|pinch|trapped|imprisoned|caught|crushed between|finger.*(?:caught|trapped|imprisoned)|hand.*(?:caught|trapped|imprisoned))\b"),
    ("Sharp object", r"\b(?:knife|sharp edge|sharp corner|sharp object|broken glass|glass shard|blade|metal edge|machete)\b"),
    ("Biological / animal", r"\b(?:bite|snake|animal|insect|bee|wasp|thorn)\b"),
    ("Ground / unstable terrain", r"\b(?:unstable ground|ground support|rock block|rockfall|pit|gallery|gable|loose ground)\b"),
]

EXPOSURES = [
    ("Direct contact", r"\b(?:contact|touched|touches|reached|hits|hit|struck|splashed|projection|projected|grazed|injured)\b"),
    ("Line-of-fire", r"\b(?:line[- ]of[- ]fire|trajectory|projected|projection|released|ejected|thrown|struck|hit)\b"),
    ("Caught-between / pinch", r"\b(?:caught[- ]between|pinch|trapped|imprisoned|caught|crushed between|finger.*(?:caught|trapped|imprisoned)|hand.*(?:caught|trapped|imprisoned))\b"),
    ("Fall exposure", r"\b(?:fell|falling|fall from|ladder|scaffold|height|reel)\b"),
    ("Chemical exposure", r"\b(?:acid|sulfuric|sulphuric|nitric|ammonia|chemical|caustic|corrosive|thinner).{0,100}\b(?:splash|spill|projection|contact|face|eye|skin|body|lip|arm|leg)\b"),
    ("Thermal exposure", r"\b(?:hot gas|hot metal|molten|liquid metal|steam|heat|heated).{0,100}\b(?:face|eye|hand|arm|worker|operator|person|contact|reached|projection)\b"),
    ("Vehicle exposure", r"\b(?:vehicle|truck|loader|tanker|reversing|reversed|backing).{0,100}\b(?:worker|operator|person|employee|hit|struck|collision|crash)\b"),
]

ACTUAL_CONSEQUENCES = [
    ("Fatality", r"\b(?:fatal|fatality|death|died|dead|killed|verify the death)\b"),
    ("Electrical injury", r"\b(?:electric shock|electrical shock|electrocution|electrical injury|electric injury|shock)\b"),
    ("Crushing / amputation", r"\b(?:crush(?:ed|ing)?|crushing|amputation|amputated|severed|atri(?:c|t)ion|imprisoned between|trapped between)\b"),
    ("Burn / thermal injury", r"\b(?:burn(?:ed)?|burn injury|thermal injury|scald(?:ed)?|redness and burning)\b"),
    ("Chemical injury", r"\b(?:chemical burn|acid burn|corrosive burn|chemical injury|burn).{0,80}\b(?:acid|chemical|sulfuric|sulphuric|nitric|ammonia|caustic|corrosive|thinner)\b"),
    ("Cut / laceration", r"\b(?:cut|cuts|cutting|laceration|lacerated|wound|blunt cut|small cuts?|puncture|pierced)\b"),
    ("Struck-by / impact injury", r"\b(?:hit|struck|impact|impacted|grazed|blunt trauma|contusion|bruise)\b"),
    ("Fall injury", r"\b(?:fell|fall(?:en)?).{0,100}\b(?:injur|trauma|pain|fracture|swelling|wound|hurt|concussion|forearm|knee|back)\b"),
    ("Eye / facial injury", r"\b(?:eye|face|facial).{0,80}\b(?:injur|irritat|pain|hit|splash|contact|wound|burn|redness)\b"),
    ("Musculoskeletal injury", r"\b(?:sprain|strain|twist|twisted|lumbar pain|overexertion|musculoskeletal)\b"),
    ("Minor irritation / swelling", r"\b(?:swelling|irritation|irritated|scratch|minor|superficial injury|little trauma|redness)\b"),
]

# Potential consequence is intentionally narrower. It should capture only
# consequences explicitly signaled by a mechanism with a credible severity
# pathway, not import worst-case language from annotations.
POTENTIAL_PATTERNS = [
    ("Potential severe fall injury", r"\b(?:height|ladder|scaffold|reel|chimney|vertical opening).{0,120}\b(?:fall|fell|falls)\b"),
    ("Potential severe electrical injury", r"\b(?:440\s*v|400\s*a|electrical|energized|live|phase[- ]to[- ]ground|arc).{0,120}\b(?:worker|operator|employee|contact|shock|flash|injur|reached)\b"),
    ("Potential severe caught-between injury", r"\b(?:caught[- ]between|imprisoned|trapped|pinch|crushed between).{0,100}\b(?:hand|finger|leg|chest|worker|mechanic|operator)\b"),
    ("Potential severe vehicle injury", r"\b(?:vehicle|truck|tanker|loader|mobile equipment).{0,120}\b(?:crash|collision|roll|turns|reverse|reversing|slides|excavated)\b"),
    ("Potential severe chemical exposure", r"\b(?:sulfuric|sulphuric|nitric|ammonia|corrosive|chemical).{0,120}\b(?:projection|splash|splashes|face|eye|upper limbs|skin|reaches)\b"),
    ("Potential severe thermal injury", r"\b(?:hot gas|molten|liquid metal|steam|high temperature).{0,120}\b(?:reaching|reaches|projection|projected|face|worker|employee)\b"),
    ("Potential severe projectile injury", r"\b(?:splinter|fragment|rock block|rock slab|metal fragment).{0,120}\b(?:project|expell|impact|hit|struck|worker|employee|operator)\b"),
]

OBSERVED_BARRIERS = [
    ("PPE", r"\b(?:helmet|hard hat|gloves?|goggles?|safety glasses|face shield|faceshield|respirator|mask|safety shoes|boots|ppe|epps)\b"),
    ("Guard / physical protection", r"\b(?:guard|guarding|protector|protective guard|cover|shield|barrier|screen|enclosure|fence|fender)\b"),
    ("Isolation / LOTO", r"\b(?:lockout|lock[- ]out|tagout|tag[- ]out|loto|isolation|isolated|isolate|disabled)\b"),
    ("De-energization", r"\b(?:de[- ]energized|deenergized|power off|switched off|shut down|shutdown|disconnected|breaker off)\b"),
    ("Fall protection", r"\b(?:harness|fall arrest|lifeline|lanyard|guardrail|handrail|toe board)\b"),
    ("Exclusion / traffic control", r"\b(?:barricade|barricaded|exclusion zone|restricted area|spotter|banksman|flagman|traffic control|cordoned)\b"),
    ("Permit / authorization", r"\b(?:permit to work|work permit|authorized|authorised)\b"),
    ("Procedure / JSA / risk assessment", r"\b(?:procedure|method statement|job safety analysis|jsa|jha|risk assessment|checklist|instruction)\b"),
]

# Explicit barrier-gap language in the narrative. "No gloves" is extractable;
# "appropriate gloves should have been used" is a recommendation, not observed
# fact, and is not assigned as a failed barrier unless absence is explicit.
BARRIER_GAPS = [
    ("PPE", r"\b(?:without|no|not wearing|did not have|did not use|not using|lack of|missing)\b.{0,45}\b(?:helmet|gloves?|goggles?|safety glasses|face shield|mask|respirator|ppe|epps)\b"),
    ("Guard / physical protection", r"\b(?:without|no|missing|removed|not in place|bypassed)\b.{0,55}\b(?:guard|guarding|protector|cover|shield|barrier|enclosure)\b"),
    ("Isolation / LOTO", r"\b(?:without|no|not|failed|failure|missing|not isolated|not lock(?:ed)? out|not tagged)\b.{0,60}\b(?:isolation|isolat(?:ed|ion)|lockout|lock[- ]out|tagout|tag[- ]out|loto)\b"),
    ("De-energization", r"\b(?:still energized|remained energized|not de[- ]energized|without shutting down|without switching off|without disconnecting|power remained on)\b"),
    ("Fall protection / safe access", r"\b(?:unstable reel|rim position|improvised|without.*(?:ladder|platform)|fell.*ladder|fall(?:ing)?.{0,40}ladder)\b"),
    ("Exclusion / line-of-fire control", r"\b(?:in the line of fire|line[- ]of[- ]fire|person.*inside.*(?:fenced|restricted)|remain(?:ed)? in the exposure area|no presence of personnel|not blocked)\b"),
    ("Structural / ground control", r"\b(?:broken.*(?:anchor|support|rivet)|rock block displacement|stone slab.*detached|unstable ground|loose.*ground)\b"),
]

ACTIVITIES = [
    ("Inspection", r"\b(?:inspection|inspecting|inspect|checking|check|survey)\b"),
    ("Assembly / installation", r"\b(?:assembly|assembling|installation|installing|installed|mounting)\b"),
    ("Maintenance / repair", r"\b(?:maintenance|repair|repairing|overhaul|servicing|service|repair work)\b"),
    ("Adjustment / tightening", r"\b(?:adjustment|adjusting|tightening|loosen(?:ing)?)\b"),
    ("Cleaning", r"\b(?:cleaning|clean|washing|washing down)\b"),
    ("Painting / coating", r"\b(?:painting|paint|coating)\b"),
    ("Lifting / handling", r"\b(?:lifting|lifted|rigging|handling|carrying)\b"),
    ("Working at height", r"\b(?:working at height|ladder|scaffold|roof|rooftop|elevated|reel)\b"),
    ("Driving / vehicle movement", r"\b(?:driving|reversing|reversed|backing|vehicle movement|transit)\b"),
    ("Cutting / drilling / grinding", r"\b(?:cutting|cutter|drilling|drill|grinding|machete)\b"),
    ("Process operation", r"\b(?:furnace|autoclave|hopper|conveyor|pump|tank|chute)\b"),
]


def extract_standard(text: str, bank, confidence=0.82):
    return [
        {
            "label": x["label"],
            "evidence": x["evidence"],
            "context": x["context"],
            "confidence": confidence,
        }
        for x in hits(text, bank)
    ]


def unique_labels(items):
    seen = set()
    out = []
    for x in items:
        if x["label"].lower() not in seen:
            seen.add(x["label"].lower())
            out.append(x)
    return out


def extract_barriers(text: str):
    observed = extract_standard(text, OBSERVED_BARRIERS, 0.79)
    gaps = []

    for label, pattern in BARRIER_GAPS:
        for m in re.finditer(pattern, text.lower(), flags=re.I):
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 60)
            gaps.append({
                "label": label,
                "status": "failed_or_absent",
                "evidence": text[m.start():m.end()].strip(),
                "context": norm(text[start:end]),
                "confidence": 0.88,
            })
            break

    # Explicitly observed PPE remains "present" even when a different barrier
    # gap is inferred elsewhere. This preserves the distinction.
    obs = [
        {
            "label": x["label"],
            "status": "present",
            "evidence": x["evidence"],
            "context": x["context"],
            "confidence": x["confidence"],
        }
        for x in observed
    ]

    merged = []
    seen = set()
    for item in gaps + obs:
        key = (item["label"].lower(), item["status"])
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)
    if "report_id" not in df.columns or "description" not in df.columns:
        raise ValueError("Input must contain report_id and description")

    flat = []
    detailed = []

    for _, row in df.iterrows():
        rid = clean(row["report_id"])
        text = norm(row["description"])

        hazards = extract_standard(text, HAZARDS)
        exposures = extract_standard(text, EXPOSURES)
        actual = extract_standard(text, ACTUAL_CONSEQUENCES, 0.80)
        potential = extract_standard(text, POTENTIAL_PATTERNS, 0.72)
        barriers = extract_barriers(text)
        activities = extract_standard(text, ACTIVITIES, 0.82)

        flat.append({
            "report_id": rid,
            "hazards": "; ".join(unique["label"] for unique in hazards),
            "hazard_evidence": "; ".join(unique["evidence"] for unique in hazards),
            "exposures": "; ".join(unique["label"] for unique in exposures),
            "exposure_evidence": "; ".join(unique["evidence"] for unique in exposures),

            "actual_consequences": "; ".join(unique["label"] for unique in actual),
            "actual_consequence_evidence": "; ".join(unique["evidence"] for unique in actual),

            "potential_consequences": "; ".join(unique["label"] for unique in potential),
            "potential_consequence_evidence": "; ".join(unique["evidence"] for unique in potential),

            "observed_barriers": "; ".join(
                f'{x["label"]} [{x["status"]}]' for x in barriers
            ),
            "barrier_evidence": "; ".join(x["evidence"] for x in barriers),

            "activities": "; ".join(unique["label"] for unique in activities),
            "activity_evidence": "; ".join(unique["evidence"] for unique in activities),

            "hazard_count": len(hazards),
            "exposure_count": len(exposures),
            "actual_consequence_count": len(actual),
            "potential_consequence_count": len(potential),
            "barrier_count": len(barriers),
            "activity_count": len(activities),
        })

        detailed.append({
            "report_id": rid,
            "description": text,
            "hazards": hazards,
            "exposures": exposures,
            "actual_consequences": actual,
            "potential_consequences": potential,
            "barriers": barriers,
            "activities": activities,
        })

    out_df = pd.DataFrame(flat)
    csv_path = OUTPUT_DIR / "safety_extraction_flat.csv"
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    jsonl_path = OUTPUT_DIR / "safety_extraction_detailed.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in detailed:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def coverage(col):
        return round(
            100 * out_df[col].fillna("").astype(str).str.strip().ne("").mean(), 2
        )

    summary = {
        "version": "0.5",
        "records": int(len(out_df)),
        "coverage_percent": {
            "hazards": coverage("hazards"),
            "exposures": coverage("exposures"),
            "actual_consequences": coverage("actual_consequences"),
            "potential_consequences": coverage("potential_consequences"),
            "observed_barriers_or_gaps": coverage("observed_barriers"),
            "activities": coverage("activities"),
        },
        "mean_items_per_record": {
            "hazards": round(float(out_df["hazard_count"].mean()), 3),
            "exposures": round(float(out_df["exposure_count"].mean()), 3),
            "actual_consequences": round(float(out_df["actual_consequence_count"].mean()), 3),
            "potential_consequences": round(float(out_df["potential_consequence_count"].mean()), 3),
            "barriers": round(float(out_df["barrier_count"].mean()), 3),
            "activities": round(float(out_df["activity_count"].mean()), 3),
        },
        "design_note": (
            "Human annotation notes can contain counterfactual or recommended "
            "controls that are not explicit in the narrative. v0.5 therefore "
            "separates actual observed consequence, potential consequence signal, "
            "observed barrier, and explicit barrier gap."
        ),
    }

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 78)
    print("RAKSHAK SAFETY INFORMATION EXTRACTOR v0.5")
    print("=" * 78)
    print(f"Records : {len(out_df)}")
    print()
    for key, value in summary["coverage_percent"].items():
        print(f"{key:<32}: {value:5.1f}%")
    print()
    for key, value in summary["mean_items_per_record"].items():
        print(f"{key:<32}: {value:5.2f}")
    print()
    print(f"Flat CSV : {csv_path}")
    print(f"Detailed : {jsonl_path}")
    print(f"Summary  : {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
