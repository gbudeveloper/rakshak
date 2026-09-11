"""
RAKSHAK Safety Intelligence Extraction v0.6
===========================================

Purpose
-------
v0.5 exposed three important problems:
1. Repeated labels from multiple keyword hits.
2. Actual and potential consequence are easy to conflate.
3. Barrier extraction must distinguish observed controls from explicit gaps.

v0.6 therefore:
- canonicalizes and deduplicates ontology labels;
- adds mechanism-aware consequence inference;
- keeps actual consequence and potential consequence separate;
- uses explicit narrative evidence for barrier gaps;
- adds relation-oriented "mechanism" records;
- does not read human annotation notes;
- does not use SIF labels as extraction inputs.

Input:
    data/annotations/resolved_annotations_v0.2.csv

Output:
    experiments/safety_information_extraction_v0.6/
        safety_extraction_flat.csv
        safety_extraction_detailed.jsonl
        summary.json
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "annotations" / "resolved_annotations_v0.2.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "safety_information_extraction_v0.6"


def clean(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def norm(v: str) -> str:
    return re.sub(r"\s+", " ", clean(v)).strip()


def regex_hits(text: str, patterns: List[Tuple[str, str]], window: int = 80, limit: int = 20):
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


def dedup(items: List[dict]) -> List[dict]:
    """
    Canonical-label dedupe.
    Preserve the strongest/longest evidence for each label.
    """
    best: Dict[str, dict] = {}

    for item in items:
        key = item["label"].strip().lower()
        if not key:
            continue

        current = best.get(key)
        if current is None:
            best[key] = item
            continue

        # Prefer a more informative evidence span.
        if len(item.get("evidence", "")) > len(current.get("evidence", "")):
            best[key] = item

    return list(best.values())


# ---------------------------------------------------------------------------
# Canonical ontology
# ---------------------------------------------------------------------------

HAZARD_PATTERNS = [
    ("Electrical energy",
     r"\b(?:\d{2,4}\s*v(?:olts?)?|\d{2,5}\s*a(?:mps?)?|electrical|electric|energized|energised|live|phase[- ]to[- ]phase|phase[- ]to[- ]ground|electric arc|arc|thermomagnetic|power cell|transformer)\b"),

    ("Mechanical / moving equipment energy",
     r"\b(?:blade|cutter|fan|conveyor|rotating|rotation|moving equipment|moving part|shaft|roller|belt|pulley|gear|motor|propeller|drill arm|guillotine)\b"),

    ("Pressure / stored energy",
     r"\b(?:pressure|pressurized|pressurised|compressed air|hydraulic|pneumatic|steam|pressure line|stored energy|hydraulic module)\b"),

    ("Chemical energy / hazardous substance",
     r"\b(?:acid|sulfuric|sulphuric|nitric|hydrochloric|caustic|ammonia|chemical|solvent|chlorine|corrosive|thinner|copper sulphate)\b"),

    ("Thermal energy",
     r"\b(?:hot gas|hot metal|molten|liquid metal|steam|heat|heated|high temperature|thermal|scald|furnace|hot pulp)\b"),

    ("Fire / explosion energy",
     r"\b(?:fire|flame|explosion|ignition|combustible|flammable|flash fire|explosive)\b"),

    ("Vehicle / mobile equipment energy",
     r"\b(?:vehicle|truck|car|bus|forklift|loader|excavator|mobile equipment|reversing|reversed|backing|tanker|scooptram|mixer|tipper)\b"),

    ("Falling object / dropped object energy",
     r"\b(?:falling object|dropped object|dropped load|rock block|rock slab|stone slab|stone|metal fragment|splinter|fragment|falling material|detachment|detached|roof material)\b"),

    ("Fall-from-height energy",
     r"\b(?:fall from height|fell from|falling from|falls from|ladder|scaffold|scaffolding|roof|platform|height|reel|chimney|vertical opening)\b"),

    ("Caught-between / pinch energy",
     r"\b(?:caught[- ]between|pinch|trapped|imprisoned|caught|crushed between|atri(?:c|t)ion|finger.*(?:caught|trapped|imprisoned)|hand.*(?:caught|trapped|imprisoned))\b"),

    ("Sharp-object energy",
     r"\b(?:knife|machete|sharp edge|sharp corner|sharp object|broken glass|glass shard|blade|metal edge|broken plate|broken dish)\b"),

    ("Biological / animal hazard",
     r"\b(?:bite|snake|animal|insect|bee|wasp|thorn|vegetation|foliage)\b"),

    ("Ground / unstable terrain",
     r"\b(?:unstable ground|ground support|rockfall|loose ground|loose rock|gallery|gable|pit|excavated area|bank)\b"),

    ("Slip / trip surface hazard",
     r"\b(?:slipped|slip|stumbled|stumble|unbalanced|loss of balance|foliage|mud|water on platform|slippery)\b"),
]


EXPOSURE_PATTERNS = [
    ("Direct physical contact",
     r"\b(?:contact|touched|touches|reached|hit|hits|struck|splashed|splash|grazed|pierced|embedded|burned|injured)\b"),

    ("Line-of-fire exposure",
     r"\b(?:line[- ]of[- ]fire|trajectory|projected|projection|released|ejected|thrown|expelled|impacted|impacting)\b"),

    ("Caught-between / pinch exposure",
     r"\b(?:caught[- ]between|pinch|trapped|imprisoned|caught|crushed between|atr(?:i|t)ition|finger.*(?:caught|trapped|imprisoned)|hand.*(?:caught|trapped|imprisoned)|chest)\b"),

    ("Fall exposure",
     r"\b(?:fell|fall|falling|fall from|ladder|scaffold|height|reel|chimney|vertical opening)\b"),

    ("Chemical exposure",
     r"\b(?:acid|sulfuric|sulphuric|nitric|ammonia|chemical|caustic|corrosive|thinner).{0,100}\b(?:splash|splashes|spill|spilled|projection|projected|contact|face|eye|skin|body|lip|arm|leg)\b"),

    ("Thermal exposure",
     r"\b(?:hot gas|hot metal|molten|liquid metal|steam|heat|heated|hot pulp).{0,100}\b(?:face|eye|hand|arm|leg|worker|operator|person|contact|reach|reaching|projection|projected)\b"),

    ("Electrical exposure",
     r"\b(?:electrical|electric|energized|energised|live|arc|phase[- ]to[- ]ground|phase[- ]to[- ]phase|voltage).{0,100}\b(?:contact|shock|reach|reached|exposed|worker|operator|employee|hand|body|panel)\b"),

    ("Vehicle exposure",
     r"\b(?:vehicle|truck|loader|tanker|mobile equipment|reversing|reversed|backing).{0,120}\b(?:worker|operator|person|employee|hit|struck|collision|crash|occupant|cabin)\b"),
]


ACTUAL_CONSEQUENCE_PATTERNS = [
    ("Fatality", r"\b(?:fatal|fatality|death|died|dead|killed|verify the death)\b"),
    ("Electrical injury", r"\b(?:electric shock|electrical shock|electrocution|electrical injury|electric injury)\b"),
    ("Crushing / amputation", r"\b(?:crush(?:ed|ing)?|crushing|amputation|amputated|severed|atri(?:c|t)ion|trapped between|imprisoned between)\b"),
    ("Burn / thermal injury", r"\b(?:burn(?:ed|ing)?|burn injury|thermal injury|scald(?:ed)?|redness and burning)\b"),
    ("Chemical injury", r"\b(?:chemical burn|acid burn|corrosive burn|chemical injury)\b"),
    ("Cut / laceration", r"\b(?:cut|cuts|cutting|laceration|lacerated|wound|blunt cut|small cuts?|puncture|pierced)\b"),
    ("Struck-by / impact injury", r"\b(?:hit|struck|impact|impacted|grazed|contusion|bruise|trauma)\b"),
    ("Fall injury", r"\b(?:fall|fell|fallen|falling).{0,120}\b(?:injur|trauma|pain|fracture|swelling|wound|hurt|concussion)\b"),
    ("Eye / facial injury", r"\b(?:eye|face|facial).{0,80}\b(?:injur|irritat|pain|splash|contact|wound|burn|redness)\b"),
    ("Musculoskeletal injury", r"\b(?:sprain|strain|twist|twisted|lumbar pain|overexertion|musculoskeletal|faintness|dizziness)\b"),
    ("Minor irritation / swelling", r"\b(?:swelling|irritation|irritated|scratch|minor|superficial injury|little trauma|redness)\b"),
]


# Mechanism-aware potential consequence mapping.
# These are generated only when the narrative contains both the mechanism
# signal and a relevant human exposure/event signal.
POTENTIAL_RULES = [
    (
        "Potential severe electrical injury",
        r"\b(?:electrical|electric|energized|energised|live|phase[- ]to[- ]ground|phase[- ]to[- ]phase|440\s*v|400\s*a|electric arc|thermomagnetic)\b",
        r"\b(?:worker|operator|employee|person|mechanic|technician).{0,130}\b(?:contact|shock|arc|flash|reached|reach|exposed|injur)\b|\b(?:contact|shock|arc|flash|reached|exposed).{0,130}\b(?:worker|operator|employee|person|mechanic|technician)\b",
    ),
    (
        "Potential severe caught-between injury",
        r"\b(?:caught[- ]between|imprisoned|trapped|pinch|crushed between|atri(?:c|t)ion)\b",
        r"\b(?:hand|finger|arm|leg|chest|body|worker|mechanic|operator|employee)\b",
    ),
    (
        "Potential severe fall injury",
        r"\b(?:height|ladder|scaffold|reel|chimney|vertical opening|platform)\b",
        r"\b(?:fall|fell|falling|falls)\b.{0,100}\b(?:worker|operator|employee|person|man|collaborator)\b|\b(?:worker|operator|employee|person|man|collaborator).{0,100}\b(?:fall|fell|falling|falls)\b",
    ),
    (
        "Potential severe vehicle injury",
        r"\b(?:vehicle|truck|tanker|loader|mobile equipment|scooptram|mixer|tipper)\b",
        r"\b(?:crash|collision|roll|turn|reversing|reverse|slides|slide|excavated|cabin|occupant|struck)\b",
    ),
    (
        "Potential severe chemical injury",
        r"\b(?:sulfuric|sulphuric|nitric|ammonia|corrosive|chemical|acid)\b",
        r"\b(?:projection|projected|splash|splashes|face|eye|skin|upper limbs|reaches|reached)\b",
    ),
    (
        "Potential severe thermal injury",
        r"\b(?:hot gas|molten|liquid metal|steam|hot pulp|furnace|high temperature)\b",
        r"\b(?:reach|reaching|reached|projection|projected|face|worker|employee|operator)\b",
    ),
    (
        "Potential severe projectile injury",
        r"\b(?:splinter|fragment|rock block|rock slab|stone slab|metal fragment|projectile)\b",
        r"\b(?:release|released|project|projected|expell|impact|hit|struck|worker|employee|operator)\b",
    ),
    (
        "Potential severe entanglement / machine injury",
        r"\b(?:fan|propeller|conveyor|rotating|moving part|blade|machine)\b",
        r"\b(?:hand|arm|finger|clothing|rag|hooked|pulled|caught|worker|employee|operator)\b",
    ),
]


BARRIER_PRESENT = [
    ("PPE",
     r"\b(?:helmet|hard hat|gloves?|goggles?|safety glasses|face shield|faceshield|respirator|mask|safety shoes|boots|ppe|epps)\b"),

    ("Guard / physical protection",
     r"\b(?:guard|guarding|protector|protective guard|cover|shield|barrier|screen|enclosure|fence|fender|cabin)\b"),

    ("Isolation / LOTO",
     r"\b(?:lockout|lock[- ]out|tagout|tag[- ]out|loto|isolation|isolated|isolate|disabled)\b"),

    ("De-energization",
     r"\b(?:de[- ]energized|deenergized|power off|switched off|shut down|shutdown|disconnected|breaker off)\b"),

    ("Fall protection",
     r"\b(?:harness|fall arrest|lifeline|lanyard|guardrail|handrail|toe board)\b"),

    ("Exclusion / traffic control",
     r"\b(?:barricade|barricaded|exclusion zone|restricted area|spotter|banksman|flagman|traffic control|cordoned|fenced area)\b"),

    ("Procedure / JSA / risk assessment",
     r"\b(?:procedure|method statement|job safety analysis|jsa|jha|risk assessment|checklist|instruction)\b"),

    ("Emergency / detection control",
     r"\b(?:storm detector|detector|alarm|emergency response|evacuation|refuge)\b"),
]


# Explicit absence/failure patterns. Do not infer absence merely because a
# control is not mentioned.
BARRIER_GAPS = [
    ("PPE",
     r"\b(?:without|no|not wearing|did not have|did not use|not using|lack of|missing)\b.{0,55}\b(?:helmet|gloves?|goggles?|safety glasses|face shield|mask|respirator|ppe|epps)\b"),

    ("Guard / physical protection",
     r"\b(?:without|no|missing|removed|not in place|bypassed)\b.{0,65}\b(?:guard|guarding|protector|cover|shield|barrier|enclosure)\b"),

    ("Isolation / LOTO",
     r"\b(?:without|no|not|failed|failure|missing|not isolated|not lock(?:ed)? out|not tagged)\b.{0,75}\b(?:isolation|isolat(?:ed|ion)|lockout|lock[- ]out|tagout|tag[- ]out|loto)\b"),

    ("De-energization",
     r"\b(?:still energized|remained energized|not de[- ]energized|without shutting down|without switching off|without disconnecting|power remained on)\b"),

    ("Fall protection / safe access",
     r"\b(?:unstable reel|rim position|improvised|without.*(?:ladder|platform)|unstable.*(?:ladder|platform))\b"),

    ("Exclusion / line-of-fire control",
     r"\b(?:in the line of fire|line[- ]of[- ]fire|not blocked|not barricaded|not cordoned|person.*inside.*(?:restricted|fenced)|remain(?:ed)? in the exposure area)\b"),

    ("Structural / ground control",
     r"\b(?:broken.*(?:anchor|support|rivet)|rock block displacement|stone slab.*detached|unstable ground|loose.*ground|roof.*(?:detach|fall))\b"),

    ("Machine energy isolation",
     r"\b(?:bonnet open).{0,80}\b(?:functioning|running|operating)\b"),
]


def make_items(text: str, bank, confidence: float):
    return [
        {
            **item,
            "confidence": confidence,
        }
        for item in dedup(regex_hits(text, bank))
    ]


def extract_barriers(text: str):
    out = []

    for label, pattern in BARRIER_GAPS:
        m = re.search(pattern, text.lower(), flags=re.I)
        if m:
            start = max(0, m.start() - 70)
            end = min(len(text), m.end() + 70)
            out.append({
                "label": label,
                "status": "failed_or_absent",
                "evidence": text[m.start():m.end()].strip(),
                "context": norm(text[start:end]),
                "confidence": 0.90,
            })

    for label, pattern in BARRIER_PRESENT:
        m = re.search(pattern, text.lower(), flags=re.I)
        if m:
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 60)
            # If a same-category gap was already identified, retain both
            # statuses as separate pieces of evidence.
            out.append({
                "label": label,
                "status": "present",
                "evidence": text[m.start():m.end()].strip(),
                "context": norm(text[start:end]),
                "confidence": 0.80,
            })

    # Deduplicate by (label, status).
    final = {}
    for x in out:
        key = (x["label"].lower(), x["status"])
        final[key] = x
    return list(final.values())


def extract_potential(text: str):
    out = []
    low = text.lower()

    for label, mechanism_pattern, exposure_pattern in POTENTIAL_RULES:
        mech = re.search(mechanism_pattern, low, flags=re.I)
        if not mech:
            continue

        exp = re.search(exposure_pattern, low, flags=re.I)
        if not exp:
            continue

        start = max(0, min(mech.start(), exp.start()) - 50)
        end = min(len(text), max(mech.end(), exp.end()) + 90)

        out.append({
            "label": label,
            "evidence": norm(text[start:end]),
            "context": norm(text[start:end]),
            "confidence": 0.74,
        })

    return dedup(out)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required = {"report_id", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    flat = []
    detailed = []

    for _, row in df.iterrows():
        rid = clean(row["report_id"])
        text = norm(row["description"])

        hazards = make_items(text, HAZARD_PATTERNS, 0.86)
        exposures = make_items(text, EXPOSURE_PATTERNS, 0.82)
        actual = make_items(text, ACTUAL_CONSEQUENCE_PATTERNS, 0.82)
        potential = extract_potential(text)
        barriers = extract_barriers(text)
        activities = make_items(text, [
            ("Inspection", r"\b(?:inspection|inspecting|inspect|checking|check|survey)\b"),
            ("Assembly / installation", r"\b(?:assembly|assembling|installation|installing|installed|mounting)\b"),
            ("Maintenance / repair", r"\b(?:maintenance|repair|repairing|overhaul|servicing|service)\b"),
            ("Adjustment / tightening", r"\b(?:adjustment|adjusting|tightening|loosen(?:ing)?)\b"),
            ("Cleaning", r"\b(?:cleaning|clean|washing|washing down)\b"),
            ("Painting / coating", r"\b(?:painting|paint|coating)\b"),
            ("Lifting / handling", r"\b(?:lifting|lifted|rigging|handling|carrying)\b"),
            ("Working at height", r"\b(?:working at height|ladder|scaffold|roof|rooftop|elevated|reel)\b"),
            ("Driving / vehicle movement", r"\b(?:driving|reversing|reversed|backing|vehicle movement|transit)\b"),
            ("Cutting / drilling / grinding", r"\b(?:cutting|cutter|drilling|drill|grinding|machete)\b"),
            ("Process operation", r"\b(?:furnace|autoclave|hopper|conveyor|pump|tank|chute|sampling)\b"),
        ], 0.83)

        flat.append({
            "report_id": rid,

            "hazards": "; ".join(x["label"] for x in hazards),
            "hazard_evidence": "; ".join(x["evidence"] for x in hazards),

            "exposures": "; ".join(x["label"] for x in exposures),
            "exposure_evidence": "; ".join(x["evidence"] for x in exposures),

            "actual_consequences": "; ".join(x["label"] for x in actual),
            "actual_consequence_evidence": "; ".join(
                x["evidence"] for x in actual
            ),

            "potential_consequences": "; ".join(x["label"] for x in potential),
            "potential_consequence_evidence": "; ".join(
                x["evidence"] for x in potential
            ),

            "observed_barriers": "; ".join(
                f'{x["label"]} [{x["status"]}]' for x in barriers
            ),
            "barrier_evidence": "; ".join(x["evidence"] for x in barriers),

            "activities": "; ".join(x["label"] for x in activities),
            "activity_evidence": "; ".join(x["evidence"] for x in activities),

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

    out = pd.DataFrame(flat)

    csv_path = OUTPUT_DIR / "safety_extraction_flat.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    jsonl_path = OUTPUT_DIR / "safety_extraction_detailed.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in detailed:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def coverage(col):
        return round(
            100 * out[col].fillna("").astype(str).str.strip().ne("").mean(), 2
        )

    summary = {
        "version": "0.6",
        "records": int(len(out)),
        "coverage_percent": {
            "hazards": coverage("hazards"),
            "exposures": coverage("exposures"),
            "actual_consequences": coverage("actual_consequences"),
            "potential_consequences": coverage("potential_consequences"),
            "barriers": coverage("observed_barriers"),
            "activities": coverage("activities"),
        },
        "deduplication": {
            "canonical_labels": True,
            "duplicate_labels_per_record_allowed": False,
        },
        "principles": [
            "Actual consequence is narrative-observed only.",
            "Potential consequence requires a mechanism plus human-exposure signal.",
            "Barrier absence is emitted only when the narrative contains explicit gap language or a strong physical-state signal.",
            "Human annotation notes are never used as extractor inputs.",
        ],
    }

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 78)
    print("RAKSHAK SAFETY INFORMATION EXTRACTOR v0.6")
    print("=" * 78)
    print(f"Records : {len(out)}")
    print()
    for key, value in summary["coverage_percent"].items():
        print(f"{key:<32}: {value:5.1f}%")
    print()
    print("FILES")
    print(f"Flat CSV : {csv_path}")
    print(f"Detailed : {jsonl_path}")
    print(f"Summary  : {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
