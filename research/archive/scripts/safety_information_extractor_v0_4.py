"""
RAKSHAK Safety Information Extractor v0.4
=========================================

Rule-based, evidence-first safety ontology extractor for short industrial
incident / unsafe-act / near-miss narratives.

Design goals:
- Increase recall without blindly firing on ambiguous keywords.
- Emit normalized ontology labels plus exact evidence snippets.
- Treat hazard, exposure, consequence, barrier, and activity as separate
  concepts.
- Support barrier status: present / failed_or_absent / unclear.
- Avoid using annotation columns or target labels.

Input:
    data/annotations/resolved_annotations_v0.2.csv
Expected:
    report_id
    description
    (optional) activity / location / site metadata

Output:
    experiments/safety_information_extraction_v0.4/
        safety_extraction_flat.csv
        safety_extraction_detailed.jsonl
        summary.json

This is a diagnostic/engineering baseline, not a production safety decision
system.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "annotations" / "resolved_annotations_v0.2.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "safety_information_extraction_v0.4"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clean_text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def find_first_context(text: str, patterns: List[str], window: int = 90) -> Optional[Tuple[str, str]]:
    """
    Return (matched evidence, context) for the first regex hit.
    """
    low = text.lower()
    for pattern in patterns:
        m = re.search(pattern, low, flags=re.I)
        if m:
            start = max(0, m.start() - window)
            end = min(len(text), m.end() + window)
            return text[m.start():m.end()], normalize_ws(text[start:end])
    return None


def collect_hits(
    text: str,
    patterns: List[Tuple[str, str]],
    window: int = 80,
    max_hits: int = 4,
) -> List[Tuple[str, str, str]]:
    """
    patterns = [(label, regex), ...]
    Returns unique (label, evidence, context).
    """
    out: List[Tuple[str, str, str]] = []
    low = text.lower()

    for label, pattern in patterns:
        for m in re.finditer(pattern, low, flags=re.I):
            evidence = text[m.start():m.end()].strip()
            start = max(0, m.start() - window)
            end = min(len(text), m.end() + window)
            context = normalize_ws(text[start:end])

            key = (label.lower(), evidence.lower())
            if any((x[0].lower(), x[1].lower()) == key for x in out):
                continue

            out.append((label, evidence, context))
            if len(out) >= max_hits:
                return out

    return out


def join_unique(items: List[str], sep: str = "; ") -> str:
    seen = set()
    out = []
    for x in items:
        x = clean_text(x)
        if not x:
            continue
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return sep.join(out)


# ---------------------------------------------------------------------------
# Ontology phrase banks
# ---------------------------------------------------------------------------

HAZARD_PATTERNS: List[Tuple[str, str]] = [
    ("Electrical energy", r"\b(?:\d{2,4}\s*v(?:olt)?s?|\d{2,5}\s*a(?:mp)?s?|electrical|electric|energized?|live|phase[- ]to[- ]phase|phase[- ]to[- ]ground|short[- ]circuit|arc(?:ing)?|thermomagnetic)\b"),
    ("Mechanical / moving equipment energy", r"\b(?:blade|cutter|cutting machine|fan|conveyor|rotating|rotation|moving equipment|moving part|mechanical|shaft|roller|belt|pulley|gear|motor)\b"),
    ("Pressure / stored pressure", r"\b(?:pressure|pressurized|compressed air|steam|hydraulic|pneumatic|pressure line|pressurized line)\b"),
    ("Chemical energy / hazardous substance", r"\b(?:acid|sulfuric|sulphuric|nitric|hydrochloric|caustic|ammonia|chemical|solvent|chlorine|toxic|corrosive|alkali|chemical solution)\b"),
    ("Thermal energy", r"\b(?:hot gas|hot metal|hot liquid|molten|liquid metal|heat|heated|steam|high temperature|thermal|burning|scald)\b"),
    ("Fire / explosion energy", r"\b(?:fire|flame|explosion|explosive|ignition|flash fire|flash|combustible|flammable)\b"),
    ("Vehicle / mobile equipment energy", r"\b(?:vehicle|truck|car|bus|forklift|loader|excavator|crane|mobile equipment|reversing|reversed|backing|moving vehicle)\b"),
    ("Falling-object / dropped-object energy", r"\b(?:falling object|dropped object|dropped load|rock block|rock slab|stone|metal fragment|fragment|splinter|falling material|fall(?:en|ing) from)\b"),
    ("Fall-from-height energy", r"\b(?:fall from height|fell from|falling from|ladder|scaffold|scaffolding|rooftop|roof|platform|elevated|height|above ground)\b"),
    ("Caught-between / pinch energy", r"\b(?:caught[- ]between|pinch|trapped|imprisoned|caught|crushed between|finger.*(?:caught|trapped|imprisoned)|hand.*(?:caught|trapped|imprisoned))\b"),
    ("Manual handling / ergonomic energy", r"\b(?:lifting|manual handling|carrying|heavy object|awkward posture|reaching|pulling|pushing|twisting)\b"),
    ("Biological / animal exposure", r"\b(?:bite|snake|animal|insect|bee|wasp|dog)\b"),
    ("Sharp-object energy", r"\b(?:knife|sharp edge|sharp object|broken glass|glass shard|blade|metal edge)\b"),
]

EXPOSURE_PATTERNS: List[Tuple[str, str]] = [
    ("Direct worker exposure", r"\b(?:worker|operator|mechanic|employee|collaborator|person|technician|man)\b.{0,70}\b(?:reached|exposed|contact|hit|struck|splashed|touched|injured|injury|burn|cut|graz(?:ed|ing))\b"),
    ("Electrical exposure", r"\b(?:touch(?:ed|ing)?|contact|reached|phase[- ]to[- ](?:phase|ground)|energized?|live).{0,70}\b(?:electrical|electric|panel|wire|cable|conductor|voltage|volt|amp)\b|\b(?:electrical|electric).{0,70}\b(?:touch|contact|shock|reached|exposed)\b"),
    ("Chemical exposure", r"\b(?:acid|sulfuric|sulphuric|nitric|ammonia|chemical|solvent|caustic|corrosive).{0,100}\b(?:splash(?:ed)?|spill(?:ed)?|drop(?:ped)?|contact|touched|reached|face|eye|skin|body|upper limbs?)\b"),
    ("Thermal exposure", r"\b(?:hot gas|hot metal|molten|liquid metal|steam|heat|heated).{0,90}\b(?:face|eye|hand|arm|worker|operator|person|contact|reached|splashed|project(?:ed|ion)?)\b"),
    ("Line-of-fire exposure", r"\b(?:line[- ]of[- ]fire|in the trajectory|trajectory|projection|projected|released|thrown|ejected|struck|hit)\b"),
    ("Caught-between / pinch exposure", r"\b(?:caught[- ]between|pinch|trapped|imprisoned|caught|crushed between|finger.*(?:caught|trapped|imprisoned)|hand.*(?:caught|trapped|imprisoned))\b"),
    ("Fall exposure", r"\b(?:fell|fall(?:ing)?|fall from|ladder|scaffold|height|roof|platform).{0,100}\b(?:worker|operator|person|employee|mechanic|technician)\b"),
    ("Body part contact", r"\b(?:finger|hand|arm|forearm|face|eye|head|leg|foot|right hand|left hand|upper limb|body part)\b.{0,75}\b(?:contact|graz(?:ed|ing)|hit|struck|cut|injur|touch|embedded|imprisoned|trapped)\b"),
    ("Vehicle exposure", r"\b(?:reversing|reversed|backing|vehicle|truck|forklift|loader|car|mobile equipment).{0,100}\b(?:worker|operator|person|employee|mechanic|technician|hit|struck|near|contact)\b"),
]

CONSEQUENCE_PATTERNS: List[Tuple[str, str]] = [
    ("Fatality", r"\b(?:fatal|fatality|death|died|dead|killed|loss of life)\b"),
    ("Electrical injury", r"\b(?:electric shock|electrical shock|electrocution|electrical injury|electric injury|shock)\b"),
    ("Crushing / amputation", r"\b(?:crush(?:ed|ing)?|crushing|amputation|amputat(?:ed|ion)|severed|imprisoned|trapped between|crushed between)\b"),
    ("Burn / thermal injury", r"\b(?:burn(?:ed)?|burn injury|thermal injury|scald(?:ed)?|heat injury)\b"),
    ("Chemical injury", r"\b(?:chemical burn|acid burn|corrosive burn|chemical injury|burn(?:ed)?).{0,80}\b(?:acid|chemical|sulfuric|sulphuric|nitric|ammonia|caustic|corrosive)\b|\b(?:acid|sulfuric|sulphuric|nitric|ammonia|chemical).{0,80}\b(?:burn|injur)\b"),
    ("Cut / laceration", r"\b(?:cut|cuts|cutting|laceration|lacerated|incision|wound|blunt cut|small cuts?)\b"),
    ("Struck-by injury", r"\b(?:hit|struck|struck-by|impact|impacted|graz(?:ed|ing)|blunt trauma)\b"),
    ("Fall injury", r"\b(?:fell|fall(?:en)?|fall injury|fall from|falling).{0,100}\b(?:injur|trauma|pain|fracture|swelling|wound|hurt)\b"),
    ("Eye / facial injury", r"\b(?:eye|face|facial).{0,70}\b(?:injur|irritat|pain|hit|splash|contact|wound|burn)\b"),
    ("Respiratory / inhalation injury", r"\b(?:inhalation|inhaled|fumes|gas|smoke).{0,80}\b(?:injur|irritat|expos|breath|respirat)\b"),
    ("Serious / potentially serious injury", r"\b(?:serious injury|severe injury|seriously injured|potentially serious|hospitali[sz]ed|fracture|amputation|loss of consciousness|unconscious)\b"),
    ("Minor injury / irritation", r"\b(?:minor injury|minor|superficial injury|little trauma|small cuts?|swelling|irritation|irritated|scratch(?:ed)?|bruise)\b"),
]

# Barrier patterns are deliberately consequence/context aware.
BARRIER_PATTERNS: List[Tuple[str, str, str]] = [
    ("PPE", r"\b(?:helmet|hard hat|gloves?|goggles?|safety glasses|face shield|faceshield|respirator|mask|earplugs?|ear protection|safety shoes|boots|ppe|personal protective equipment)\b", "present"),
    ("Guard / physical protection", r"\b(?:guard|guarding|protector|protective guard|cover|shield|barrier|screen|enclosure|fence)\b", "present"),
    ("Isolation / LOTO", r"\b(?:lockout|lock[- ]out|tagout|tag[- ]out|loto|isolation|isolated|isolate|electrically isolated)\b", "present"),
    ("De-energization", r"\b(?:de[- ]energized|deenergized|power off|switched off|shut down|shutdown|disconnected|breaker off|breaker disconnected)\b", "present"),
    ("Fall protection", r"\b(?:harness|fall arrest|lifeline|lanyard|guardrail|handrail|toe board)\b", "present"),
    ("Traffic control / exclusion zone", r"\b(?:barricade|barricaded|exclusion zone|restricted area|spotter|banksman|flagman|traffic control|cordoned off)\b", "present"),
    ("Permit / authorization", r"\b(?:permit to work|work permit|permit|authorized|authori[sz]ed)\b", "present"),
    ("Procedure / JSA", r"\b(?:procedure|safe work method|method statement|job safety analysis|jsa|jha|risk assessment|checklist|instruction)\b", "present"),
    ("Supervision", r"\b(?:supervisor|supervision|supervised|foreman)\b", "present"),
]

BARRIER_FAILURE_PATTERNS: List[Tuple[str, str]] = [
    ("PPE", r"\b(?:without|no|not wearing|failed to wear|missing|lack of)\b.{0,40}\b(?:helmet|gloves?|goggles?|safety glasses|face shield|mask|respirator|ppe|personal protective equipment)\b"),
    ("Guard / physical protection", r"\b(?:without|no|missing|removed|removed the|failed|failure|not in place|bypassed|bypass(?:ed)?)\b.{0,50}\b(?:guard|guarding|protector|cover|shield|barrier|enclosure)\b"),
    ("Isolation / LOTO", r"\b(?:without|no|not|failed|failure|missing|not isolated|not lock(?:ed)? out|not tagged)\b.{0,60}\b(?:isolation|isolat(?:ed|ion)|lockout|lock[- ]out|tagout|tag[- ]out|loto)\b"),
    ("De-energization", r"\b(?:still energized|remained energized|not de[- ]energized|without shutting down|without switching off|without disconnecting|power remained on)\b"),
    ("Fall protection", r"\b(?:without|no|missing|not using|not wearing|failed to use)\b.{0,50}\b(?:harness|fall arrest|lifeline|lanyard|guardrail|handrail)\b"),
    ("Traffic control / exclusion zone", r"\b(?:without|no|missing|not established|not barricaded|not cordoned)\b.{0,60}\b(?:exclusion zone|barricade|barricaded|restricted area|spotter|banksman|traffic control)\b"),
]

ACTIVITY_PATTERNS: List[Tuple[str, str]] = [
    ("Assembly / installation", r"\b(?:assembly|assembling|installation|installing|installed|mounting)\b"),
    ("Adjustment / tightening", r"\b(?:adjustment|adjusting|tightening|loosen(?:ing)?|setting)\b"),
    ("Maintenance / repair", r"\b(?:maintenance|maintaining|repair|repairing|overhaul|servicing|service)\b"),
    ("Inspection", r"\b(?:inspection|inspecting|inspect|checking|check)\b"),
    ("Cleaning", r"\b(?:cleaning|clean|washing|washing down)\b"),
    ("Painting / coating", r"\b(?:painting|paint|coating|coated)\b"),
    ("Lifting / handling", r"\b(?:lifting|lifted|lifting operation|rigging|handling|carrying)\b"),
    ("Excavation / earthwork", r"\b(?:excavation|excavating|trenching|earthwork|digging)\b"),
    ("Working at height", r"\b(?:working at height|ladder|scaffold|scaffolding|roof|rooftop|elevated platform)\b"),
    ("Driving / vehicle movement", r"\b(?:driving|reversing|reversed|backing|vehicle movement|moving vehicle)\b"),
    ("Drilling / cutting", r"\b(?:drilling|drill|cutting|cutter|sawing|grinding)\b"),
    ("Electrical work", r"\b(?:electrical work|electrical installation|wiring|panel work|testing electrical)\b"),
]

# ---------------------------------------------------------------------------
# Evidence logic
# ---------------------------------------------------------------------------

def extract_hazards(text: str) -> List[dict]:
    hits = collect_hits(text, HAZARD_PATTERNS, window=75, max_hits=8)

    # Guard ambiguous "flash": only accept as fire/explosion when context supports it.
    cleaned = []
    for label, evidence, context in hits:
        if label == "Fire / explosion energy" and evidence.lower() == "flash":
            ctx = context.lower()
            if not any(x in ctx for x in ["fire", "flame", "ignition", "explosion", "combust", "gas"]):
                continue
        cleaned.append({
            "label": label,
            "evidence": evidence,
            "context": context,
            "confidence": 0.85 if label != "Manual handling / ergonomic energy" else 0.70,
        })

    return cleaned


def extract_exposures(text: str, hazards: List[dict]) -> List[dict]:
    hits = collect_hits(text, [(a, b) for a, b in EXPOSURE_PATTERNS], window=80, max_hits=8)

    out = [{
        "label": label,
        "evidence": evidence,
        "context": context,
        "confidence": 0.82,
    } for label, evidence, context in hits]

    low = text.lower()

    # Explicit injury/contact often implies actual worker exposure, but only
    # when a human action/body part/worker reference exists nearby.
    injury_terms = r"(injur|hurt|pain|burn|cut|graz|hit|struck|splashed|contact|touched|embedded)"
    human_terms = r"(worker|operator|mechanic|employee|person|collaborator|technician|man|finger|hand|arm|face|eye|leg|foot)"
    if re.search(human_terms + r".{0,80}" + injury_terms, low) or re.search(injury_terms + r".{0,80}" + human_terms, low):
        if not any(x["label"] == "Direct worker exposure" for x in out):
            m = re.search(human_terms + r".{0,80}" + injury_terms, low)
            if m:
                snippet = text[max(0, m.start()-20):min(len(text), m.end()+20)]
                out.append({
                    "label": "Direct worker exposure",
                    "evidence": normalize_ws(m.group(0)),
                    "context": normalize_ws(snippet),
                    "confidence": 0.74,
                })

    # If an explicit caught-between statement exists, exposure is highly specific.
    if re.search(r"\b(?:caught[- ]between|imprisoned between|crushed between|finger.*(?:caught|trapped|imprisoned)|hand.*(?:caught|trapped|imprisoned))\b", low):
        if not any(x["label"] == "Caught-between / pinch exposure" for x in out):
            m = re.search(r"\b(?:caught[- ]between|imprisoned between|crushed between|finger.*?(?:caught|trapped|imprisoned)|hand.*?(?:caught|trapped|imprisoned))\b", low)
            if m:
                out.append({
                    "label": "Caught-between / pinch exposure",
                    "evidence": text[m.start():m.end()],
                    "context": normalize_ws(text[max(0,m.start()-50):min(len(text),m.end()+50)]),
                    "confidence": 0.95,
                })

    return out


def extract_consequences(text: str) -> List[dict]:
    hits = collect_hits(text, [(a, b) for a, b in CONSEQUENCE_PATTERNS], window=75, max_hits=10)
    out = []

    # Severity-specific post-filter: don't call generic "burn" chemical/thermal
    # without substance or heat context.
    for label, evidence, context in hits:
        c = context.lower()
        if label == "Chemical injury" and not re.search(r"acid|chemical|sulfuric|sulphuric|nitric|ammonia|caustic|corrosive", c):
            continue
        if label == "Burn / thermal injury" and not re.search(r"hot|heat|thermal|steam|molten|metal|gas|burn", c):
            continue
        out.append({
            "label": label,
            "evidence": evidence,
            "context": context,
            "confidence": 0.84,
        })

    return out


def extract_barriers(text: str) -> List[dict]:
    low = text.lower()
    out: List[dict] = []
    seen = set()

    # Failure/absence rules first so status is preserved.
    for label, pattern in BARRIER_FAILURE_PATTERNS:
        m = re.search(pattern, low, flags=re.I)
        if m:
            key = label.lower()
            if key not in seen:
                snippet = text[max(0, m.start()-30):min(len(text), m.end()+30)]
                out.append({
                    "label": label,
                    "status": "failed_or_absent",
                    "evidence": normalize_ws(m.group(0)),
                    "context": normalize_ws(snippet),
                    "confidence": 0.90,
                })
                seen.add(key)

    # Presence rules. When a barrier is explicitly mentioned with a failure
    # signal nearby, do not overwrite the failure status.
    for label, pattern, default_status in BARRIER_PATTERNS:
        for m in re.finditer(pattern, low, flags=re.I):
            key = label.lower()
            if key in seen:
                continue

            # Nearby negation/failure context.
            nearby = low[max(0, m.start()-70):min(len(low), m.end()+70)]
            failed = bool(re.search(
                r"\b(?:without|no|not|missing|removed|failed|failure|lack|bypassed|bypass)\b",
                nearby,
            ))
            status = "failed_or_absent" if failed else default_status

            snippet = text[max(0, m.start()-45):min(len(text), m.end()+45)]
            out.append({
                "label": label,
                "status": status,
                "evidence": text[m.start():m.end()],
                "context": normalize_ws(snippet),
                "confidence": 0.87 if failed else 0.78,
            })
            seen.add(key)
            break

    return out


def extract_activities(text: str) -> List[dict]:
    hits = collect_hits(text, ACTIVITY_PATTERNS, window=70, max_hits=5)
    return [{
        "label": label,
        "evidence": evidence,
        "context": context,
        "confidence": 0.84,
    } for label, evidence, context in hits]


# ---------------------------------------------------------------------------
# Derived SIF intelligence signals
# ---------------------------------------------------------------------------

def derived_signals(
    text: str,
    hazards: List[dict],
    exposures: List[dict],
    consequences: List[dict],
    barriers: List[dict],
) -> Dict[str, object]:
    low = text.lower()

    fatality_evidence = bool(re.search(
        r"\b(?:fatal|fatality|death|died|dead|killed|loss of life)\b", low
    ))

    caught_between = any(
        x["label"] == "Caught-between / pinch exposure" for x in exposures
    ) or bool(re.search(r"\b(?:caught[- ]between|imprisoned between|crushed between)\b", low))

    direct_exposure = any(
        x["label"] in {
            "Direct worker exposure",
            "Electrical exposure",
            "Chemical exposure",
            "Thermal exposure",
            "Line-of-fire exposure",
            "Caught-between / pinch exposure",
            "Vehicle exposure",
            "Fall exposure",
        } for x in exposures
    )

    severe = any(
        x["label"] in {
            "Fatality",
            "Crushing / amputation",
            "Electrical injury",
            "Serious / potentially serious injury",
        } for x in consequences
    )

    # Stronger SIF precursor signals. This is not the SIF classifier itself.
    high_energy = any(
        x["label"] in {
            "Electrical energy",
            "Pressure / stored pressure",
            "Thermal energy",
            "Fire / explosion energy",
            "Vehicle / mobile equipment energy",
            "Falling-object / dropped-object energy",
            "Fall-from-height energy",
            "Caught-between / pinch energy",
        } for x in hazards
    )

    barrier_failure = any(
        x["status"] == "failed_or_absent" for x in barriers
    )

    sif_signal_score = 0.0
    sif_signal_score += 0.30 if high_energy else 0
    sif_signal_score += 0.25 if direct_exposure else 0
    sif_signal_score += 0.20 if caught_between else 0
    sif_signal_score += 0.25 if severe or fatality_evidence else 0
    sif_signal_score += 0.10 if barrier_failure else 0
    sif_signal_score = min(1.0, sif_signal_score)

    return {
        "has_fatality_evidence": fatality_evidence,
        "has_caught_between_evidence": caught_between,
        "has_direct_exposure_evidence": direct_exposure,
        "has_severe_consequence_signal": severe,
        "has_high_energy_signal": high_energy,
        "has_barrier_failure_signal": barrier_failure,
        "sif_precursor_signal_score": round(sif_signal_score, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required = {"report_id", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required input columns: {sorted(missing)}")

    flat_rows: List[dict] = []
    detailed: List[dict] = []

    for _, row in df.iterrows():
        report_id = clean_text(row.get("report_id"))
        text = normalize_ws(clean_text(row.get("description")))

        hazards = extract_hazards(text)
        exposures = extract_exposures(text, hazards)
        consequences = extract_consequences(text)
        barriers = extract_barriers(text)
        activities = extract_activities(text)
        signals = derived_signals(text, hazards, exposures, consequences, barriers)

        # Preserve concise flat schema for downstream analytics.
        flat_rows.append({
            "report_id": report_id,
            "hazards": join_unique([x["label"] for x in hazards]),
            "hazard_evidence": join_unique([x["evidence"] for x in hazards]),
            "exposures": join_unique([x["label"] for x in exposures]),
            "exposure_evidence": join_unique([x["evidence"] for x in exposures]),
            "consequences": join_unique([x["label"] for x in consequences]),
            "consequence_evidence": join_unique([x["evidence"] for x in consequences]),
            "barriers": join_unique([
                f'{x["label"]} [{x["status"]}]' for x in barriers
            ]),
            "barrier_evidence": join_unique([x["evidence"] for x in barriers]),
            "activities": join_unique([x["label"] for x in activities]),
            "activity_evidence": join_unique([x["evidence"] for x in activities]),
            "hazard_count": len(hazards),
            "exposure_count": len(exposures),
            "consequence_count": len(consequences),
            "barrier_count": len(barriers),
            "activity_count": len(activities),
            **signals,
        })

        detailed.append({
            "report_id": report_id,
            "description": text,
            "hazards": hazards,
            "exposures": exposures,
            "consequences": consequences,
            "barriers": barriers,
            "activities": activities,
            "signals": signals,
        })

    flat_df = pd.DataFrame(flat_rows)
    flat_path = OUTPUT_DIR / "safety_extraction_flat.csv"
    flat_df.to_csv(flat_path, index=False, encoding="utf-8-sig")

    jsonl_path = OUTPUT_DIR / "safety_extraction_detailed.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in detailed:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def pct_nonempty(col: str) -> float:
        if len(flat_df) == 0:
            return 0.0
        return round(
            100.0 * flat_df[col].fillna("").astype(str).str.strip().ne("").mean(), 2
        )

    summary = {
        "version": "0.4",
        "records": int(len(flat_df)),
        "input": str(INPUT_PATH),
        "output_dir": str(OUTPUT_DIR),
        "coverage_percent": {
            "hazards": pct_nonempty("hazards"),
            "exposures": pct_nonempty("exposures"),
            "consequences": pct_nonempty("consequences"),
            "barriers": pct_nonempty("barriers"),
            "activities": pct_nonempty("activities"),
        },
        "mean_counts": {
            "hazards": round(float(flat_df["hazard_count"].mean()), 3),
            "exposures": round(float(flat_df["exposure_count"].mean()), 3),
            "consequences": round(float(flat_df["consequence_count"].mean()), 3),
            "barriers": round(float(flat_df["barrier_count"].mean()), 3),
            "activities": round(float(flat_df["activity_count"].mean()), 3),
        },
        "derived_signal_counts": {
            col: int(flat_df[col].sum())
            for col in [
                "has_fatality_evidence",
                "has_caught_between_evidence",
                "has_direct_exposure_evidence",
                "has_severe_consequence_signal",
                "has_high_energy_signal",
                "has_barrier_failure_signal",
            ]
        },
        "notes": [
            "Rule-based diagnostic extractor only.",
            "Does not consume human annotation fields.",
            "SIF precursor signal score is a derived feature, not the final SIF classifier.",
        ],
    }

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 78)
    print("RAKSHAK SAFETY INFORMATION EXTRACTOR v0.4")
    print("=" * 78)
    print(f"Records : {len(flat_df)}")
    print()
    print("FIELD COVERAGE")
    for key, value in summary["coverage_percent"].items():
        print(f"  {key:<15}: {value:5.1f}%")
    print()
    print("MEAN ITEMS / RECORD")
    for key, value in summary["mean_counts"].items():
        print(f"  {key:<15}: {value:5.2f}")
    print()
    print("DERIVED SIGNALS")
    for key, value in summary["derived_signal_counts"].items():
        print(f"  {key:<35}: {value}")
    print()
    print("FILES")
    print(f"  Flat CSV : {flat_path}")
    print(f"  Detailed : {jsonl_path}")
    print(f"  Summary  : {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
