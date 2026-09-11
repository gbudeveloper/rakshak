"""
RAKSHAK SIF Precursor Reasoning Engine v0.2
==========================================

Purpose
-------
Improves v0.1 by separating:
  1. mechanism evidence
  2. human exposure evidence
  3. severity evidence
  4. barrier/control evidence

The engine is deliberately conservative:
- Generic contact, cut, minor fall, or generic chemical splash does not
  automatically become a SIF precursor.
- High-energy mechanism alone is insufficient.
- Potential severe consequence must be tied to a concrete mechanism/exposure.
- Observed barriers are evidence, but "not mentioned" is not itself a failure.
- No human SIF labels, potential accident level, or annotation notes are used.

This is still a rule-based diagnostic layer, not a calibrated classifier.

Inputs
------
experiments/safety_information_extraction_v0.6/safety_extraction_flat.csv

Outputs
-------
experiments/sif_reasoning_v0.2/
  sif_reasoning.csv
  sif_reasoning_detailed.jsonl
  summary.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "safety_information_extraction_v0.6"
    / "safety_extraction_flat.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "sif_reasoning_v0.2"


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


def low_words(text: str) -> str:
    return re.sub(r"\s+", " ", clean(text).lower())


def contains_any(text: str, terms: List[str]) -> bool:
    low = low_words(text)
    return any(t.lower() in low for t in terms)


def has_exact_label(items: List[str], wanted: str) -> bool:
    return wanted in items


# ---------------------------------------------------------------------------
# LSR candidate mapping
# ---------------------------------------------------------------------------

LSR_RULES = {
    "Work at Height": {
        "hazards": {"Fall-from-height energy"},
        "exposures": {"Fall exposure"},
        "activities": {"Working at height"},
        "keywords": ["ladder", "scaffold", "height", "reel", "chimney", "vertical opening"],
    },
    "Isolation": {
        "hazards": {
            "Electrical energy",
            "Mechanical / moving equipment energy",
            "Pressure / stored energy",
        },
        "keywords": [
            "energized", "energised", "live", "lockout", "loto",
            "isolation", "isolated", "de-energized", "deenergized",
            "bonnet open", "operating",
        ],
    },
    "Line of Fire": {
        "hazards": {
            "Falling object / dropped object energy",
            "Vehicle / mobile equipment energy",
            "Pressure / stored energy",
            "Thermal energy",
            "Fire / explosion energy",
            "Mechanical / moving equipment energy",
        },
        "exposures": {"Line-of-fire exposure"},
        "keywords": [
            "projection", "projected", "line of fire", "trajectory",
            "released", "ejected", "expelled", "struck", "impact",
        ],
    },
    "Seat Belt": {
        "hazards": {"Vehicle / mobile equipment energy"},
        "keywords": ["seat belt", "safety belt"],
    },
    "Driving": {
        "hazards": {"Vehicle / mobile equipment energy"},
        "activities": {"Driving / vehicle movement"},
        "keywords": ["driving", "reversing", "reversed", "backing", "transit", "distracted"],
    },
    "Safe Mechanical Lifting": {
        "hazards": {"Falling object / dropped object energy"},
        "activities": {"Lifting / handling"},
        "keywords": ["lifting", "lifted", "rigging", "load", "suspended"],
    },
    "Confined Space": {
        "keywords": ["confined space", "inside the chute", "discharge chute", "restricted space"],
    },
    "Hot Work": {
        "hazards": {"Thermal energy", "Fire / explosion energy"},
        "keywords": ["furnace", "oxyfuel", "ignition", "flame", "hot work"],
    },
    "Work Authorisation": {
        "keywords": ["permit to work", "work permit", "authorization", "authorisation"],
    },
    "Bypassing Safety Controls": {
        "keywords": ["bypassed", "override", "not isolated", "not blocked", "bonnet open"],
    },
}


def map_lsr(
    hazards: List[str],
    exposures: List[str],
    activities: List[str],
    evidence_text: str,
) -> List[Dict]:
    out = []
    for rule, cfg in LSR_RULES.items():
        score = 0.0
        evidence = []

        for item in cfg.get("hazards", set()):
            if item in hazards:
                score += 0.35
                evidence.append(item)

        for item in cfg.get("exposures", set()):
            if item in exposures:
                score += 0.35
                evidence.append(item)

        for item in cfg.get("activities", set()):
            if item in activities:
                score += 0.20
                evidence.append(item)

        for kw in cfg.get("keywords", []):
            if kw.lower() in evidence_text.lower():
                score += 0.08
                evidence.append(kw)

        # Important: generic direct contact is NOT enough for Line of Fire.
        if rule == "Line of Fire" and "Line-of-fire exposure" not in exposures:
            score *= 0.25

        if score >= 0.35:
            out.append({
                "rule": rule,
                "score": round(min(1.0, score), 3),
                "evidence": sorted(set(evidence)),
            })

    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:3]


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------

HIGH_ENERGY = {
    "Electrical energy",
    "Pressure / stored energy",
    "Thermal energy",
    "Fire / explosion energy",
    "Vehicle / mobile equipment energy",
    "Falling object / dropped object energy",
    "Fall-from-height energy",
    "Caught-between / pinch energy",
    "Mechanical / moving equipment energy",
}


def barrier_status(value: str) -> Tuple[List[str], List[str]]:
    present, failed = [], []
    for item in labels(value):
        m = re.match(r"(.+?)\s+\[(present|failed_or_absent)\]$", item, flags=re.I)
        if not m:
            present.append(item)
            continue
        name = m.group(1).strip()
        if m.group(2).lower() == "present":
            present.append(name)
        else:
            failed.append(name)
    return sorted(set(present)), sorted(set(failed))


def mechanism_signals(row) -> Dict[str, bool]:
    hazards = labels(row.get("hazards"))
    exposures = labels(row.get("exposures"))
    actual = labels(row.get("actual_consequences"))
    potential = labels(row.get("potential_consequences"))
    activities = labels(row.get("activities"))

    evidence = " ".join(
        clean(row.get(c))
        for c in [
            "hazard_evidence",
            "exposure_evidence",
            "actual_consequence_evidence",
            "potential_consequence_evidence",
            "barrier_evidence",
            "activity_evidence",
        ]
    )
    low = low_words(evidence)

    return {
        "electrical": "Electrical energy" in hazards,
        "chemical": "Chemical energy / hazardous substance" in hazards,
        "thermal": "Thermal energy" in hazards,
        "vehicle": "Vehicle / mobile equipment energy" in hazards,
        "fall_height": "Fall-from-height energy" in hazards,
        "caught_between": "Caught-between / pinch energy" in hazards
        or "Caught-between / pinch exposure" in exposures,
        "projectile": "Falling object / dropped object energy" in hazards
        or "Line-of-fire exposure" in exposures,
        "mechanical": "Mechanical / moving equipment energy" in hazards,
        "actual_severe": any(
            x in {"Fatality", "Electrical injury", "Crushing / amputation"}
            for x in actual
        ),
        "potential_severe": bool(potential),
        "direct_exposure": bool(exposures),
        "line_of_fire": "Line-of-fire exposure" in exposures,
        "work_at_height": "Working at height" in activities,
        "confined_space": contains_any(
            low,
            ["confined space", "inside the chute", "discharge chute", "restricted space"],
        ),
        "barrier_failure": any(
            "[failed_or_absent]" in clean(row.get("observed_barriers", ""))
            for _ in [0]
        ),
        "seat_belt_present": contains_any(
            low, ["seat belt", "safety belt"]
        ),
        "strong_vehicle_event": contains_any(
            low,
            ["rollover", "roll over", "ejection", "collision", "crash", "vehicle turns", "excavated"],
        ),
        "low_fall_06m": contains_any(low, ["0.60", "0.6 m", "0.60 cm"]),
        "medium_fall": contains_any(low, ["2.5 m", "2.98m", "2.98 m", "3 m", "3m"]),
        "chemical_face_eye": contains_any(
            low, ["face", "eye", "eyes", "upper limb", "respiratory", "inhalation"]
        ),
        "chemical_concentration_evidence": contains_any(
            low, ["concentration", "high concentration", "large volume", "significant release"]
        ),
        "vehicle_occupant_protection": contains_any(
            low, ["seat belt", "safety belt"]
        ),
    }


def score(row) -> Dict:
    hazards = labels(row.get("hazards"))
    exposures = labels(row.get("exposures"))
    actual = labels(row.get("actual_consequences"))
    potential = labels(row.get("potential_consequences"))
    activities = labels(row.get("activities"))

    present, failed = barrier_status(row.get("observed_barriers", ""))

    sig = mechanism_signals(row)

    score = 0.0
    reasons = []
    gates = []

    # A. Mechanism severity.
    if any(h in HIGH_ENERGY for h in hazards):
        score += 0.20
        reasons.append("high-energy mechanism")

    # B. Meaningful human exposure.
    if sig["caught_between"] or sig["line_of_fire"] or sig["electrical"] or sig["vehicle"]:
        score += 0.20
        reasons.append("specific exposure pathway")
    elif exposures:
        score += 0.06
        reasons.append("generic exposure signal")

    # C. Actual severe event.
    if sig["actual_severe"]:
        score += 0.35
        reasons.append("severe actual consequence")

    # D. Credible potential severity, but apply mechanism-specific gates.
    potential_allowed = False

    if sig["potential_severe"]:
        # Electrical: requires electrical mechanism plus person-exposure evidence.
        if sig["electrical"]:
            potential_allowed = sig["direct_exposure"]

        # Caught-between: explicit pinch/trapping is strong enough.
        elif sig["caught_between"]:
            potential_allowed = True

        # Fall: medium/high fall can qualify; low 0.6 m alone cannot.
        elif sig["fall_height"]:
            potential_allowed = sig["medium_fall"] and sig["direct_exposure"]

        # Vehicle: require an actual uncontrolled event beyond merely having
        # a vehicle in the narrative.
        elif sig["vehicle"]:
            potential_allowed = sig["strong_vehicle_event"] and sig["direct_exposure"]

        # Chemical: require direct face/eye/respiratory exposure AND stronger
        # hazardous-release evidence. A generic splash is insufficient.
        elif sig["chemical"]:
            potential_allowed = sig["chemical_face_eye"] and (
                sig["chemical_concentration_evidence"]
                or sig["line_of_fire"]
            )

        # Thermal/projectile: direct exposure + mechanism.
        elif sig["thermal"] or sig["projectile"]:
            potential_allowed = sig["direct_exposure"]

        # Mechanical entanglement: explicit caught/pulled/hooked signals.
        elif sig["mechanical"]:
            evidence = low_words(
                clean(row.get("hazard_evidence"))
                + " "
                + clean(row.get("exposure_evidence"))
            )
            potential_allowed = contains_any(
                evidence,
                ["hooked", "pulled", "caught", "propeller", "fan", "rotating"],
            ) and sig["direct_exposure"]

    if potential_allowed:
        score += 0.25
        gates.append("potential severity gate passed")
        reasons.append("credible severe potential pathway")
    elif sig["potential_severe"]:
        gates.append("potential severity gate failed")
        reasons.append("potential severity signal not sufficiently grounded")

    # E. Barrier failure is a modifier, not a primary SIF gate.
    if failed:
        score += 0.10
        reasons.append("explicit barrier failure/absence")

    # F. Explicitly observed protective controls reduce escalation a little,
    # but never erase a severe mechanism.
    if present and not sig["actual_severe"]:
        score -= 0.03
        reasons.append("protective control explicitly present")

    # Hard low-energy exclusions.
    low_energy = False

    if sig["low_fall_06m"] and not sig["actual_severe"]:
        low_energy = True
        reasons.append("0.6 m fall is low-energy without secondary severe mechanism")

    if actual and all(
        c in {
            "Cut / laceration",
            "Minor irritation / swelling",
            "Musculoskeletal injury",
            "Struck-by / impact injury",
        }
        for c in actual
    ) and not sig["actual_severe"] and not potential_allowed:
        low_energy = True
        reasons.append("observed outcome is low severity without credible escalation gate")

    if low_energy:
        score = min(score, 0.30)

    score = max(0.0, min(1.0, score))

    # Bands are operating categories, not probabilities.
    if sig["actual_severe"] and (sig["electrical"] or sig["caught_between"] or sig["fall_height"] or sig["vehicle"]):
        band = "HIGH"
    elif score >= 0.60:
        band = "HIGH"
    elif score >= 0.35:
        band = "MEDIUM"
    else:
        band = "LOW"

    evidence_text = " ".join(
        clean(row.get(c))
        for c in [
            "hazard_evidence",
            "exposure_evidence",
            "actual_consequence_evidence",
            "potential_consequence_evidence",
            "barrier_evidence",
            "activity_evidence",
        ]
    )

    lsr = map_lsr(hazards, exposures, activities, evidence_text)

    return {
        "sif_precursor_band": band,
        "sif_precursor_score": round(score, 3),
        "primary_trigger": (
            potential[0]
            if potential_allowed and potential
            else actual[0]
            if actual
            else hazards[0]
            if hazards
            else "No strong SIF precursor signal"
        ),
        "reason_codes": "; ".join(dict.fromkeys(reasons)),
        "gate_codes": "; ".join(gates),
        "barriers_present": "; ".join(present),
        "barriers_failed": "; ".join(failed),
        "lsr_candidates": "; ".join(x["rule"] for x in lsr),
        "lsr_scores": "; ".join(f'{x["rule"]}:{x["score"]:.3f}' for x in lsr),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)

    results = []
    detailed = []

    for _, row in df.iterrows():
        result = score(row)
        rid = clean(row.get("report_id"))

        results.append({
            "report_id": rid,
            **result,
        })

        detailed.append({
            "report_id": rid,
            "reasoning": result,
        })

    out = pd.DataFrame(results)

    csv_path = OUTPUT_DIR / "sif_reasoning.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    jsonl_path = OUTPUT_DIR / "sif_reasoning_detailed.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in detailed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    lsr_counts: Dict[str, int] = {}
    for cell in out["lsr_candidates"].fillna(""):
        for rule in [x.strip() for x in cell.split(";") if x.strip()]:
            lsr_counts[rule] = lsr_counts.get(rule, 0) + 1

    summary = {
        "version": "0.2",
        "records": int(len(out)),
        "bands": out["sif_precursor_band"].value_counts().to_dict(),
        "mean_score": round(float(out["sif_precursor_score"].mean()), 4),
        "high_score_count": int((out["sif_precursor_score"] >= 0.60).sum()),
        "lsr_candidate_counts": dict(sorted(
            lsr_counts.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )),
        "notes": [
            "Heuristic operating score, not a probability.",
            "Mechanism-specific gates reduce generic contact/fall/chemical false positives.",
            "Human annotations and gold SIF labels are not inputs.",
        ],
    }

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 78)
    print("RAKSHAK SIF PRECURSOR REASONING ENGINE v0.2")
    print("=" * 78)
    print(f"Records : {len(out)}")
    print()
    print("BANDS")
    for band in ["HIGH", "MEDIUM", "LOW"]:
        print(f"  {band:<8}: {int((out['sif_precursor_band'] == band).sum())}")
    print()
    print(f"Mean score        : {summary['mean_score']:.3f}")
    print(f"Score >= 0.60     : {summary['high_score_count']}")
    print()
    print("TOP LSR CANDIDATE COUNTS")
    for rule, count in list(summary["lsr_candidate_counts"].items())[:10]:
        print(f"  {rule:<30}: {count}")
    print()
    print(f"CSV    : {csv_path}")
    print(f"JSONL  : {jsonl_path}")
    print(f"Summary: {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
