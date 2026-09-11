"""
RAKSHAK SIF Precursor Reasoning Engine v0.1
==========================================

Consumes normalized v0.6 extraction output and produces an interpretable,
rule-based SIF precursor assessment.

Important:
- This is NOT a trained SIF classifier.
- It does NOT use human SIF labels, potential accident level, or annotation notes.
- It does NOT invent a consequence solely from a generic keyword.
- It keeps "observed event", "credible severity pathway", and "barrier condition"
  separate.
- It maps to a conservative LSR candidate taxonomy for engineering use.

IOGP note:
The mapping names below use the established IOGP Life-Saving Rule concepts
(e.g. Work at Height, Isolation, Line of Fire, Seat Belt, Safe Mechanical
Lifting, Work Authorisation, Confined Space, Hot Work, Bypassing Safety
Controls, Driving). Verify against the exact OIL/IOGP implementation version
before production deployment.

Input:
    experiments/safety_information_extraction_v0.6/safety_extraction_flat.csv

Outputs:
    experiments/sif_reasoning_v0.1/
        sif_reasoning.csv
        sif_reasoning_detailed.jsonl
        summary.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "safety_information_extraction_v0.6"
    / "safety_extraction_flat.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "sif_reasoning_v0.1"


def clean(v) -> str:
    if v is None:
        return ""
    try:
        import pandas as pd
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def has_any(text: str, phrases: List[str]) -> bool:
    low = text.lower()
    return any(p.lower() in low for p in phrases)


def split_labels(value: str) -> List[str]:
    value = clean(value)
    if not value:
        return []
    parts = re.split(r"\s*;\s*", value)
    return [x.strip() for x in parts if x.strip()]


# ---------------------------------------------------------------------------
# LSR candidate mapping
# ---------------------------------------------------------------------------

LSR_RULES = {
    "Work at Height": {
        "hazards": ["Fall-from-height energy"],
        "exposures": ["Fall exposure"],
        "activity": ["Working at height"],
        "keywords": ["ladder", "scaffold", "height", "reel", "chimney"],
    },
    "Isolation": {
        "hazards": [
            "Electrical energy",
            "Mechanical / moving equipment energy",
            "Pressure / stored energy",
        ],
        "keywords": [
            "energized", "live", "lockout", "loto", "isolation",
            "de-energized", "bonnet open", "operating",
        ],
    },
    "Line of Fire": {
        "exposures": [
            "Line-of-fire exposure",
            "Direct physical contact",
            "Vehicle exposure",
            "Thermal exposure",
            "Chemical exposure",
        ],
        "hazards": [
            "Falling object / dropped object energy",
            "Vehicle / mobile equipment energy",
            "Pressure / stored energy",
            "Thermal energy",
            "Fire / explosion energy",
        ],
        "keywords": [
            "projection", "projected", "line of fire", "impact",
            "released", "struck", "splash",
        ],
    },
    "Seat Belt": {
        "keywords": ["seat belt", "safety belt", "belt"],
        "hazards": ["Vehicle / mobile equipment energy"],
    },
    "Driving": {
        "hazards": ["Vehicle / mobile equipment energy"],
        "activity": ["Driving / vehicle movement"],
        "keywords": [
            "driving", "reversing", "reversed", "backing",
            "vehicle movement", "distracted",
        ],
    },
    "Safe Mechanical Lifting": {
        "hazards": ["Falling object / dropped object energy"],
        "activity": ["Lifting / handling"],
        "keywords": ["lifting", "lifted", "rigging", "suspended", "load"],
    },
    "Confined Space": {
        "keywords": [
            "confined space", "discharge chute", "inside the chute",
            "restricted space",
        ],
    },
    "Hot Work": {
        "hazards": ["Fire / explosion energy", "Thermal energy"],
        "activity": ["Process operation"],
        "keywords": ["furnace", "oxyfuel", "ignition", "flame", "hot work"],
    },
    "Work Authorisation": {
        "keywords": ["permit to work", "work permit", "authorization", "authorisation"],
    },
    "Bypassing Safety Controls": {
        "keywords": [
            "bypassed", "override", "disabled", "bonnet open",
            "not blocked", "not isolated",
        ],
    },
}


def lsr_candidates(
    hazards: List[str],
    exposures: List[str],
    activities: List[str],
    text: str,
) -> List[dict]:
    out = []
    for rule, cfg in LSR_RULES.items():
        score = 0.0
        evidence = []

        for x in cfg.get("hazards", []):
            if x in hazards:
                score += 0.35
                evidence.append(x)

        for x in cfg.get("exposures", []):
            if x in exposures:
                score += 0.35
                evidence.append(x)

        for x in cfg.get("activity", []):
            if x in activities:
                score += 0.20
                evidence.append(x)

        for kw in cfg.get("keywords", []):
            if kw.lower() in text.lower():
                score += 0.08
                evidence.append(kw)

        score = min(1.0, score)

        if score >= 0.35:
            out.append({
                "rule": rule,
                "score": round(score, 3),
                "evidence": sorted(set(evidence)),
            })

    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:3]


# ---------------------------------------------------------------------------
# SIF precursor reasoning
# ---------------------------------------------------------------------------

HIGH_ENERGY_HAZARDS = {
    "Electrical energy",
    "Pressure / stored energy",
    "Thermal energy",
    "Fire / explosion energy",
    "Vehicle / mobile equipment energy",
    "Falling object / dropped object energy",
    "Fall-from-height energy",
    "Caught-between / pinch energy",
}

HIGH_SEVERITY_CONSEQUENCES = {
    "Fatality",
    "Electrical injury",
    "Crushing / amputation",
}

POTENTIAL_SEVERE = {
    "Potential severe electrical injury",
    "Potential severe caught-between injury",
    "Potential severe fall injury",
    "Potential severe vehicle injury",
    "Potential severe chemical injury",
    "Potential severe thermal injury",
    "Potential severe projectile injury",
    "Potential severe entanglement / machine injury",
}


def parse_barrier_status(value: str):
    present = []
    failed = []

    for item in split_labels(value):
        m = re.match(r"(.+?)\s+\[(present|failed_or_absent)\]$", item, flags=re.I)
        if m:
            label = m.group(1).strip()
            status = m.group(2).lower()
            if status == "present":
                present.append(label)
            else:
                failed.append(label)
        else:
            present.append(item)

    return sorted(set(present)), sorted(set(failed))


def score_record(row) -> dict:
    hazards = split_labels(row.get("hazards", ""))
    exposures = split_labels(row.get("exposures", ""))
    actual = split_labels(row.get("actual_consequences", ""))
    potential = split_labels(row.get("potential_consequences", ""))
    activities = split_labels(row.get("activities", ""))
    barriers_present, barriers_failed = parse_barrier_status(
        row.get("observed_barriers", "")
    )

    # The v0.6 flat file does not contain the original narrative. Use evidence
    # fields for keyword-based LSR matching.
    text = " ".join([
        clean(row.get("hazard_evidence", "")),
        clean(row.get("exposure_evidence", "")),
        clean(row.get("actual_consequence_evidence", "")),
        clean(row.get("potential_consequence_evidence", "")),
        clean(row.get("barrier_evidence", "")),
        clean(row.get("activity_evidence", "")),
    ])

    score = 0.0
    reasons = []

    if any(h in HIGH_ENERGY_HAZARDS for h in hazards):
        score += 0.25
        reasons.append("high-energy hazard")

    if exposures:
        score += 0.20
        reasons.append("human exposure signal")

    if any(c in HIGH_SEVERITY_CONSEQUENCES for c in actual):
        score += 0.35
        reasons.append("severe/critical actual consequence")

    if any(c in POTENTIAL_SEVERE for c in potential):
        score += 0.25
        reasons.append("credible severe potential pathway")

    if barriers_failed:
        score += 0.15
        reasons.append("barrier gap/failure signal")

    # Strongly penalize treating low-energy events as SIF simply because a
    # generic fall/cut/injury term exists.
    low_energy_only = (
        bool(actual)
        and all(
            c in {
                "Cut / laceration",
                "Minor irritation / swelling",
                "Minor irritation / swelling",
                "Musculoskeletal injury",
                "Struck-by / impact injury",
            }
            for c in actual
        )
        and not any(h in HIGH_ENERGY_HAZARDS for h in hazards)
        and not potential
    )

    if low_energy_only:
        score -= 0.20
        reasons.append("low-energy-only observed event")

    score = max(0.0, min(1.0, score))

    # Conservative bands.
    if score >= 0.70:
        band = "HIGH"
    elif score >= 0.45:
        band = "MEDIUM"
    else:
        band = "LOW"

    # Human-readable trigger.
    if potential:
        trigger = " + ".join(potential[:2])
    elif any(h in HIGH_ENERGY_HAZARDS for h in hazards):
        trigger = "High-energy hazard with exposure"
    elif actual:
        trigger = actual[0]
    else:
        trigger = "No strong SIF precursor signal"

    lsr = lsr_candidates(hazards, exposures, activities, text)

    return {
        "sif_precursor_band": band,
        "sif_precursor_score": round(score, 3),
        "primary_trigger": trigger,
        "reason_codes": "; ".join(reasons),
        "lsr_candidates": "; ".join(x["rule"] for x in lsr),
        "lsr_scores": "; ".join(f'{x["rule"]}:{x["score"]:.3f}' for x in lsr),
        "barriers_present": "; ".join(barriers_present),
        "barriers_failed": "; ".join(barriers_failed),
    }


def main():
    import pandas as pd

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    results = []
    detailed = []

    for _, row in df.iterrows():
        result = score_record(row)

        output = {
            "report_id": clean(row.get("report_id")),
            **result,
        }

        results.append(output)

        detailed.append({
            "report_id": clean(row.get("report_id")),
            "input": {
                "hazards": split_labels(row.get("hazards", "")),
                "exposures": split_labels(row.get("exposures", "")),
                "actual_consequences": split_labels(row.get("actual_consequences", "")),
                "potential_consequences": split_labels(row.get("potential_consequences", "")),
                "activities": split_labels(row.get("activities", "")),
                "observed_barriers": split_labels(row.get("observed_barriers", "")),
            },
            "reasoning": result,
        })

    out = pd.DataFrame(results)

    csv_path = OUTPUT_DIR / "sif_reasoning.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    jsonl_path = OUTPUT_DIR / "sif_reasoning_detailed.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in detailed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "version": "0.1",
        "records": int(len(out)),
        "band_counts": out["sif_precursor_band"].value_counts().to_dict(),
        "mean_score": round(float(out["sif_precursor_score"].mean()), 4),
        "lsr_candidate_counts": {},
        "notes": [
            "Rule-based reasoning diagnostic, not a trained classifier.",
            "Does not consume human SIF labels or annotation notes.",
            "LSR names are candidate mappings and require final verification against the OIL implementation.",
        ],
    }

    for cell in out["lsr_candidates"].fillna(""):
        for rule in [x.strip() for x in cell.split(";") if x.strip()]:
            summary["lsr_candidate_counts"][rule] = (
                summary["lsr_candidate_counts"].get(rule, 0) + 1
            )

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 78)
    print("RAKSHAK SIF PRECURSOR REASONING ENGINE v0.1")
    print("=" * 78)
    print(f"Records : {len(out)}")
    print()
    print("BANDS")
    for band in ["HIGH", "MEDIUM", "LOW"]:
        print(f"  {band:<8}: {int((out['sif_precursor_band'] == band).sum())}")
    print()
    print(f"Mean score : {summary['mean_score']:.3f}")
    print()
    print("TOP LSR CANDIDATE COUNTS")
    top = sorted(
        summary["lsr_candidate_counts"].items(),
        key=lambda x: x[1],
        reverse=True,
    )[:10]
    for rule, count in top:
        print(f"  {rule:<30}: {count}")
    print()
    print(f"CSV    : {csv_path}")
    print(f"JSONL  : {jsonl_path}")
    print(f"Summary: {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
