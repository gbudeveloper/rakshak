"""
RAKSHAK / SIF-Insight Inference Pipeline v1.0
==============================================

Production-style research/demo inference layer for one new incident narrative.

Pipeline
--------
Narrative
   -> SentenceTransformer embedding
   -> frozen RAKSHAK v1.2 model
   -> evidence engine v1.5
   -> IOGP Life-Saving Rule candidates
   -> model/evidence conflict logic
   -> reviewer priority
   -> structured JSON / human-readable report

Important
---------
- The classifier is frozen. This script does not retrain it.
- Model inputs are narrative-only: description text.
- Evidence/LSR mapping is reviewer support, not autonomous SIF adjudication.
- Missing severity/energy/proximity facts are never invented.
- LSR names below follow IOGP's published Life-Saving Rules terminology.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "rakshak_final_v1.2"
    / "rakshak_final_model.joblib"
)
EVIDENCE_MODULE = SCRIPT_DIR / "evidence_engine_v1_5.py"

ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_SEQ_LENGTH = 256

try:
    sys.path.insert(0, str(SCRIPT_DIR))
    from evidence_engine_v1_5 import analyze
except Exception as exc:
    raise RuntimeError(
        f"Could not import evidence engine from {EVIDENCE_MODULE}: {exc}"
    ) from exc


# ---------------------------------------------------------------------------
# IOGP Life-Saving Rules mapping
# ---------------------------------------------------------------------------

LSR_RULES = {
    "Bypassing Safety Controls": [
        "safety control",
        "override",
        "bypass",
        "disable",
        "barrier",
    ],
    "Confined Space": [
        "confined space",
        "tank",
        "vessel",
        "inside tank",
        "inside vessel",
    ],
    "Driving": [
        "vehicle",
        "truck",
        "driver",
        "driving",
        "road",
        "collision",
        "run over",
    ],
    "Energy Isolation": [
        "energized",
        "electric",
        "voltage",
        "isolation",
        "lockout",
        "tagout",
        "de-energized",
        "stored energy",
    ],
    "Hot Work": [
        "welding",
        "cutting",
        "grinding",
        "hot work",
        "ignition",
        "flammable",
        "oxyfuel",
    ],
    "Line of Fire": [
        "line of fire",
        "struck",
        "projectile",
        "dropped object",
        "moving object",
        "pressure release",
        "vehicle",
        "flow",
    ],
    "Safe Mechanical Lifting": [
        "lifting",
        "lifted",
        "suspended load",
        "crane",
        "rigging",
        "load",
    ],
    "Work Authorisation": [
        "permit",
        "authorisation",
        "authorization",
        "work permit",
    ],
    "Working at Height": [
        "height",
        "fall",
        "scaffold",
        "ladder",
        "tower",
        "platform",
        "elevated",
    ],
}


def load_model() -> dict[str, Any]:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Frozen model package not found:\n{MODEL_FILE}\n"
            "Run train_rakshak_final_pipeline_v1_2.py first."
        )
    return joblib.load(MODEL_FILE)


def encode_text(encoder: SentenceTransformer, text: str) -> np.ndarray:
    encoder.max_seq_length = MAX_SEQ_LENGTH

    emb = encoder.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ).astype(np.float32)

    return emb


def build_pathway_vector_from_evidence(
    text: str,
    evidence: dict[str, Any],
    feature_names: list[str],
    expected_count: int,
) -> np.ndarray:
    """
    Reconstruct the same 38 pathway-feature family used by the frozen model.

    This is intentionally aligned with the v1.2 feature naming convention:
    path_hazard_*, path_exposure_*, pathway_*, counts, interactions and
    mechanism-complete flags.

    We derive values from the v1.5 evidence engine's semantic categories.
    """
    raw = str(text).lower()
    hazards = set(
        x.strip() for x in str(evidence.get("hazards", "")).split("|") if x.strip()
    )
    exposures = set(
        x.strip() for x in str(evidence.get("exposures", "")).split("|") if x.strip()
    )
    pathways = set(
        x.strip() for x in str(evidence.get("pathways", "")).split("|") if x.strip()
    )

    # Conservative family mapping. Unknown feature names become zero.
    hazard_map = {
        "electrical": int("electrical" in hazards),
        "chemical": int("chemical" in hazards),
        "thermal": int("thermal" in hazards),
        "pressure": int("pressure" in hazards),
        "fire_explosion": int("fire_explosion" in hazards),
        "vehicle": int("vehicle" in hazards),
        "fall_height": int("fall_height" in hazards),
        "mechanical": int("mechanical" in hazards),
        "stored_projectile": int("stored_projectile" in hazards),
    }

    exposure_map = {
        "chemical_exposure": int("chemical_exposure" in exposures),
        "thermal_exposure": int("thermal_exposure" in exposures),
        "electrical_exposure": int("electrical_exposure" in exposures),
        "vehicle_exposure": int("vehicle_exposure" in exposures),
        "fall_exposure": int("fall_exposure" in exposures),
        "line_of_fire": int("line_of_fire" in exposures),
        "caught_between_exposure": int("caught_between_exposure" in exposures),
    }

    path_map = {
        "electrical_contact": int("electrical_contact" in pathways),
        "chemical_release_exposure": int("chemical_release_exposure" in pathways),
        "thermal_contact": int("thermal_contact" in pathways),
        "fall_from_height": int("fall_from_height" in pathways),
        "struck_by_projectile": int("struck_by_projectile" in pathways),
        "caught_crush": int("caught_crush" in pathways),
        "vehicle_person_collision": int("vehicle_person_collision" in pathways),
    }

    hazard_count = sum(hazard_map.values())
    exposure_count = sum(exposure_map.values())
    pathway_count = sum(path_map.values())

    values: dict[str, float] = {}

    for k, v in hazard_map.items():
        values[f"path_hazard_{k}"] = float(v)

    for k, v in exposure_map.items():
        values[f"path_exposure_{k}"] = float(v)

    for k, v in path_map.items():
        values[f"pathway_{k}"] = float(v)

    values["path_hazard_count"] = float(hazard_count)
    values["path_exposure_count"] = float(exposure_count)
    values["pathway_count"] = float(pathway_count)

    values["path_hazard_exposure"] = float(
        hazard_count > 0 and exposure_count > 0
    )
    values["path_hazard_pathway"] = float(
        hazard_count > 0 and pathway_count > 0
    )
    values["path_exposure_pathway"] = float(
        exposure_count > 0 and pathway_count > 0
    )
    values["path_complete"] = float(
        hazard_count > 0 and exposure_count > 0 and pathway_count > 0
    )

    values["path_electrical_complete"] = float(
        hazard_map["electrical"]
        and exposure_map["electrical_exposure"]
        and path_map["electrical_contact"]
    )
    values["path_chemical_complete"] = float(
        hazard_map["chemical"]
        and exposure_map["chemical_exposure"]
        and path_map["chemical_release_exposure"]
    )
    values["path_fall_complete"] = float(
        hazard_map["fall_height"]
        and exposure_map["fall_exposure"]
        and path_map["fall_from_height"]
    )
    values["path_vehicle_complete"] = float(
        hazard_map["vehicle"]
        and exposure_map["vehicle_exposure"]
        and path_map["vehicle_person_collision"]
    )
    values["path_caught_complete"] = float(
        hazard_map["mechanical"]
        and exposure_map["caught_between_exposure"]
        and path_map["caught_crush"]
    )
    values["path_line_fire"] = float(
        exposure_map["line_of_fire"]
        and (
            path_map["struck_by_projectile"]
            or hazard_map["vehicle"]
            or hazard_map["stored_projectile"]
        )
    )

    # The feature list saved with v1.2 is authoritative. Fill unknowns with 0.
    vector = np.asarray(
        [values.get(name, 0.0) for name in feature_names],
        dtype=np.float32,
    )

    if len(vector) != expected_count:
        raise ValueError(
            "Pathway feature reconstruction mismatch: "
            f"expected {expected_count}, produced {len(vector)}"
        )

    return vector


def score_model(text: str, evidence: dict[str, Any]) -> float:
    package = load_model()
    model = package["model"]
    model_feature_names = package["feature_names_after_keep"]
    keep_mask = np.asarray(package["feature_keep_mask"], dtype=bool)
    embedding_dim = int(package["embedding_dimension"])

    encoder = SentenceTransformer(ENCODER_NAME)
    embedding = encode_text(encoder, text)[0]

    # Reconstruct the full 384 + 38 vector used by the v1.2 model.
    all_feature_names = [
        f"embedding_{i:03d}" for i in range(embedding_dim)
    ]

    # feature names after the embedding features are present in the saved
    # package. Recover the complete pathway family by their order in the
    # package, then fill the remaining pathway names from a canonical set.
    canonical_pathway_names = [
        "path_hazard_electrical",
        "path_hazard_chemical",
        "path_hazard_thermal",
        "path_hazard_pressure",
        "path_hazard_fire_explosion",
        "path_hazard_vehicle",
        "path_hazard_fall_height",
        "path_hazard_mechanical",
        "path_hazard_stored_projectile",
        "path_exposure_chemical_exposure",
        "path_exposure_thermal_exposure",
        "path_exposure_electrical_exposure",
        "path_exposure_vehicle_exposure",
        "path_exposure_fall_exposure",
        "path_exposure_line_of_fire",
        "path_exposure_caught_between_exposure",
        "pathway_electrical_contact",
        "pathway_chemical_release_exposure",
        "pathway_thermal_contact",
        "pathway_fall_from_height",
        "pathway_struck_by_projectile",
        "pathway_caught_crush",
        "pathway_vehicle_person_collision",
        "path_hazard_count",
        "path_exposure_count",
        "pathway_count",
        "path_hazard_exposure",
        "path_hazard_pathway",
        "path_exposure_pathway",
        "path_complete",
        "path_electrical_complete",
        "path_chemical_complete",
        "path_fall_complete",
        "path_vehicle_complete",
        "path_caught_complete",
        "path_line_fire",
    ]

    full_names = all_feature_names + canonical_pathway_names
    pathway_vector = build_pathway_vector_from_evidence(
        text,
        evidence,
        canonical_pathway_names,
        len(canonical_pathway_names),
    )

    full_vector = np.concatenate(
        [embedding, pathway_vector],
        axis=0,
    )

    if len(full_vector) != len(keep_mask):
        raise ValueError(
            f"Frozen feature mask expects {len(keep_mask)} values, "
            f"but inference produced {len(full_vector)}."
        )

    X = full_vector[keep_mask].reshape(1, -1)

    probability = float(model.predict_proba(X)[0, 1])

    return probability


def lsr_candidates(
    text: str,
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    text_n = text.lower()

    candidates: list[dict[str, Any]] = []

    for rule, terms in LSR_RULES.items():
        matched = [term for term in terms if term in text_n]

        score = 0.0

        if matched:
            score += min(0.60, 0.20 * len(matched))

        pathways = str(evidence.get("pathways", ""))
        exposures = str(evidence.get("exposures", ""))
        hazards = str(evidence.get("hazards", ""))

        if rule == "Driving" and (
            "vehicle" in hazards or "vehicle_exposure" in exposures
            or "vehicle_person_collision" in pathways
        ):
            score += 0.40

        elif rule == "Energy Isolation" and (
            "electrical" in hazards or "electrical_exposure" in exposures
            or "electrical_contact" in pathways
        ):
            score += 0.40

        elif rule == "Working at Height" and (
            "fall_height" in hazards or "fall_exposure" in exposures
            or "fall_from_height" in pathways
        ):
            score += 0.40

        elif rule == "Line of Fire" and (
            "line_of_fire" in exposures
            or "struck_by_projectile" in pathways
            or "vehicle_person_collision" in pathways
        ):
            score += 0.40

        elif rule == "Hot Work" and (
            "fire_explosion" in hazards
            or "hot" in text_n
            or "oxyfuel" in text_n
        ):
            score += 0.40

        elif rule == "Confined Space" and (
            "confined space" in text_n
            or "tank" in text_n
            or "vessel" in text_n
        ):
            score += 0.40

        elif rule == "Safe Mechanical Lifting" and (
            "lifting" in text_n
            or "crane" in text_n
            or "suspended load" in text_n
        ):
            score += 0.40

        elif rule == "Bypassing Safety Controls" and (
            "bypass" in text_n
            or "override" in text_n
            or "disable" in text_n
            or "barrier" in text_n
        ):
            score += 0.40

        elif rule == "Work Authorisation" and (
            "permit" in text_n
            or "authorization" in text_n
            or "authorisation" in text_n
        ):
            score += 0.40

        score = min(1.0, score)

        if score > 0:
            candidates.append({
                "rule": rule,
                "candidate_score": round(score, 4),
                "matched_terms": matched[:8],
            })

    candidates.sort(
        key=lambda x: (-x["candidate_score"], x["rule"])
    )

    return candidates[:3]


def final_priority(
    probability: float,
    evidence: dict[str, Any],
    conflicts: list[str],
) -> str:
    if conflicts:
        if any(
            c in {
                "HIGH_SCORE_MODEL_CONFLICT",
                "LOW_SCORE_MODEL_MISS_WITH_EXPLICIT_PATHWAY",
                "MODEL_MISS_WITH_EXPLICIT_EVIDENCE",
            }
            for c in conflicts
        ):
            return "P1"

    if probability >= 0.80:
        return "P2"

    if probability >= 0.50:
        return "P3"

    if evidence.get("complete_pathway") == 1:
        return "P3"

    if evidence.get("pathway_signal_strength", 0) >= 1:
        return "P4"

    return "P5"


def build_conflicts(
    probability: float,
    evidence: dict[str, Any],
) -> list[str]:
    conflicts: list[str] = []

    complete = bool(evidence.get("complete_pathway"))
    explicit_path = bool(evidence.get("pathway_signal"))
    exposure = bool(evidence.get("exposure_signal"))

    # Strong model score with no/support-limited evidence.
    if probability >= 0.75 and evidence.get("sif_context_status") in {
        "UNKNOWN",
        "CONTEXT_LIMITED",
    }:
        conflicts.append("HIGH_SCORE_MODEL_CONFLICT")

    # Narrative evidence that materially contradicts a low model score.
    if probability < 0.20 and explicit_path and exposure:
        conflicts.append("MODEL_MISS_WITH_EXPLICIT_EVIDENCE")

    if probability < 0.50 and complete:
        conflicts.append("LOW_SCORE_MODEL_MISS_WITH_EXPLICIT_PATHWAY")

    return conflicts


def run_inference(text: str) -> dict[str, Any]:
    text = str(text).strip()

    if not text:
        raise ValueError("Incident narrative cannot be empty.")

    evidence = analyze(text)
    probability = score_model(text, evidence)

    conflicts = build_conflicts(
        probability,
        evidence,
    )

    lsr = lsr_candidates(
        text,
        evidence,
    )

    priority = final_priority(
        probability,
        evidence,
        conflicts,
    )

    if conflicts:
        decision = "HSSE_REVIEW_REQUIRED"
    elif probability >= 0.50:
        decision = "PRIORITIZE_FOR_HSSE_REVIEW"
    else:
        decision = "LOWER_PRIORITY_RETAIN_FOR_REVIEW"

    return {
        "system": "RAKSHAK / SIF-Insight",
        "version": "1.0",
        "input": {
            "narrative": text,
            "source_policy": "Narrative-only model inference",
        },
        "model": {
            "encoder": ENCODER_NAME,
            "sif_precursor_probability": round(probability, 6),
            "reference_threshold": 0.50,
            "interpretation": (
                "Ranking/triage score; not a calibrated autonomous SIF verdict."
            ),
        },
        "evidence": evidence,
        "lsr_candidates": lsr,
        "review": {
            "reviewer_priority": priority,
            "conflicts": conflicts,
            "decision": decision,
            "human_in_loop": True,
        },
        "governance": {
            "model_does_not_use": [
                "SIF label",
                "actual outcome",
                "potential consequence",
                "potential accident level",
                "annotation notes",
                "LSR annotation label",
            ],
            "counterfactual_policy": (
                "Do not invent missing severity, energy, distance, concentration, "
                "quantity, duration, exposure area or number of affected people."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAKSHAK single-incident inference pipeline"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--text",
        help="Incident narrative text.",
    )
    source.add_argument(
        "--file",
        help="UTF-8 text file containing one incident narrative.",
    )
    source.add_argument(
        "--json",
        dest="json_file",
        help="JSON file with {'description': '...'}",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.text is not None:
        text = args.text

    elif args.file is not None:
        text = Path(args.file).read_text(encoding="utf-8")

    else:
        payload = json.loads(
            Path(args.json_file).read_text(encoding="utf-8")
        )
        text = payload["description"]

    result = run_inference(text)

    print(
        json.dumps(
            result,
            indent=None if args.compact else 2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
