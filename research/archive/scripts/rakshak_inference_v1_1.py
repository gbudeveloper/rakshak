"""
RAKSHAK / SIF-Insight Inference Pipeline v1.1
==============================================

Fixes v1.0 feature-shape bug.

The frozen v1.2 model expects:
    384 transformer features + 38 pathway features = 422 features

v1.0 reconstructed only 36 pathway features. v1.1 reconstructs the full
38-feature family used by the v1.2 training pipeline:

10 hazard signals
8 exposure signals
7 mechanism/pathway signals
3 counts
4 pairwise signals
6 mechanism-complete signals
= 38

The classifier remains frozen; no retraining is performed.
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

ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_SEQ_LENGTH = 256

sys.path.insert(0, str(SCRIPT_DIR))

try:
    from evidence_engine_v1_5 import analyze
except Exception as exc:
    raise RuntimeError(
        f"Could not import evidence_engine_v1_5.py: {exc}"
    ) from exc


# ---------------------------------------------------------------------------
# IMPORTANT: Exact 38-feature pathway schema expected by v1.2
# ---------------------------------------------------------------------------

PATHWAY_FEATURE_NAMES = [
    # 10 hazards
    "path_hazard_electrical",
    "path_hazard_mechanical",
    "path_hazard_pressure",
    "path_hazard_chemical",
    "path_hazard_thermal",
    "path_hazard_fire_explosion",
    "path_hazard_vehicle",
    "path_hazard_fall_height",
    "path_hazard_caught_between",
    "path_hazard_dropped_object",

    # 8 exposures
    "path_exposure_direct_contact",
    "path_exposure_line_of_fire",
    "path_exposure_caught_between",
    "path_exposure_fall",
    "path_exposure_chemical",
    "path_exposure_thermal",
    "path_exposure_electrical",
    "path_exposure_vehicle",

    # 7 mechanism/pathway features
    "pathway_struck_by_projectile",
    "pathway_caught_crush",
    "pathway_fall_from_height",
    "pathway_electrical_contact",
    "pathway_chemical_release_exposure",
    "pathway_thermal_contact",
    "pathway_vehicle_person_collision",

    # 3 counts
    "path_hazard_count",
    "path_exposure_count",
    "pathway_count",

    # 4 pairwise interactions
    "path_hazard_exposure",
    "path_hazard_pathway",
    "path_exposure_pathway",
    "path_complete",

    # 6 mechanism-complete flags
    "path_electrical_complete",
    "path_chemical_complete",
    "path_fall_complete",
    "path_vehicle_complete",
    "path_caught_complete",
    "path_line_fire",
]

if len(PATHWAY_FEATURE_NAMES) != 38:
    raise RuntimeError(
        f"Internal schema error: expected 38 pathway features, "
        f"got {len(PATHWAY_FEATURE_NAMES)}"
    )


# ---------------------------------------------------------------------------
# IOGP Life-Saving Rule candidate map
# ---------------------------------------------------------------------------

LSR_RULES = {
    "Bypassing Safety Controls": [
        "bypass", "override", "disable", "safety control", "barrier"
    ],
    "Confined Space": [
        "confined space", "tank", "vessel", "inside tank", "inside vessel"
    ],
    "Driving": [
        "vehicle", "truck", "driver", "driving", "collision", "run over"
    ],
    "Energy Isolation": [
        "energized", "electric", "voltage", "isolation",
        "lockout", "tagout", "de-energized", "stored energy"
    ],
    "Hot Work": [
        "welding", "cutting", "grinding", "hot work",
        "ignition", "flammable", "oxyfuel"
    ],
    "Line of Fire": [
        "line of fire", "struck", "projectile", "dropped object",
        "moving object", "pressure release", "vehicle", "flow"
    ],
    "Safe Mechanical Lifting": [
        "lifting", "lifted", "suspended load", "crane", "rigging"
    ],
    "Work Authorisation": [
        "permit", "authorization", "authorisation"
    ],
    "Working at Height": [
        "height", "fall", "scaffold", "ladder", "tower",
        "platform", "elevated"
    ],
}


def load_package() -> dict[str, Any]:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Frozen model package not found:\n{MODEL_FILE}"
        )
    return joblib.load(MODEL_FILE)


def encode(text: str) -> np.ndarray:
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = SentenceTransformer(ENCODER_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH

    emb = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
        device=device,
    )

    return np.asarray(emb[0], dtype=np.float32)


def reconstruct_pathway_features(evidence: dict[str, Any]) -> np.ndarray:
    hazards = {
        x.strip()
        for x in str(evidence.get("hazards", "")).split("|")
        if x.strip()
    }

    exposures_raw = {
        x.strip()
        for x in str(evidence.get("exposures", "")).split("|")
        if x.strip()
    }

    # Evidence engine v1.5 may emit semantic names. Map them to the exact
    # training representation.
    exposures = set()
    if "direct_contact" in exposures_raw:
        exposures.add("direct_contact")
    if "line_of_fire" in exposures_raw:
        exposures.add("line_of_fire")
    if "caught_between_exposure" in exposures_raw:
        exposures.add("caught_between")
    if "fall_exposure" in exposures_raw:
        exposures.add("fall")
    if "chemical_exposure" in exposures_raw:
        exposures.add("chemical")
    if "thermal_exposure" in exposures_raw:
        exposures.add("thermal")
    if "electrical_exposure" in exposures_raw:
        exposures.add("electrical")
    if "vehicle_exposure" in exposures_raw:
        exposures.add("vehicle")

    pathways = {
        x.strip()
        for x in str(evidence.get("pathways", "")).split("|")
        if x.strip()
    }

    # The evidence engine has "stored_projectile" as a context category,
    # but the training representation uses dropped_object for this family.
    hazard_presence = {
        "electrical": int("electrical" in hazards),
        "mechanical": int("mechanical" in hazards),
        "pressure": int("pressure" in hazards),
        "chemical": int("chemical" in hazards),
        "thermal": int("thermal" in hazards),
        "fire_explosion": int("fire_explosion" in hazards),
        "vehicle": int("vehicle" in hazards),
        "fall_height": int("fall_height" in hazards),
        "caught_between": int("caught_between" in hazards),
        "dropped_object": int(
            "stored_projectile" in hazards
            or "dropped_object" in hazards
        ),
    }

    exposure_presence = {
        "direct_contact": int("direct_contact" in exposures),
        "line_of_fire": int("line_of_fire" in exposures),
        "caught_between": int("caught_between" in exposures),
        "fall": int("fall" in exposures),
        "chemical": int("chemical" in exposures),
        "thermal": int("thermal" in exposures),
        "electrical": int("electrical" in exposures),
        "vehicle": int("vehicle" in exposures),
    }

    pathway_presence = {
        "struck_by_projectile": int(
            "struck_by_projectile" in pathways
        ),
        "caught_crush": int(
            "caught_crush" in pathways
        ),
        "fall_from_height": int(
            "fall_from_height" in pathways
        ),
        "electrical_contact": int(
            "electrical_contact" in pathways
        ),
        "chemical_release_exposure": int(
            "chemical_release_exposure" in pathways
        ),
        "thermal_contact": int(
            "thermal_contact" in pathways
        ),
        "vehicle_person_collision": int(
            "vehicle_person_collision" in pathways
        ),
    }

    hazard_count = sum(hazard_presence.values())
    exposure_count = sum(exposure_presence.values())
    pathway_count = sum(pathway_presence.values())

    values: dict[str, float] = {}

    for k, v in hazard_presence.items():
        values[f"path_hazard_{k}"] = float(v)

    for k, v in exposure_presence.items():
        values[f"path_exposure_{k}"] = float(v)

    for k, v in pathway_presence.items():
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
        hazard_count > 0
        and exposure_count > 0
        and pathway_count > 0
    )

    values["path_electrical_complete"] = float(
        hazard_presence["electrical"]
        and exposure_presence["electrical"]
        and pathway_presence["electrical_contact"]
    )

    values["path_chemical_complete"] = float(
        hazard_presence["chemical"]
        and exposure_presence["chemical"]
        and pathway_presence["chemical_release_exposure"]
    )

    values["path_fall_complete"] = float(
        hazard_presence["fall_height"]
        and exposure_presence["fall"]
        and pathway_presence["fall_from_height"]
    )

    values["path_vehicle_complete"] = float(
        hazard_presence["vehicle"]
        and exposure_presence["vehicle"]
        and pathway_presence["vehicle_person_collision"]
    )

    values["path_caught_complete"] = float(
        hazard_presence["caught_between"]
        and exposure_presence["caught_between"]
        and pathway_presence["caught_crush"]
    )

    values["path_line_fire"] = float(
        exposure_presence["line_of_fire"]
        and (
            pathway_presence["struck_by_projectile"]
            or hazard_presence["vehicle"]
            or hazard_presence["dropped_object"]
        )
    )

    vector = np.asarray(
        [values.get(name, 0.0) for name in PATHWAY_FEATURE_NAMES],
        dtype=np.float32,
    )

    return vector


def score(text: str, evidence: dict[str, Any]) -> float:
    package = load_package()
    model = package["model"]
    keep_mask = np.asarray(package["feature_keep_mask"], dtype=bool)
    embedding_dim = int(package["embedding_dimension"])

    if embedding_dim != 384:
        raise ValueError(
            f"Unexpected frozen embedding dimension: {embedding_dim}"
        )

    if keep_mask.size != 422:
        raise ValueError(
            f"Unexpected frozen feature schema: mask length={keep_mask.size}, "
            "expected 422."
        )

    embedding = encode(text)

    if embedding.shape[0] != embedding_dim:
        raise ValueError(
            f"Embedding shape mismatch: got {embedding.shape[0]}, "
            f"expected {embedding_dim}"
        )

    pathway = reconstruct_pathway_features(evidence)

    full = np.concatenate([embedding, pathway])

    if full.shape[0] != 422:
        raise ValueError(
            f"Inference vector mismatch: got {full.shape[0]}, expected 422."
        )

    X = full[keep_mask].reshape(1, -1)

    if X.shape[1] != int(getattr(model, "n_features_in_", X.shape[1])):
        raise ValueError(
            f"Model input mismatch: got {X.shape[1]}, "
            f"model expects {getattr(model, 'n_features_in_', 'unknown')}"
        )

    return float(model.predict_proba(X)[0, 1])


def lsr_candidates(text: str, evidence: dict[str, Any]) -> list[dict[str, Any]]:
    t = text.lower()
    hazards = str(evidence.get("hazards", ""))
    exposures = str(evidence.get("exposures", ""))
    pathways = str(evidence.get("pathways", ""))

    out = []

    for rule, terms in LSR_RULES.items():
        matched = sorted({term for term in terms if term in t})
        s = min(0.60, 0.20 * len(matched))

        if rule == "Driving" and (
            "vehicle" in hazards
            or "vehicle_exposure" in exposures
            or "vehicle_person_collision" in pathways
        ):
            s += 0.40

        elif rule == "Energy Isolation" and (
            "electrical" in hazards
            or "electrical_exposure" in exposures
            or "electrical_contact" in pathways
        ):
            s += 0.40

        elif rule == "Working at Height" and (
            "fall_height" in hazards
            or "fall_exposure" in exposures
            or "fall_from_height" in pathways
        ):
            s += 0.40

        elif rule == "Line of Fire" and (
            "line_of_fire" in exposures
            or "struck_by_projectile" in pathways
            or "vehicle_person_collision" in pathways
        ):
            s += 0.40

        elif rule == "Hot Work" and (
            "fire_explosion" in hazards
            or "oxyfuel" in t
            or "hot work" in t
        ):
            s += 0.40

        elif rule == "Confined Space" and (
            "confined space" in t
            or "tank" in t
            or "vessel" in t
        ):
            s += 0.40

        elif rule == "Safe Mechanical Lifting" and (
            "lifting" in t
            or "crane" in t
            or "suspended load" in t
        ):
            s += 0.40

        elif rule == "Bypassing Safety Controls" and (
            "bypass" in t
            or "override" in t
            or "disable" in t
            or "barrier" in t
        ):
            s += 0.40

        elif rule == "Work Authorisation" and (
            "permit" in t
            or "authorization" in t
            or "authorisation" in t
        ):
            s += 0.40

        s = min(1.0, s)

        if s > 0:
            out.append({
                "rule": rule,
                "candidate_score": round(s, 4),
                "matched_terms": matched[:8],
            })

    return sorted(
        out,
        key=lambda x: (-x["candidate_score"], x["rule"])
    )[:3]


def build_result(text: str) -> dict[str, Any]:
    evidence = analyze(text)
    probability = score(text, evidence)
    conflicts = []

    if (
        probability >= 0.75
        and evidence.get("sif_context_status")
        in {"UNKNOWN", "CONTEXT_LIMITED"}
    ):
        conflicts.append("HIGH_SCORE_MODEL_CONFLICT")

    if (
        probability < 0.20
        and evidence.get("pathway_signal")
        and evidence.get("exposure_signal")
    ):
        conflicts.append("MODEL_MISS_WITH_EXPLICIT_EVIDENCE")

    if (
        probability < 0.50
        and evidence.get("complete_pathway") == 1
    ):
        conflicts.append("LOW_SCORE_MODEL_MISS_WITH_EXPLICIT_PATHWAY")

    if conflicts:
        priority = "P1"
        decision = "HSSE_REVIEW_REQUIRED"
    elif probability >= 0.80:
        priority = "P2"
        decision = "PRIORITIZE_FOR_HSSE_REVIEW"
    elif probability >= 0.50:
        priority = "P3"
        decision = "PRIORITIZE_FOR_HSSE_REVIEW"
    elif evidence.get("complete_pathway") == 1:
        priority = "P3"
        decision = "PRIORITIZE_FOR_HSSE_REVIEW"
    elif evidence.get("pathway_signal_strength", 0) >= 1:
        priority = "P4"
        decision = "LOWER_PRIORITY_RETAIN_FOR_REVIEW"
    else:
        priority = "P5"
        decision = "LOWER_PRIORITY_RETAIN_FOR_REVIEW"

    return {
        "system": "RAKSHAK / SIF-Insight",
        "version": "1.1",
        "model": {
            "encoder": ENCODER_NAME,
            "sif_precursor_probability": round(probability, 6),
            "reference_threshold": 0.50,
            "interpretation": (
                "Ranking/triage score; not an autonomous SIF verdict."
            ),
        },
        "evidence": evidence,
        "lsr_candidates": lsr_candidates(text, evidence),
        "review": {
            "reviewer_priority": priority,
            "conflicts": conflicts,
            "decision": decision,
            "human_in_loop": True,
        },
        "governance": {
            "model_input": "Narrative description only",
            "counterfactual_policy": (
                "Do not invent missing severity, energy, distance, concentration, "
                "quantity, duration, exposure area or affected-person count."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--file")
    group.add_argument("--json", dest="json_file")
    parser.add_argument("--compact", action="store_true")
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

    result = build_result(text)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=None if args.compact else 2,
        )
    )


if __name__ == "__main__":
    main()
