"""
RAKSHAK / SIF-Insight EXACT INFERENCE v1.2
==========================================

Uses the SAME pathway feature builder as the frozen v1.2 training pipeline.
This is the required inference implementation for the current frozen model.

Run:
    python scripts\rakshak_inference_exact_v1_2.py --text "..."

The previous inference implementation reconstructed the 38 features from the
later Evidence Engine and therefore produced materially different scores.
This version imports build_pathway_features directly from
train_rakshak_final_pipeline_v1_2.py, guaranteeing schema/order/pattern parity.

The Evidence Engine v1.5 remains a separate reviewer/explainability layer.
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
    PROJECT_ROOT / "experiments" / "rakshak_final_v1.2" / "rakshak_final_model.joblib"
)

TRAINING_PIPELINE = SCRIPT_DIR / "train_rakshak_final_pipeline_v1_2.py"
ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

sys.path.insert(0, str(SCRIPT_DIR))

try:
    # Import the exact training-time feature builder.
    from train_rakshak_final_pipeline_v1_2 import (
        build_pathway_features,
    )
except Exception as exc:
    raise RuntimeError(
        f"Could not import exact v1.2 feature builder from "
        f"{TRAINING_PIPELINE}: {exc}"
    ) from exc

try:
    from evidence_engine_v1_5 import analyze
except Exception as exc:
    raise RuntimeError(f"Could not import evidence engine v1.5: {exc}") from exc


def load_package() -> dict[str, Any]:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Frozen model package not found:\n{MODEL_FILE}")
    return joblib.load(MODEL_FILE)


def encode_text(text: str) -> np.ndarray:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = SentenceTransformer(
        ENCODER_NAME,
        device=device,
    )
    encoder.max_seq_length = 256

    emb = encoder.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
        device=device,
    ).astype(np.float32)

    return emb[0]


def build_exact_feature_vector(
    text: str,
    package: dict[str, Any],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    embedding_dim = int(package["embedding_dimension"])
    keep_mask = np.asarray(
        package["feature_keep_mask"],
        dtype=bool,
    )

    # Exact training-time pathway builder.
    pathway_dict = build_pathway_features(text)

    pathway_df_features = list(pathway_dict.keys())

    # Pandas DataFrame construction in training preserved dict insertion order.
    pathway_vector = np.asarray(
        [pathway_dict[k] for k in pathway_df_features],
        dtype=np.float32,
    )

    embedding = encode_text(text)

    if embedding.shape[0] != embedding_dim:
        raise ValueError(f"Embedding mismatch: {embedding.shape[0]} != {embedding_dim}")

    if pathway_vector.shape[0] != 38:
        raise ValueError(
            f"Exact pathway builder produced {pathway_vector.shape[0]} "
            "features; frozen v1.2 expects 38."
        )

    full = np.concatenate([embedding, pathway_vector])

    if full.shape[0] != 422:
        raise ValueError(f"Full feature vector is {full.shape[0]}, expected 422.")

    if keep_mask.size != 422:
        raise ValueError(
            f"Frozen feature mask has {keep_mask.size} entries, expected 422."
        )

    X = full[keep_mask].reshape(1, -1)

    return X, pathway_df_features, pathway_dict


def lsr_candidates(
    text: str,
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Reviewer-only IOGP LSR candidate mapper.
    It does not affect the classifier score.
    """

    t = text.lower()
    hazards = str(evidence.get("hazards", ""))
    exposures = str(evidence.get("exposures", ""))
    pathways = str(evidence.get("pathways", ""))

    rules = {
        "Bypassing Safety Controls": [
            "bypass",
            "override",
            "disable",
            "safety control",
            "barrier",
        ],
        "Confined Space": [
            "confined space",
            "tank",
            "vessel",
            "inside tank",
            "inside vessel",
        ],
        "Driving": ["vehicle", "truck", "driver", "driving", "collision", "run over"],
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
        ],
        "Work Authorisation": ["permit", "authorization", "authorisation"],
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

    candidates = []

    for rule, terms in rules.items():
        matched = sorted({term for term in terms if term in t})

        score = min(0.60, 0.20 * len(matched))

        if rule == "Driving" and (
            "vehicle" in hazards
            or "vehicle_exposure" in exposures
            or "vehicle_person_collision" in pathways
        ):
            score += 0.40

        elif rule == "Energy Isolation" and (
            "electrical" in hazards
            or "electrical_exposure" in exposures
            or "electrical_contact" in pathways
        ):
            score += 0.40

        elif rule == "Working at Height" and (
            "fall_height" in hazards
            or "fall_exposure" in exposures
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
            "fire_explosion" in hazards or "oxyfuel" in t or "hot work" in t
        ):
            score += 0.40

        elif rule == "Confined Space" and (
            "confined space" in t or "tank" in t or "vessel" in t
        ):
            score += 0.40

        elif rule == "Safe Mechanical Lifting" and (
            "lifting" in t or "crane" in t or "suspended load" in t
        ):
            score += 0.40

        elif rule == "Bypassing Safety Controls" and (
            "bypass" in t or "override" in t or "disable" in t or "barrier" in t
        ):
            score += 0.40

        elif rule == "Work Authorisation" and (
            "permit" in t or "authorization" in t or "authorisation" in t
        ):
            score += 0.40

        score = min(1.0, score)

        if score > 0:
            candidates.append(
                {
                    "rule": rule,
                    "candidate_score": round(score, 4),
                    "matched_terms": matched[:8],
                }
            )

    return sorted(candidates, key=lambda x: (-x["candidate_score"], x["rule"]))[:3]


def infer(text: str) -> dict[str, Any]:
    text = str(text).strip()

    if not text:
        raise ValueError("Narrative cannot be empty.")

    package = load_package()

    model = package["model"]

    X, pathway_names, pathway_values = build_exact_feature_vector(
        text,
        package,
    )

    expected_features = getattr(
        model,
        "n_features_in_",
        None,
    )

    if expected_features is not None and X.shape[1] != expected_features:
        raise ValueError(
            f"Model input mismatch: X={X.shape[1]}, " f"model={expected_features}"
        )

    probability = float(model.predict_proba(X)[0, 1])

    # Evidence is deliberately computed independently of model scoring.
    evidence = analyze(text)
    lsr = lsr_candidates(text, evidence)

    conflicts = []

    if probability >= 0.75 and evidence.get("sif_context_status") in {
        "UNKNOWN",
        "CONTEXT_LIMITED",
    }:
        conflicts.append("HIGH_SCORE_MODEL_CONFLICT")

    if (
        probability < 0.20
        and evidence.get("pathway_signal")
        and evidence.get("exposure_signal")
    ):
        conflicts.append("MODEL_MISS_WITH_EXPLICIT_EVIDENCE")

    if probability < 0.50 and evidence.get("complete_pathway") == 1:
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
        "version": "1.2-exact",
        "model": {
            "encoder": ENCODER_NAME,
            "embedding_dimension": 384,
            "pathway_features": 38,
            "total_features_before_mask": 422,
            "sif_precursor_probability": round(probability, 6),
            "reference_threshold": 0.50,
            "interpretation": ("Ranking/triage score; not an autonomous SIF verdict."),
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
            "model_input": "Incident narrative only",
            "feature_builder": "Exact frozen v1.2 training-time builder",
            "counterfactual_policy": (
                "Do not invent missing severity, energy, distance, concentration, "
                "quantity, duration, exposure area or affected-person count."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--file")
    group.add_argument("--json", dest="json_file")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    if args.text is not None:
        text = args.text
    elif args.file is not None:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        payload = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
        text = payload["description"]

    result = infer(text)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=None if args.compact else 2,
        )
    )


if __name__ == "__main__":
    main()
