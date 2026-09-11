"""
RAKSHAK / SIF-Insight FastAPI backend v1.0

Frozen components:
- Final train+validation model: experiments/rakshak_final_v1.2/rakshak_final_model.joblib
- Exact v1.2 pathway feature builder
- Evidence Engine v1.5
- Reviewer Policy v1.4

No retraining occurs in this service.

Run from project root:
    uvicorn scripts.rakshak_api_v1_0:app --host 127.0.0.1 --port 8000

Endpoints:
    GET  /health
    GET  /model-info
    POST /predict
    POST /predict/batch
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_DIR = PROJECT_ROOT / "experiments" / "rakshak_final_v1.2"
MODEL_FILE = MODEL_DIR / "rakshak_final_model.joblib"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 8

sys.path.insert(0, str(SCRIPT_DIR))

from train_rakshak_final_pipeline_v1_2 import build_pathway_features
from evidence_engine_v1_5 import analyze
from rakshak_reviewer_policy_v1_4 import policy


app = FastAPI(
    title="RAKSHAK / SIF-Insight API",
    description=(
        "AI-powered SIF precursor triage API. "
        "The model produces a ranking/triage signal for HSSE review; "
        "it is not an autonomous safety decision system."
    ),
    version="1.0.0",
)


class PredictRequest(BaseModel):
    report_id: str | None = Field(
        default=None,
        description="Optional source report identifier.",
    )
    description: str = Field(
        min_length=1,
        description="Incident / unsafe-act / unsafe-condition narrative.",
    )


class BatchItem(BaseModel):
    report_id: str | None = None
    description: str = Field(min_length=1)


class BatchRequest(BaseModel):
    records: list[BatchItem] = Field(min_length=1, max_length=500)


_MODEL_PACKAGE: dict[str, Any] | None = None
_MODEL = None
_ENCODER: SentenceTransformer | None = None
_DEVICE: str | None = None


def load_resources():
    global _MODEL_PACKAGE, _MODEL, _ENCODER, _DEVICE

    if _MODEL_PACKAGE is None:
        if not MODEL_FILE.exists():
            raise RuntimeError(f"Frozen model not found: {MODEL_FILE}")

        _MODEL_PACKAGE = joblib.load(MODEL_FILE)
        _MODEL = _MODEL_PACKAGE["model"]

        expected = getattr(_MODEL, "n_features_in_", None)
        if expected != 422:
            raise RuntimeError(
                f"Frozen model schema mismatch: expected 422, got {expected}"
            )

    if _ENCODER is None:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _ENCODER = SentenceTransformer(
            MODEL_NAME,
            device=_DEVICE,
        )
        _ENCODER.max_seq_length = MAX_SEQ_LENGTH

    return _MODEL_PACKAGE, _MODEL, _ENCODER, _DEVICE


def build_features(texts: list[str]) -> np.ndarray:
    package, _, encoder, device = load_resources()

    embeddings = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
        device=device,
    ).astype(np.float32)

    expected_embedding_dim = int(package["embedding_dimension"])
    if embeddings.shape != (len(texts), expected_embedding_dim):
        raise RuntimeError(
            f"Embedding schema mismatch: {embeddings.shape}"
        )

    names = None
    path_rows = []

    for text in texts:
        features = build_pathway_features(text)
        current_names = list(features.keys())

        if names is None:
            names = current_names
        elif current_names != names:
            raise RuntimeError("Pathway feature ordering changed.")

        path_rows.append([features[k] for k in names])

    pathway = np.asarray(path_rows, dtype=np.float32)

    if pathway.shape != (len(texts), 38):
        raise RuntimeError(
            f"Pathway schema mismatch: {pathway.shape}"
        )

    full = np.hstack([embeddings, pathway])

    if full.shape[1] != 422:
        raise RuntimeError(
            f"Raw feature schema mismatch: {full.shape[1]}"
        )

    keep = np.asarray(package["feature_keep_mask"], dtype=bool)

    if keep.size != 422:
        raise RuntimeError(
            f"Feature mask mismatch: {keep.size}"
        )

    X = full[:, keep]

    expected_model_features = getattr(
        _MODEL,
        "n_features_in_",
        None,
    )

    if expected_model_features is not None and X.shape[1] != expected_model_features:
        raise RuntimeError(
            f"Model input mismatch: {X.shape[1]} != {expected_model_features}"
        )

    return X


def predict_records(records: list[BatchItem]) -> list[dict[str, Any]]:
    texts = [r.description.strip() for r in records]
    X = build_features(texts)

    probabilities = _MODEL.predict_proba(X)[:, 1]

    results = []

    for item, text, probability in zip(
        records,
        texts,
        probabilities,
    ):
        probability = float(probability)

        evidence = analyze(text)

        # v1.4 reviewer policy operates on the same model/evidence columns.
        policy_row = {
            "sif_precursor_probability": probability,
            "hazard_signal": evidence.get("hazard_signal", 0),
            "exposure_signal": evidence.get("exposure_signal", 0),
            "pathway_signal": evidence.get("pathway_signal", 0),
            "complete_pathway": evidence.get("complete_pathway", 0),
        }

        reviewer_priority, reviewer_reason = policy(
            policy_row
        )

        lsr_candidates = build_lsr_candidates(
            text,
            evidence,
        )

        results.append(
            {
                "report_id": item.report_id,
                "sif_precursor_probability": round(probability, 6),
                "prediction_at_0_50": bool(probability >= 0.50),

                "review": {
                    "priority": reviewer_priority,
                    "reason": reviewer_reason,
                    "human_review_required": True,
                },

                "evidence": {
                    "hazards": evidence.get("hazards", ""),
                    "exposures": evidence.get("exposures", ""),
                    "pathways": evidence.get("pathways", ""),
                    "hazard_evidence": evidence.get(
                        "hazard_evidence", ""
                    ),
                    "exposure_evidence": evidence.get(
                        "exposure_evidence", ""
                    ),
                    "pathway_evidence": evidence.get(
                        "pathway_evidence", ""
                    ),
                    "hazard_signal": evidence.get(
                        "hazard_signal", 0
                    ),
                    "exposure_signal": evidence.get(
                        "exposure_signal", 0
                    ),
                    "pathway_signal": evidence.get(
                        "pathway_signal", 0
                    ),
                    "pathway_signal_strength": evidence.get(
                        "pathway_signal_strength", 0
                    ),
                    "complete_pathway": evidence.get(
                        "complete_pathway", 0
                    ),
                    "sif_context_status": evidence.get(
                        "sif_context_status",
                        "UNKNOWN",
                    ),
                    "pathway_assessment": evidence.get(
                        "pathway_assessment",
                        "",
                    ),
                },

                "lsr_candidates": lsr_candidates,

                "governance": {
                    "model_input": "Narrative description only",
                    "autonomous_sif_decision": False,
                    "requires_hsse_review": True,
                },
            }
        )

    return results


def build_lsr_candidates(
    text: str,
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    t = text.lower()

    hazards = str(evidence.get("hazards", ""))
    exposures = str(evidence.get("exposures", ""))
    pathways = str(evidence.get("pathways", ""))

    rules = {
        "Bypassing Safety Controls": [
            "bypass", "override", "disable", "safety control", "barrier"
        ],
        "Confined Space": [
            "confined space", "tank", "vessel"
        ],
        "Driving": [
            "vehicle", "truck", "driver", "driving",
            "collision", "forklift", "excavator", "loader"
        ],
        "Energy Isolation": [
            "energized", "electric", "voltage", "isolation",
            "lockout", "tagout", "de-energized", "stored energy"
        ],
        "Hot Work": [
            "welding", "cutting", "grinding",
            "hot work", "ignition", "oxyfuel"
        ],
        "Line of Fire": [
            "line of fire", "struck", "projectile",
            "dropped object", "moving object",
            "pressure release"
        ],
        "Safe Mechanical Lifting": [
            "lifting", "lifted", "suspended load",
            "crane", "rigging"
        ],
        "Work Authorisation": [
            "permit", "authorization", "authorisation"
        ],
        "Working at Height": [
            "height", "fall", "scaffold",
            "ladder", "tower", "platform"
        ],
    }

    scores = []

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
            "fire_explosion" in hazards
            or "hot work" in t
            or "oxyfuel" in t
        ):
            score += 0.40

        elif rule == "Confined Space" and (
            "confined space" in t
            or "tank" in t
            or "vessel" in t
        ):
            score += 0.40

        elif rule == "Safe Mechanical Lifting" and (
            "lifting" in t
            or "crane" in t
            or "suspended load" in t
        ):
            score += 0.40

        elif rule == "Bypassing Safety Controls" and (
            "bypass" in t
            or "override" in t
            or "disable" in t
            or "barrier" in t
        ):
            score += 0.40

        elif rule == "Work Authorisation" and (
            "permit" in t
            or "authorization" in t
            or "authorisation" in t
        ):
            score += 0.40

        score = min(1.0, score)

        if score > 0:
            scores.append(
                {
                    "rule": rule,
                    "candidate_score": round(score, 4),
                    "matched_terms": matched[:8],
                }
            )

    return sorted(
        scores,
        key=lambda x: (-x["candidate_score"], x["rule"]),
    )[:3]


@app.get("/health")
def health():
    try:
        package, model, _, device = load_resources()

        return {
            "status": "ok",
            "service": "RAKSHAK / SIF-Insight",
            "api_version": "1.0.0",
            "model_version": "rakshak_final_v1.2",
            "review_policy": "v1.4",
            "evidence_engine": "v1.5",
            "device": device,
            "model_input_dimension": int(
                getattr(model, "n_features_in_", 0)
            ),
            "human_in_loop": True,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )


@app.get("/model-info")
def model_info():
    try:
        package, model, _, device = load_resources()

        return {
            "system": "RAKSHAK / SIF-Insight",
            "model_version": "rakshak_final_v1.2",
            "encoder": MODEL_NAME,
            "embedding_dimension": int(
                package["embedding_dimension"]
            ),
            "pathway_features": 38,
            "raw_features": 422,
            "model_input_dimension": int(
                getattr(model, "n_features_in_", 0)
            ),
            "reference_threshold": float(
                package.get("reference_threshold", 0.50)
            ),
            "review_policy": "v1.4",
            "evidence_engine": "v1.5",
            "device": device,
            "training_records": 70,
            "locked_test_records": 20,
            "purpose": "HSSE SIF-precursor triage and reviewer prioritization",
            "autonomous_decision": False,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        result = predict_records(
            [
                BatchItem(
                    report_id=request.report_id,
                    description=request.description,
                )
            ]
        )
        return result[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    try:
        return {
            "records": predict_records(request.records)
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
