"""
RAKSHAK / SIF-Insight FastAPI backend v1.1

Adds browser CORS support for the reviewer dashboard.
Frozen ML/evidence/reviewer components are unchanged.

Run:
    uvicorn scripts.rakshak_api_v1_1:app --host 127.0.0.1 --port 8000

Endpoints:
    GET  /health
    GET  /model-info
    POST /predict
    POST /predict/batch
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
        "The score is a ranking/triage signal for HSSE review, "
        "not an autonomous safety decision."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    report_id: str | None = None
    description: str = Field(min_length=1)


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
        _ENCODER = SentenceTransformer(MODEL_NAME, device=_DEVICE)
        _ENCODER.max_seq_length = MAX_SEQ_LENGTH

    return _MODEL_PACKAGE, _MODEL, _ENCODER, _DEVICE


def build_features(texts: list[str]) -> np.ndarray:
    package, model, encoder, device = load_resources()

    embeddings = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
        device=device,
    ).astype(np.float32)

    embedding_dim = int(package["embedding_dimension"])
    if embeddings.shape != (len(texts), embedding_dim):
        raise RuntimeError(f"Embedding schema mismatch: {embeddings.shape}")

    names = None
    pathway_rows = []
    for text in texts:
        d = build_pathway_features(text)
        current_names = list(d.keys())
        if names is None:
            names = current_names
        elif current_names != names:
            raise RuntimeError("Pathway feature ordering changed.")
        pathway_rows.append([d[k] for k in names])

    pathway = np.asarray(pathway_rows, dtype=np.float32)
    if pathway.shape != (len(texts), 38):
        raise RuntimeError(f"Pathway schema mismatch: {pathway.shape}")

    full = np.hstack([embeddings, pathway])
    if full.shape[1] != 422:
        raise RuntimeError(f"Raw feature schema mismatch: {full.shape[1]}")

    keep = np.asarray(package["feature_keep_mask"], dtype=bool)
    if keep.size != 422:
        raise RuntimeError(f"Feature mask mismatch: {keep.size}")

    X = full[:, keep]

    expected = getattr(model, "n_features_in_", None)
    if expected is not None and X.shape[1] != expected:
        raise RuntimeError(f"Model input mismatch: {X.shape[1]} != {expected}")

    return X


def build_lsr_candidates(text: str, evidence: dict[str, Any]):
    t = text.lower()
    hazards = str(evidence.get("hazards", ""))
    exposures = str(evidence.get("exposures", ""))
    pathways = str(evidence.get("pathways", ""))

    rules = {
        "Bypassing Safety Controls": ["bypass", "override", "disable", "safety control", "barrier"],
        "Confined Space": ["confined space", "tank", "vessel"],
        "Driving": ["vehicle", "truck", "driver", "driving", "collision", "forklift", "excavator", "loader"],
        "Energy Isolation": ["energized", "electric", "voltage", "isolation", "lockout", "tagout", "de-energized", "stored energy"],
        "Hot Work": ["welding", "cutting", "grinding", "hot work", "ignition", "oxyfuel"],
        "Line of Fire": ["line of fire", "struck", "projectile", "dropped object", "moving object", "pressure release"],
        "Safe Mechanical Lifting": ["lifting", "lifted", "suspended load", "crane", "rigging"],
        "Work Authorisation": ["permit", "authorization", "authorisation"],
        "Working at Height": ["height", "fall", "scaffold", "ladder", "tower", "platform"],
    }

    scores = []
    for rule, terms in rules.items():
        matched = sorted({term for term in terms if term in t})
        score = min(0.60, 0.20 * len(matched))

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
            "line_of_fire" in exposures or "struck_by_projectile" in pathways
            or "vehicle_person_collision" in pathways
        ):
            score += 0.40
        elif rule == "Hot Work" and (
            "fire_explosion" in hazards or "hot work" in t or "oxyfuel" in t
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
            scores.append({
                "rule": rule,
                "candidate_score": round(score, 4),
                "matched_terms": matched[:8],
            })

    return sorted(scores, key=lambda x: (-x["candidate_score"], x["rule"]))[:3]


def predict_records(records: list[BatchItem]):
    texts = [r.description.strip() for r in records]
    X = build_features(texts)
    probabilities = _MODEL.predict_proba(X)[:, 1]

    results = []
    for item, text, probability in zip(records, texts, probabilities):
        probability = float(probability)
        evidence = analyze(text)

        row = {
            "sif_precursor_probability": probability,
            "hazard_signal": evidence.get("hazard_signal", 0),
            "exposure_signal": evidence.get("exposure_signal", 0),
            "pathway_signal": evidence.get("pathway_signal", 0),
            "complete_pathway": evidence.get("complete_pathway", 0),
        }
        priority, reason = policy(row)

        results.append({
            "report_id": item.report_id,
            "sif_precursor_probability": round(probability, 6),
            "prediction_at_0_50": bool(probability >= 0.50),
            "review": {
                "priority": priority,
                "reason": reason,
                "human_review_required": True,
            },
            "evidence": {
                "hazards": evidence.get("hazards", ""),
                "exposures": evidence.get("exposures", ""),
                "pathways": evidence.get("pathways", ""),
                "hazard_evidence": evidence.get("hazard_evidence", ""),
                "exposure_evidence": evidence.get("exposure_evidence", ""),
                "pathway_evidence": evidence.get("pathway_evidence", ""),
                "hazard_signal": evidence.get("hazard_signal", 0),
                "exposure_signal": evidence.get("exposure_signal", 0),
                "pathway_signal": evidence.get("pathway_signal", 0),
                "pathway_signal_strength": evidence.get("pathway_signal_strength", 0),
                "complete_pathway": evidence.get("complete_pathway", 0),
                "sif_context_status": evidence.get("sif_context_status", "UNKNOWN"),
                "pathway_assessment": evidence.get("pathway_assessment", ""),
            },
            "lsr_candidates": build_lsr_candidates(text, evidence),
            "governance": {
                "model_input": "Narrative description only",
                "autonomous_sif_decision": False,
                "requires_hsse_review": True,
            },
        })

    return results


@app.get("/health")
def health():
    try:
        _, model, _, device = load_resources()
        return {
            "status": "ok",
            "service": "RAKSHAK / SIF-Insight",
            "api_version": app.version,
            "model_version": "rakshak_final_v1.2",
            "review_policy": "v1.4",
            "evidence_engine": "v1.5",
            "device": device,
            "model_input_dimension": int(model.n_features_in_),
            "human_in_loop": True,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/model-info")
def model_info():
    try:
        package, model, _, device = load_resources()
        return {
            "system": "RAKSHAK / SIF-Insight",
            "api_version": app.version,
            "model_version": "rakshak_final_v1.2",
            "encoder": MODEL_NAME,
            "embedding_dimension": int(package["embedding_dimension"]),
            "pathway_features": 38,
            "raw_features": 422,
            "model_input_dimension": int(model.n_features_in_),
            "reference_threshold": float(package.get("reference_threshold", 0.50)),
            "review_policy": "v1.4",
            "evidence_engine": "v1.5",
            "device": device,
            "training_records": 70,
            "locked_test_records": 20,
            "purpose": "HSSE SIF-precursor triage and reviewer prioritization",
            "autonomous_decision": False,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        return predict_records([
            BatchItem(
                report_id=request.report_id,
                description=request.description,
            )
        ])[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    try:
        return {"records": predict_records(request.records)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
