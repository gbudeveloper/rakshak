"""
RAKSHAK / SIF-Insight batch inference v1.2

Purpose:
- Load the frozen final train+validation model once.
- Load the SentenceTransformer encoder once.
- Use the exact v1.2 pathway feature builder.
- Score an entire CSV in one batch.
- Attach Evidence Engine v1.5 reviewer signals.
- Never let evidence/LSR override the ML probability.

Input CSV requirements:
- `report_id` is recommended.
- `description` is required.

Example:
    python scripts\rakshak_batch_inference_v1_2.py ^
      --input data\annotations\industrial_safety_annotation_pool.csv ^
      --output experiments\batch_predictions.csv

The output is a reviewer-facing triage artifact, not an autonomous safety decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
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


def load_model_package():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}")
    return joblib.load(MODEL_FILE)


def load_encoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading encoder once on {device}: {MODEL_NAME}")
    encoder = SentenceTransformer(MODEL_NAME, device=device)
    encoder.max_seq_length = MAX_SEQ_LENGTH
    return encoder, device


def build_feature_matrix(texts, package, encoder, device):
    embeddings = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
        device=device,
    ).astype(np.float32)

    expected_embedding_dim = int(package["embedding_dimension"])
    if embeddings.shape[1] != expected_embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"{embeddings.shape[1]} != {expected_embedding_dim}"
        )

    names = None
    pathway_rows = []

    for text in texts:
        features = build_pathway_features(text)
        current_names = list(features.keys())

        if names is None:
            names = current_names
        elif current_names != names:
            raise ValueError("Pathway feature ordering changed.")

        pathway_rows.append([features[k] for k in names])

    pathway = np.asarray(pathway_rows, dtype=np.float32)

    if pathway.shape[1] != 38:
        raise ValueError(
            f"Pathway feature count mismatch: {pathway.shape[1]} != 38"
        )

    full = np.hstack([embeddings, pathway])

    if full.shape[1] != 422:
        raise ValueError(
            f"Total raw feature count mismatch: {full.shape[1]} != 422"
        )

    keep = np.asarray(package["feature_keep_mask"], dtype=bool)
    if keep.size != 422:
        raise ValueError(
            f"Feature mask size mismatch: {keep.size} != 422"
        )

    return full[:, keep]


def classify_priority(probability: float, evidence: dict) -> tuple[str, str]:
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
        return "P1", "|".join(conflicts)

    if probability >= 0.80:
        return "P2", ""

    if probability >= 0.50:
        return "P3", ""

    if evidence.get("complete_pathway") == 1:
        return "P3", ""

    if evidence.get("pathway_signal_strength", 0) >= 1:
        return "P4", ""

    return "P5", ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--id-column",
        default="report_id",
        help="ID column; created automatically when absent.",
    )
    parser.add_argument(
        "--text-column",
        default="description",
        help="Narrative column.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)

    if args.text_column not in df.columns:
        raise ValueError(
            f"Required narrative column '{args.text_column}' is missing. "
            f"Available columns: {list(df.columns)}"
        )

    if args.id_column not in df.columns:
        df[args.id_column] = [
            f"input_{i:06d}" for i in range(len(df))
        ]

    df[args.id_column] = df[args.id_column].astype(str)
    texts = df[args.text_column].fillna("").astype(str).tolist()

    if not texts:
        raise ValueError("Input CSV contains zero rows.")

    package = load_model_package()
    model = package["model"]

    encoder, device = load_encoder()
    X = build_feature_matrix(
        texts,
        package,
        encoder,
        device,
    )

    expected = getattr(model, "n_features_in_", None)
    if expected is not None and X.shape[1] != expected:
        raise ValueError(
            f"Model expects {expected} features, got {X.shape[1]}"
        )

    probabilities = model.predict_proba(X)[:, 1]

    rows = []

    for report_id, text, probability in zip(
        df[args.id_column].tolist(),
        texts,
        probabilities,
    ):
        evidence = analyze(text)
        priority, conflicts = classify_priority(
            float(probability),
            evidence,
        )

        rows.append({
            args.id_column: report_id,
            "sif_precursor_probability": float(probability),
            "prediction_at_0_50": int(probability >= 0.50),

            "review_priority": priority,
            "model_evidence_conflicts": conflicts,

            "hazards": evidence.get("hazards", ""),
            "exposures": evidence.get("exposures", ""),
            "pathways": evidence.get("pathways", ""),
            "hazard_evidence": evidence.get("hazard_evidence", ""),
            "exposure_evidence": evidence.get("exposure_evidence", ""),
            "pathway_evidence": evidence.get("pathway_evidence", ""),

            "hazard_signal": evidence.get("hazard_signal", 0),
            "exposure_signal": evidence.get("exposure_signal", 0),
            "pathway_signal": evidence.get("pathway_signal", 0),
            "pathway_signal_strength": evidence.get(
                "pathway_signal_strength", 0
            ),
            "complete_pathway": evidence.get(
                "complete_pathway", 0
            ),

            "sif_context_status": evidence.get(
                "sif_context_status", "UNKNOWN"
            ),
            "sif_context_reason": evidence.get(
                "sif_context_reason", ""
            ),
            "pathway_assessment": evidence.get(
                "pathway_assessment", ""
            ),

            # Preserve the original narrative for direct reviewer access.
            "description": text,
        })

    result = pd.DataFrame(rows)

    # Deterministic reviewer ordering.
    priority_order = {
        "P1": 1,
        "P2": 2,
        "P3": 3,
        "P4": 4,
        "P5": 5,
    }

    result["_priority_rank"] = result["review_priority"].map(
        priority_order
    )
    result = result.sort_values(
        ["_priority_rank", "sif_precursor_probability"],
        ascending=[True, False],
    ).drop(columns="_priority_rank")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "system": "RAKSHAK / SIF-Insight",
        "pipeline": "v1.2-final-refit-batch-inference",
        "model_file": str(MODEL_FILE),
        "input_file": str(input_path),
        "output_file": str(output_path),
        "records": int(len(result)),
        "embedding_dimension": 384,
        "pathway_dimension": 38,
        "raw_dimension": 422,
        "model_input_dimension": int(X.shape[1]),
        "reference_threshold": 0.50,
        "device": device,
        "human_in_loop": True,
        "autonomous_sif_decision": False,
        "note": (
            "Probability is a ranking/triage signal. Evidence and LSR "
            "signals are reviewer support and do not override the model."
        ),
    }

    summary_path = output_path.with_name(
        output_path.stem + "_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("RAKSHAK BATCH INFERENCE v1.2")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print("\nReviewer queue preview:")
    print(
        result[
            [
                args.id_column,
                "sif_precursor_probability",
                "prediction_at_0_50",
                "review_priority",
                "model_evidence_conflicts",
                "hazards",
                "exposures",
                "pathways",
            ]
        ].head(20).to_string(index=False)
    )
    print("\nSaved:")
    print(output_path)
    print(summary_path)


if __name__ == "__main__":
    main()
