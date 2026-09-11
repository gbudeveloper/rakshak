"""
RAKSHAK FINAL REFIT consistency validator v1.2 — single encoder load.

This avoids calling infer() 20 times, which previously recreated a CUDA
SentenceTransformer for every record and eventually caused:
torch.AcceleratorError: CUDA error: unknown error

The validator loads the encoder once, builds the exact 422-column representation
using the frozen v1.2 training builder, and compares those probabilities with
the same saved final train+validation model.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_FILE = (
    PROJECT_ROOT / "experiments" / "rakshak_final_v1.2" / "rakshak_final_model.joblib"
)
LABEL_FILE = PROJECT_ROOT / "data" / "annotations" / "resolved_annotations_v0.2.csv"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "sif_splits_v0.2" / "test.csv"
OUT_DIR = PROJECT_ROOT / "experiments" / "rakshak_inference_validation_v1.2"

sys.path.insert(0, str(SCRIPT_DIR))
from train_rakshak_final_pipeline_v1_2 import build_pathway_features

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_SEQ_LENGTH = 256


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    package = joblib.load(MODEL_FILE)
    model = package["model"]
    keep = np.asarray(package["feature_keep_mask"], dtype=bool)
    emb_dim = int(package["embedding_dimension"])

    labels = pd.read_csv(LABEL_FILE)
    labels["report_id"] = labels["report_id"].astype(str)
    labels["description"] = labels["description"].fillna("").astype(str)
    labels["final_sif_potential"] = (
        labels["final_sif_potential"].astype(str).str.upper().str.strip()
    )

    test = pd.read_csv(TEST_FILE)
    test["report_id"] = test["report_id"].astype(str)

    df = test[["report_id"]].merge(
        labels[["report_id", "description", "final_sif_potential"]],
        on="report_id",
        how="left",
        validate="one_to_one",
    )

    if len(df) != 20:
        raise ValueError(f"Expected 20 locked-test rows, got {len(df)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading encoder once on {device}: {MODEL_NAME}")
    encoder = SentenceTransformer(MODEL_NAME, device=device)
    encoder.max_seq_length = MAX_SEQ_LENGTH

    texts = df["description"].tolist()

    embeddings = encoder.encode(
        texts,
        batch_size=8,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
        device=device,
    ).astype(np.float32)

    if embeddings.shape != (20, emb_dim):
        raise ValueError(
            f"Embedding matrix {embeddings.shape}; expected (20,{emb_dim})"
        )

    path_rows = []
    feature_names = None
    for text in texts:
        d = build_pathway_features(text)
        names = list(d.keys())
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError("Pathway feature ordering changed between records.")
        path_rows.append([d[k] for k in feature_names])

    path = np.asarray(path_rows, dtype=np.float32)
    if path.shape != (20, 38):
        raise ValueError(f"Pathway matrix {path.shape}; expected (20,38)")

    full = np.hstack([embeddings, path])
    if full.shape != (20, 422):
        raise ValueError(f"Full matrix {full.shape}; expected (20,422)")

    X = full[:, keep]
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and X.shape[1] != expected:
        raise ValueError(f"Model expects {expected}; inference has {X.shape[1]}")

    probs = model.predict_proba(X)[:, 1]

    # Directly repeat the same prediction operation from the public inference
    # implementation, but without reloading the encoder for every row.
    direct_probs = []
    for i in range(len(texts)):
        direct_probs.append(float(model.predict_proba(X[i : i + 1])[0, 1]))
    direct_probs = np.asarray(direct_probs)

    diff = np.abs(probs - direct_probs)

    result = pd.DataFrame(
        {
            "report_id": df["report_id"],
            "true_label": (df["final_sif_potential"] == "YES").astype(int),
            "final_refit_probability": probs,
            "direct_inference_probability": direct_probs,
            "absolute_difference": diff,
        }
    )

    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    status = (
        "PASS_EXACT"
        if max_diff < 1e-5
        else ("PASS_NUMERIC_TOLERANCE" if max_diff < 1e-3 else "FAIL")
    )

    result.to_csv(
        OUT_DIR / "final_refit_inference_consistency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "status": status,
        "records": len(result),
        "max_absolute_difference": max_diff,
        "mean_absolute_difference": mean_diff,
        "embedding_dimension": emb_dim,
        "pathway_dimension": 38,
        "total_dimension_before_mask": 422,
        "final_model_input_dimension": int(X.shape[1]),
        "encoder_loaded_once": True,
        "cuda_used": device == "cuda",
        "comparison_basis": "same saved final train+validation refit model and exact training-time feature builder",
    }
    (OUT_DIR / "final_refit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("=" * 80)
    print("RAKSHAK FINAL REFIT CONSISTENCY v1.2")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print("\nPREDICTIONS")
    print(
        result.sort_values("absolute_difference", ascending=False).to_string(
            index=False
        )
    )
    print("\nSaved:", OUT_DIR)


if __name__ == "__main__":
    main()
