"""
Validate exact v1.2 inference against the 20 locked-test probabilities.

Run:
    python scripts\validate_rakshak_inference_exact_v1_2.py

PASS_EXACT is expected because the same build_pathway_features() function
used during v1.2 training is imported during inference.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_PRED = (
    PROJECT_ROOT
    / "experiments"
    / "rakshak_final_v1.2"
    / "locked_test_predictions_with_evidence.csv"
)
LABELS = PROJECT_ROOT / "data" / "annotations" / "resolved_annotations_v0.2.csv"
TEST = PROJECT_ROOT / "data" / "processed" / "sif_splits_v0.2" / "test.csv"

sys.path.insert(0, str(SCRIPT_DIR))
from train_rakshak_final_pipeline_v1_2 import build_pathway_features
from rakshak_inference_exact_v1_2 import load_package


def main():
    package = load_package()
    model = package["model"]
    keep = np.asarray(package["feature_keep_mask"], dtype=bool)
    emb_dim = int(package["embedding_dimension"])

    saved = pd.read_csv(MODEL_PRED)
    labels = pd.read_csv(LABELS)
    test = pd.read_csv(TEST)
    for df in (saved, labels, test):
        df["report_id"] = df["report_id"].astype(str)

    data = (
        test[["report_id"]]
        .merge(
            labels[["report_id", "description", "final_sif_potential"]],
            on="report_id",
            how="left",
        )
        .merge(
            saved[["report_id", "sif_precursor_probability"]],
            on="report_id",
            how="left",
        )
    )
    if len(data) != 20:
        raise ValueError(f"Expected 20 records, got {len(data)}")

    encoder = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    encoder.max_seq_length = 256

    rows = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for _, r in data.iterrows():
        text = str(r["description"])
        emb = encoder.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
            device=device,
        )[0].astype(np.float32)
        d = build_pathway_features(text)
        path = np.asarray(list(d.values()), dtype=np.float32)
        full = np.concatenate([emb, path])
        X = full[keep].reshape(1, -1)
        prob = float(model.predict_proba(X)[0, 1])
        saved_prob = float(r["sif_precursor_probability"])
        rows.append(
            {
                "report_id": r["report_id"],
                "saved_probability_v1_2": saved_prob,
                "recomputed_probability": prob,
                "absolute_difference": abs(prob - saved_prob),
            }
        )
    out = pd.DataFrame(rows)
    max_diff = float(out.absolute_difference.max())
    mean_diff = float(out.absolute_difference.mean())
    status = (
        "PASS_EXACT"
        if max_diff < 1e-5
        else (
            "PASS_NUMERIC_TOLERANCE"
            if max_diff < 1e-3
            else "FAIL_SCHEMA_OR_RUNTIME_MISMATCH"
        )
    )
    summary = {
        "status": status,
        "records": len(out),
        "max_absolute_difference": max_diff,
        "mean_absolute_difference": mean_diff,
        "embedding_dimension": emb_dim,
        "pathway_dimension": 38,
        "total_dimension": 422,
    }
    od = PROJECT_ROOT / "experiments" / "rakshak_inference_validation_v1.2"
    od.mkdir(parents=True, exist_ok=True)
    out.sort_values("absolute_difference", ascending=False).to_csv(
        od / "locked_test_inference_consistency.csv", index=False
    )
    (od / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 80)
    print("RAKSHAK EXACT INFERENCE CONSISTENCY VALIDATION v1.2")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print("\nTOP DIFFERENCES")
    print(
        out.sort_values("absolute_difference", ascending=False)
        .head(10)
        .to_string(index=False)
    )
    print("\nSaved:")
    print(od)


if __name__ == "__main__":
    main()
