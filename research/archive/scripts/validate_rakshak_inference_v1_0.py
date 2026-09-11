"""
Validate the frozen RAKSHAK v1.2 inference package against the exact
20 locked-test narratives.

Run from:
    C:\Projects\SIH26165>

    python scripts\validate_rakshak_inference_v1_0.py

The script:
- loads the exact locked test narratives;
- runs the inference feature reconstruction;
- compares probabilities with the probabilities saved by v1.2;
- reports absolute differences;
- does NOT retrain the model.

A small numerical difference can occur because of floating-point/runtime
differences. Large differences indicate an inference feature/schema mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "rakshak_final_v1.2"
    / "rakshak_final_model.joblib"
)
PREDICTION_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "rakshak_final_v1.2"
    / "locked_test_predictions_with_evidence.csv"
)
LABEL_FILE = PROJECT_ROOT / "data" / "annotations" / "resolved_annotations_v0.2.csv"
TEST_SPLIT = PROJECT_ROOT / "data" / "processed" / "sif_splits_v0.2" / "test.csv"

sys.path.insert(0, str(SCRIPT_DIR))

from evidence_engine_v1_5 import analyze
from rakshak_inference_v1_1 import (
    PATHWAY_FEATURE_NAMES,
    encode,
    load_package,
    reconstruct_pathway_features,
)


def main() -> None:
    package = load_package()
    model = package["model"]
    keep_mask = np.asarray(package["feature_keep_mask"], dtype=bool)
    embedding_dim = int(package["embedding_dimension"])

    if keep_mask.size != 422:
        raise ValueError(
            f"Frozen mask must have 422 entries, got {keep_mask.size}"
        )

    if embedding_dim != 384:
        raise ValueError(
            f"Frozen embedding dimension must be 384, got {embedding_dim}"
        )

    labels = pd.read_csv(LABEL_FILE)
    test = pd.read_csv(TEST_SPLIT)
    saved = pd.read_csv(PREDICTION_FILE)

    for df in (labels, test, saved):
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
        raise ValueError(f"Expected 20 locked-test records, got {len(data)}")

    # Load one encoder only.
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device=device,
    )
    encoder.max_seq_length = 256

    rows = []

    for _, row in data.iterrows():
        text = str(row["description"])

        embedding = encoder.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
            device=device,
        )[0].astype(np.float32)

        evidence = analyze(text)
        pathway = reconstruct_pathway_features(evidence)

        if embedding.shape[0] != 384:
            raise ValueError(
                f"{row['report_id']}: embedding has {embedding.shape[0]} dims"
            )

        if pathway.shape[0] != 38:
            raise ValueError(
                f"{row['report_id']}: pathway vector has {pathway.shape[0]} dims"
            )

        full = np.concatenate([embedding, pathway])

        if full.shape[0] != 422:
            raise ValueError(
                f"{row['report_id']}: full vector has {full.shape[0]} dims"
            )

        X = full[keep_mask].reshape(1, -1)
        probability = float(model.predict_proba(X)[0, 1])

        saved_probability = float(row["sif_precursor_probability"])
        diff = abs(probability - saved_probability)

        rows.append({
            "report_id": row["report_id"],
            "true_sif_potential": row["final_sif_potential"],
            "saved_probability_v1_2": saved_probability,
            "recomputed_probability": probability,
            "absolute_difference": diff,
            "pathway_assessment": evidence.get("pathway_assessment"),
            "pathways": evidence.get("pathways"),
        })

    result = pd.DataFrame(rows).sort_values(
        "absolute_difference",
        ascending=False,
    )

    output_dir = PROJECT_ROOT / "experiments" / "rakshak_inference_validation_v1.0"
    output_dir.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        output_dir / "locked_test_inference_consistency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    max_diff = float(result["absolute_difference"].max())
    mean_diff = float(result["absolute_difference"].mean())

    if max_diff < 1e-5:
        status = "PASS_EXACT"
    elif max_diff < 1e-3:
        status = "PASS_NUMERIC_TOLERANCE"
    else:
        status = "FAIL_SCHEMA_OR_RUNTIME_MISMATCH"

    summary = {
        "status": status,
        "records": int(len(result)),
        "max_absolute_difference": max_diff,
        "mean_absolute_difference": mean_diff,
        "embedding_dimension": embedding_dim,
        "pathway_dimension": len(PATHWAY_FEATURE_NAMES),
        "total_dimension": 422,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("RAKSHAK INFERENCE CONSISTENCY VALIDATION v1.0")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print("\nTOP DIFFERENCES")
    print(result.head(10).to_string(index=False))
    print("\nSaved:")
    print(output_dir)


if __name__ == "__main__":
    main()
