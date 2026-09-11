"""
RAKSHAK exact inference validation — FINAL TRAIN+VALIDATION REFIT

Why this validator exists:
The file locked_test_predictions_with_evidence.csv in the v1.2 pipeline is
written BEFORE the final train+validation refit. The saved joblib model is
written AFTER that refit. Therefore comparing inference from the saved joblib
against locked_test_predictions_with_evidence.csv is an invalid comparison.

This validator:
1. Loads the SAVED final refit model package.
2. Builds the locked-test features with the exact training-time builder.
3. Recomputes the final-refit reference probabilities.
4. Runs the public inference script for every locked-test narrative.
5. Compares the two.
6. Saves a final-refit reference CSV.

Expected result:
    PASS_EXACT
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import subprocess

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "experiments" / "rakshak_final_v1.2"
MODEL_FILE = MODEL_DIR / "rakshak_final_model.joblib"
LABEL_FILE = PROJECT_ROOT / "data" / "annotations" / "resolved_annotations_v0.2.csv"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "sif_splits_v0.2" / "test.csv"

sys.path.insert(0, str(SCRIPT_DIR))

from train_rakshak_final_pipeline_v1_2 import build_pathway_features
from rakshak_inference_exact_v1_2 import infer, load_package, build_exact_feature_vector


def build_saved_model_reference() -> pd.DataFrame:
    package = load_package()
    model = package["model"]
    keep = np.asarray(package["feature_keep_mask"], dtype=bool)

    labels = pd.read_csv(LABEL_FILE)
    labels["report_id"] = labels["report_id"].astype(str)
    labels["description"] = labels["description"].fillna("").astype(str)

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
    encoder = SentenceTransformer(
        (
            package["model_name"]
            if "model_name" in package
            else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        device=device,
    )
    encoder.max_seq_length = 256

    rows = []
    for _, row in df.iterrows():
        text = str(row["description"])

        emb = encoder.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
            device=device,
        )[0].astype(np.float32)

        path_dict = build_pathway_features(text)
        path = np.asarray(list(path_dict.values()), dtype=np.float32)

        full = np.concatenate([emb, path])

        if full.shape[0] != 422:
            raise ValueError(
                f"{row['report_id']}: expected 422 raw features, got {full.shape[0]}"
            )

        X = full[keep].reshape(1, -1)
        prob = float(model.predict_proba(X)[0, 1])

        rows.append(
            {
                "report_id": row["report_id"],
                "description": text,
                "true_label": (
                    1 if str(row["final_sif_potential"]).upper() == "YES" else 0
                ),
                "final_refit_probability": prob,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    reference = build_saved_model_reference()

    inference_rows = []
    for _, row in reference.iterrows():
        result = infer(str(row["description"]))
        inference_rows.append(
            {
                "report_id": row["report_id"],
                "inference_probability": result["model"]["sif_precursor_probability"],
            }
        )

    inf = pd.DataFrame(inference_rows)

    merged = reference.merge(inf, on="report_id", how="inner", validate="one_to_one")
    merged["absolute_difference"] = (
        merged["final_refit_probability"] - merged["inference_probability"]
    ).abs()

    max_diff = float(merged["absolute_difference"].max())
    mean_diff = float(merged["absolute_difference"].mean())

    if max_diff < 1e-5:
        status = "PASS_EXACT"
    elif max_diff < 1e-3:
        status = "PASS_NUMERIC_TOLERANCE"
    else:
        status = "FAIL"

    out_dir = PROJECT_ROOT / "experiments" / "rakshak_inference_validation_v1.2"
    out_dir.mkdir(parents=True, exist_ok=True)

    reference.to_csv(
        out_dir / "locked_test_predictions_final_refit_reference.csv",
        index=False,
        encoding="utf-8-sig",
    )
    merged.sort_values("absolute_difference", ascending=False).to_csv(
        out_dir / "final_refit_inference_consistency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "status": status,
        "records": int(len(merged)),
        "max_absolute_difference": max_diff,
        "mean_absolute_difference": mean_diff,
        "model_file": str(MODEL_FILE),
        "comparison_basis": (
            "Saved final train+validation refit joblib model versus "
            "exact public inference."
        ),
        "not_used_as_reference": (
            "locked_test_predictions_with_evidence.csv, because that file "
            "contains pre-refit train-only test_prob values."
        ),
    }

    (out_dir / "final_refit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("RAKSHAK FINAL REFIT INFERENCE CONSISTENCY v1.2")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print("\nTOP DIFFERENCES")
    print(
        merged.sort_values("absolute_difference", ascending=False)
        .head(10)
        .to_string(index=False)
    )
    print("\nSaved:")
    print(out_dir)


if __name__ == "__main__":
    main()
