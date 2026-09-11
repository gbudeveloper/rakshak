"""
RAKSHAK SIF Reasoning Evaluation v0.1
=====================================

Evaluates sif_reasoning_v0.1 against the adjudicated binary SIF labels
in resolved_annotations_v0.2.csv.

Important:
- Diagnostic evaluation only.
- Uses train/validation/test split manifest from sif_splits_v0.2.
- Threshold/operating point is selected ONLY on validation.
- Test is locked for final diagnostic reporting.
- Does not retrain the reasoning rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix,
    average_precision_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REASONING_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "sif_reasoning_v0.1"
    / "sif_reasoning.csv"
)

LABEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "resolved_annotations_v0.2.csv"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sif_splits_v0.2"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "sif_reasoning_evaluation_v0.1"
)


def load_split_ids():
    result = {}
    for split in ["train", "validation", "test"]:
        path = SPLIT_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        result[split] = set(df["report_id"].astype(str))
    return result


def metrics(y_true, y_pred, prob=None):
    out = {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f2": round(float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)), 4),
    }

    if prob is not None and len(set(y_true)) > 1:
        out["pr_auc"] = round(float(average_precision_score(y_true, prob)), 4)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    out.update({
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    })

    return out


def band_binary(df, positive_bands):
    return df["sif_precursor_band"].isin(positive_bands).astype(int).to_numpy()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not REASONING_PATH.exists():
        raise FileNotFoundError(REASONING_PATH)
    if not LABEL_PATH.exists():
        raise FileNotFoundError(LABEL_PATH)

    reasoning = pd.read_csv(REASONING_PATH)
    labels = pd.read_csv(LABEL_PATH)

    if "sif_potential" not in labels.columns:
        raise ValueError(
            f"sif_potential not found. Available: {labels.columns.tolist()}"
        )

    merged = reasoning.merge(
        labels[["report_id", "sif_potential"]],
        on="report_id",
        how="inner",
    )

    merged["y"] = (
        merged["sif_potential"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"YES": 1, "NO": 0})
    )

    merged = merged.dropna(subset=["y"]).copy()
    merged["y"] = merged["y"].astype(int)

    split_ids = load_split_ids()

    rows = []
    summary = {
        "records": int(len(merged)),
        "band_rule_evaluations": {},
        "validation_threshold_selection": {},
        "locked_test": {},
    }

    print("=" * 86)
    print("RAKSHAK SIF REASONING EVALUATION v0.1")
    print("=" * 86)
    print(f"Records with binary gold label : {len(merged)}")
    print()

    # 1. Evaluate simple band operating points.
    for positive_bands in [
        ["HIGH"],
        ["HIGH", "MEDIUM"],
    ]:
        name = "+".join(positive_bands)

        subset_metrics = {}
        for split, ids in split_ids.items():
            part = merged[merged["report_id"].astype(str).isin(ids)].copy()
            if len(part) == 0:
                continue

            y = part["y"].to_numpy()
            pred = band_binary(part, positive_bands)
            subset_metrics[split] = metrics(y, pred)

        summary["band_rule_evaluations"][name] = subset_metrics

        print(f"OPERATING POINT: {name}")
        for split in ["train", "validation", "test"]:
            if split in subset_metrics:
                print(f"  {split:<11}: {subset_metrics[split]}")
        print()

    # 2. Validation threshold search on the continuous reasoning score.
    # Candidate thresholds are chosen only from validation.
    val = merged[
        merged["report_id"].astype(str).isin(split_ids["validation"])
    ].copy()

    test = merged[
        merged["report_id"].astype(str).isin(split_ids["test"])
    ].copy()

    threshold_rows = []
    for threshold in np.linspace(0.0, 1.0, 201):
        pred = (val["sif_precursor_score"].to_numpy() >= threshold).astype(int)
        m = metrics(val["y"].to_numpy(), pred)
        threshold_rows.append({
            "threshold": round(float(threshold), 4),
            **m,
        })

    threshold_df = pd.DataFrame(threshold_rows)

    # Recall-oriented selection. Require at least a modest precision floor
    # when possible; otherwise choose max F2.
    candidates = threshold_df[threshold_df["precision"] >= 0.50]

    if len(candidates) == 0:
        best = threshold_df.sort_values(
            ["f2", "recall", "precision"],
            ascending=[False, False, False],
        ).iloc[0]
    else:
        best = candidates.sort_values(
            ["f2", "recall", "precision"],
            ascending=[False, False, False],
        ).iloc[0]

    threshold = float(best["threshold"])

    y_val_pred = (val["sif_precursor_score"].to_numpy() >= threshold).astype(int)
    y_test_pred = (test["sif_precursor_score"].to_numpy() >= threshold).astype(int)

    val_metrics = metrics(
        val["y"].to_numpy(),
        y_val_pred,
        val["sif_precursor_score"].to_numpy(),
    )

    test_metrics = metrics(
        test["y"].to_numpy(),
        y_test_pred,
        test["sif_precursor_score"].to_numpy(),
    )

    summary["validation_threshold_selection"] = {
        "selected_threshold": round(threshold, 4),
        "validation_metrics": val_metrics,
    }

    summary["locked_test"] = {
        "threshold_from_validation_only": round(threshold, 4),
        "test_metrics": test_metrics,
    }

    print("=" * 86)
    print("VALIDATION-SELECTED SCORE THRESHOLD")
    print("=" * 86)
    print(f"Threshold : {threshold:.3f}")
    print(f"Validation: {val_metrics}")
    print()
    print("LOCKED TEST")
    print(f"Test      : {test_metrics}")
    print()

    # 3. Error cases on locked test.
    test_out = test[[
        "report_id",
        "y",
        "sif_potential",
        "sif_precursor_score",
        "sif_precursor_band",
        "primary_trigger",
        "reason_codes",
        "lsr_candidates",
    ]].copy()

    test_out["prediction"] = y_test_pred
    test_out["error"] = np.where(
        test_out["prediction"] == test_out["y"],
        "CORRECT",
        np.where(
            test_out["prediction"] == 1,
            "FALSE_POSITIVE",
            "FALSE_NEGATIVE",
        ),
    )

    test_out = test_out.sort_values(
        ["error", "sif_precursor_score"],
        ascending=[True, False],
    )

    test_out_path = OUTPUT_DIR / "locked_test_cases.csv"
    test_out.to_csv(test_out_path, index=False, encoding="utf-8-sig")

    threshold_path = OUTPUT_DIR / "validation_threshold_search.csv"
    threshold_df.to_csv(
        threshold_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("TOP LOCKED-TEST ERRORS")
    print("=" * 86)

    errors = test_out[test_out["error"] != "CORRECT"]
    if len(errors) == 0:
        print("No locked-test errors at the validation-selected threshold.")
    else:
        print(errors.to_string(index=False))

    print()
    print("FILES")
    print(f"  Test cases : {test_out_path}")
    print(f"  Thresholds : {threshold_path}")
    print(f"  Summary    : {summary_path}")
    print("=" * 86)


if __name__ == "__main__":
    main()
