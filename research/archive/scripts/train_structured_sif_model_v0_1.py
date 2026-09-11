"""
RAKSHAK Structured SIF Model v0.1
=================================

Train/evaluate a leakage-safe SIF classifier using the 119 structured features
created by sif_feature_builder_v0_1.py.

Design:
- Locked split from data/processed/sif_splits_v0.2
- Train only on train split
- Hyperparameter/threshold selection only on validation
- Test remains locked
- Reports both:
    A) FULL structured feature model
    B) MECHANISM-ONLY model excluding actual/potential consequence features
       to test whether the representation is learning SIF mechanism rather
       than simply using outcome severity.

Models:
- L1 Logistic Regression (primary; suitable for small n / many sparse features)
- L2 Logistic Regression (secondary)
- Random Forest is included as a diagnostic nonlinear baseline.

Outputs:
    experiments/sif_structured_model_v0.1/
        model_comparison.csv
        validation_threshold_search.csv
        locked_test_predictions.csv
        feature_coefficients.csv
        summary.json

This is a research/engineering baseline, not a production safety decision
system.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT / "experiments" / "sif_feature_builder_v0.1" / "sif_features.csv"
)

LABEL_PATH = PROJECT_ROOT / "data" / "annotations" / "resolved_annotations_v0.2.csv"

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "sif_splits_v0.2"

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "sif_structured_model_v0.1"


def metrics(y_true, prob, threshold=0.5):
    pred = (np.asarray(prob) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f2": round(float(fbeta_score(y_true, pred, beta=2, zero_division=0)), 4),
        "pr_auc": (
            round(
                float(average_precision_score(y_true, prob)),
                4,
            )
            if len(np.unique(y_true)) == 2
            else None
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_splits():
    result = {}
    for split in ["train", "validation", "test"]:
        path = SPLIT_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        result[split] = set(pd.read_csv(path)["report_id"].astype(str))
    return result


def threshold_search(y, prob):
    rows = []
    for threshold in np.linspace(0.20, 0.80, 121):
        m = metrics(y, prob, threshold)
        rows.append(
            {
                "threshold": round(float(threshold), 4),
                **m,
            }
        )

    table = pd.DataFrame(rows)

    # For a safety triage use-case, prioritize recall/F2 but don't accept a
    # validation operating point with precision below 0.50 when alternatives
    # exist.
    candidates = table[table["precision"] >= 0.50]

    if candidates.empty:
        candidates = table

    best = candidates.sort_values(
        ["f2", "recall", "precision", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]

    return float(best["threshold"]), table


def build_pipeline(model):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def evaluate_variant(
    name,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):
    candidates = {
        "l1_logistic_C_0.10": build_pipeline(
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=0.10,
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            )
        ),
        "l1_logistic_C_0.50": build_pipeline(
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=0.50,
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            )
        ),
        "l1_logistic_C_1.00": build_pipeline(
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=1.00,
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            )
        ),
        "l2_logistic_C_0.50": build_pipeline(
            LogisticRegression(
                penalty="l2",
                solver="liblinear",
                C=0.50,
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            )
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    records = []
    fitted = {}

    for model_name, model in candidates.items():
        model.fit(X_train, y_train)

        p_train = model.predict_proba(X_train)[:, 1]
        p_val = model.predict_proba(X_val)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]

        threshold, threshold_table = threshold_search(y_val, p_val)

        val_metrics = metrics(y_val, p_val, threshold)
        test_metrics = metrics(y_test, p_test, threshold)

        records.append(
            {
                "variant": name,
                "model": model_name,
                "threshold": round(threshold, 4),
                "train_f2_at_0.5": metrics(y_train, p_train, 0.5)["f2"],
                "validation_f2": val_metrics["f2"],
                "validation_precision": val_metrics["precision"],
                "validation_recall": val_metrics["recall"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f2": test_metrics["f2"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_fp": test_metrics["fp"],
                "test_fn": test_metrics["fn"],
                "test_tp": test_metrics["tp"],
                "test_tn": test_metrics["tn"],
            }
        )

        fitted[(name, model_name)] = (
            model,
            threshold,
            threshold_table,
            p_test,
        )

    table = pd.DataFrame(records)
    best_row = table.sort_values(
        ["validation_f2", "validation_recall", "validation_precision"],
        ascending=[False, False, False],
    ).iloc[0]

    best_key = (name, best_row["model"])
    return table, fitted[best_key], fitted


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(FEATURE_PATH)
    if not LABEL_PATH.exists():
        raise FileNotFoundError(LABEL_PATH)

    features = pd.read_csv(FEATURE_PATH)
    labels = pd.read_csv(LABEL_PATH)[["report_id", "sif_potential"]].drop_duplicates(
        "report_id"
    )

    df = features.merge(labels, on="report_id", how="inner")
    df["y"] = (
        df["sif_potential"].astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0})
    )
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)

    split_ids = load_splits()

    train_ids = split_ids["train"]
    val_ids = split_ids["validation"]
    test_ids = split_ids["test"]

    train = df[df["report_id"].astype(str).isin(train_ids)].copy()
    val = df[df["report_id"].astype(str).isin(val_ids)].copy()
    test = df[df["report_id"].astype(str).isin(test_ids)].copy()

    # IDs/labels are never model features.
    all_features = [c for c in features.columns if c != "report_id"]

    # Mechanism-only excludes direct consequence summaries. These features can
    # otherwise make the model overly dependent on what already happened.
    excluded_mechanism = [
        c for c in all_features if c.startswith("actual_") or c.startswith("potential_")
    ]

    variants = {
        "full_structured": all_features,
        "mechanism_only": [c for c in all_features if c not in excluded_mechanism],
    }

    print("=" * 86)
    print("RAKSHAK STRUCTURED SIF MODEL v0.1")
    print("=" * 86)
    print(f"Gold binary records : {len(df)}")
    print(f"Train/Val/Test      : {len(train)}/{len(val)}/{len(test)}")
    print(f"Full features       : {len(variants['full_structured'])}")
    print(f"Mechanism-only      : {len(variants['mechanism_only'])}")
    print()

    comparison = []
    fitted_best = {}
    all_fitted = {}

    for variant_name, cols in variants.items():
        X_train = train[cols]
        X_val = val[cols]
        X_test = test[cols]

        y_train = train["y"].to_numpy()
        y_val = val["y"].to_numpy()
        y_test = test["y"].to_numpy()

        table, best, fitted = evaluate_variant(
            variant_name,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        )

        comparison.append(table)
        fitted_best[variant_name] = best
        all_fitted.update(fitted)

        print(f"VARIANT: {variant_name}")
        print(
            table[
                [
                    "model",
                    "threshold",
                    "validation_f2",
                    "validation_precision",
                    "validation_recall",
                    "test_precision",
                    "test_recall",
                    "test_f2",
                    "test_fp",
                    "test_fn",
                ]
            ].to_string(index=False)
        )
        print()

    comparison_df = pd.concat(comparison, ignore_index=True)

    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    comparison_df.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Persist all validation threshold searches.
    threshold_records = []
    for (variant, model_name), (
        model,
        threshold,
        threshold_table,
        p_test,
    ) in all_fitted.items():
        temp = threshold_table.copy()
        temp.insert(0, "variant", variant)
        temp.insert(1, "model", model_name)
        threshold_records.append(temp)

    threshold_df = pd.concat(
        threshold_records,
        ignore_index=True,
    )
    threshold_path = OUTPUT_DIR / "validation_threshold_search.csv"
    threshold_df.to_csv(
        threshold_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Best model globally by validation F2.
    best_global = comparison_df.sort_values(
        ["validation_f2", "validation_recall", "validation_precision"],
        ascending=[False, False, False],
    ).iloc[0]

    best_variant = best_global["variant"]
    best_model_name = best_global["model"]

    model, best_threshold, _, test_prob = all_fitted[(best_variant, best_model_name)]

    best_test = test[["report_id", "y", "sif_potential"]].copy()

    best_test["probability"] = test_prob
    best_test["threshold"] = best_threshold
    best_test["prediction"] = (best_test["probability"] >= best_threshold).astype(int)

    best_test["error"] = np.where(
        best_test["prediction"] == best_test["y"],
        "CORRECT",
        np.where(
            best_test["prediction"] == 1,
            "FALSE_POSITIVE",
            "FALSE_NEGATIVE",
        ),
    )

    test_prediction_path = OUTPUT_DIR / "locked_test_predictions.csv"
    best_test.to_csv(
        test_prediction_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Feature coefficients for the globally best linear model.
    coef_rows = []
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        final_model = model.named_steps["model"]
        if hasattr(final_model, "coef_"):
            selected_cols = variants[best_variant]
            coefs = final_model.coef_[0]

            for feature, coef in sorted(
                zip(selected_cols, coefs),
                key=lambda x: abs(x[1]),
                reverse=True,
            ):
                coef_rows.append(
                    {
                        "feature": feature,
                        "coefficient": round(float(coef), 6),
                        "direction": "SIF_POSITIVE" if coef > 0 else "SIF_NEGATIVE",
                    }
                )

    coef_path = OUTPUT_DIR / "feature_coefficients.csv"
    pd.DataFrame(coef_rows).to_csv(
        coef_path,
        index=False,
        encoding="utf-8-sig",
    )

    best_test_metrics = metrics(
        best_test["y"].to_numpy(),
        best_test["probability"].to_numpy(),
        best_threshold,
    )

    summary = {
        "version": "0.1",
        "gold_binary_records": int(len(df)),
        "split_sizes": {
            "train": int(len(train)),
            "validation": int(len(val)),
            "test": int(len(test)),
        },
        "feature_counts": {
            "full_structured": int(len(variants["full_structured"])),
            "mechanism_only": int(len(variants["mechanism_only"])),
        },
        "best_model_by_validation_f2": {
            "variant": str(best_variant),
            "model": str(best_model_name),
            "validation_threshold": round(float(best_threshold), 4),
            "locked_test_metrics": best_test_metrics,
        },
        "notes": [
            "Threshold selected on validation only.",
            "Test set remains locked.",
            "Mechanism-only variant excludes actual_ and potential_ features.",
            "Probabilities are model scores and should not be interpreted as calibrated safety probabilities without calibration validation.",
        ],
    }

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 86)
    print("BEST MODEL")
    print("=" * 86)
    print(f"Variant   : {best_variant}")
    print(f"Model     : {best_model_name}")
    print(f"Threshold : {best_threshold:.3f}")
    print(f"Test      : {best_test_metrics}")
    print()
    print("LOCKED TEST PREDICTIONS")
    print(best_test.to_string(index=False))
    print()
    print("FILES")
    print(f"  Comparison : {comparison_path}")
    print(f"  Thresholds : {threshold_path}")
    print(f"  Predictions: {test_prediction_path}")
    print(f"  Coefficients: {coef_path}")
    print(f"  Summary    : {summary_path}")
    print("=" * 86)


if __name__ == "__main__":
    main()
