from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "baseline_tfidf_logreg_v0.1"
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "sif_splits_v0.1"
MODEL_PATH = EXPERIMENT_DIR / "model.joblib"
PREDICTIONS_PATH = EXPERIMENT_DIR / "test_predictions.csv"
TEST_PATH = SPLIT_DIR / "test.csv"


def main() -> None:
    for path in (MODEL_PATH, PREDICTIONS_PATH, TEST_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    model = joblib.load(MODEL_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    required_prediction_cols = {
        "report_id",
        "description",
        "final_sif_potential",
        "sif_probability",
        "prediction_at_0.5",
        "prediction_at_validation_threshold",
    }
    missing = required_prediction_cols - set(predictions.columns)
    if missing:
        raise ValueError(f"test_predictions.csv is missing columns: {sorted(missing)}")

    print("=" * 78)
    print("RAKSHAK TF-IDF BASELINE — ERROR ANALYSIS v2")
    print("=" * 78)
    print(f"Prediction rows: {len(predictions)}")
    print("Descriptions are read directly from test_predictions.csv; no merge is used.")

    predictions = predictions.copy()
    y_true = (
        predictions["final_sif_potential"]
        .astype(str)
        .str.upper()
        .map({"NO": 0, "YES": 1})
    )
    y_pred = (
        predictions["prediction_at_0.5"]
        .astype(str)
        .str.upper()
        .map({"NO": 0, "YES": 1})
    )

    if y_true.isna().any() or y_pred.isna().any():
        raise ValueError("Unexpected labels found in prediction file.")

    predictions["correct_at_0.5"] = y_true.eq(y_pred)
    predictions["error_type"] = "CORRECT"
    predictions.loc[(y_true == 0) & (y_pred == 1), "error_type"] = "FALSE_POSITIVE"
    predictions.loc[(y_true == 1) & (y_pred == 0), "error_type"] = "FALSE_NEGATIVE"

    print("\nTEST RECORDS")
    print("-" * 78)
    cols = [
        "report_id",
        "final_sif_potential",
        "sif_probability",
        "prediction_at_0.5",
        "prediction_at_validation_threshold",
        "error_type",
    ]
    print(
        predictions[cols]
        .sort_values("sif_probability", ascending=False)
        .to_string(index=False)
    )

    print("\nERROR SUMMARY")
    print("-" * 78)
    print(predictions["error_type"].value_counts().to_string())

    print("\nFULL TEST CASE REVIEW")
    print("-" * 78)
    for _, row in predictions.sort_values(
        "sif_probability", ascending=False
    ).iterrows():
        print()
        print(f"Report ID  : {row['report_id']}")
        print(f"True       : {row['final_sif_potential']}")
        print(f"Probability: {float(row['sif_probability']):.6f}")
        print(f"Prediction : {row['prediction_at_0.5']}")
        print(f"Error type : {row['error_type']}")
        print(f"Narrative  :\n{row['description']}")

    try:
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["classifier"]
        feature_names = np.asarray(vectorizer.get_feature_names_out())
        coefficients = classifier.coef_[0]

        print("\nTOP FEATURES PUSHING TOWARD YES")
        print("-" * 78)
        for idx in np.argsort(coefficients)[-25:][::-1]:
            print(f"{feature_names[idx]:<35}{coefficients[idx]: .6f}")

        print("\nTOP FEATURES PUSHING TOWARD NO")
        print("-" * 78)
        for idx in np.argsort(coefficients)[:25]:
            print(f"{feature_names[idx]:<35}{coefficients[idx]: .6f}")
    except Exception as exc:
        print(f"\nCould not extract model coefficients: {exc}")

    print("\nNEXT EXPERIMENT NOTE")
    print("-" * 78)
    print(
        "This test set contains only 6 records, so these results are diagnostic, not a reliable generalization estimate."
    )
    print(
        "The immediate purpose is to inspect false positives/negatives and establish a baseline before transformer experiments."
    )

    print("\n" + "=" * 78)
    print("ANALYSIS COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
