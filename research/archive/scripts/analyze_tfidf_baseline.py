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
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(PREDICTIONS_PATH)

    model = joblib.load(MODEL_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    test = pd.read_csv(TEST_PATH)

    print("=" * 70)
    print("RAKSHAK TF-IDF BASELINE — ERROR ANALYSIS")
    print("=" * 70)

    print("\nTEST RECORDS")
    print("-" * 70)

    output_columns = [
        "report_id",
        "final_sif_potential",
        "sif_probability",
        "prediction_at_0.5",
        "prediction_at_validation_threshold",
    ]

    print(
        predictions[output_columns]
        .sort_values("sif_probability", ascending=False)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # Error analysis at threshold 0.50
    # ---------------------------------------------------------

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

    predictions["correct_at_0.5"] = y_true == y_pred

    predictions["error_type"] = "CORRECT"

    predictions.loc[
        (y_true == 0) & (y_pred == 1),
        "error_type",
    ] = "FALSE_POSITIVE"

    predictions.loc[
        (y_true == 1) & (y_pred == 0),
        "error_type",
    ] = "FALSE_NEGATIVE"

    print("\nERRORS AT THRESHOLD 0.50")
    print("-" * 70)

    errors = predictions[predictions["error_type"] != "CORRECT"]

    if errors.empty:
        print("No errors.")
    else:
        print(
            errors[
                [
                    "report_id",
                    "final_sif_potential",
                    "sif_probability",
                    "error_type",
                ]
            ]
            .sort_values("sif_probability", ascending=False)
            .to_string(index=False)
        )

    # ---------------------------------------------------------
    # Reconnect descriptions
    # ---------------------------------------------------------

    descriptions = test[["report_id", "description"]].copy()

    review = predictions.merge(
        descriptions,
        on="report_id",
        how="left",
    )

    print("\nFULL TEST CASE REVIEW")
    print("-" * 70)

    for _, row in review.sort_values(
        "sif_probability",
        ascending=False,
    ).iterrows():

        print()
        print(f"Report ID : {row['report_id']}")
        print(f"True      : {row['final_sif_potential']}")
        print(f"Probability: " f"{float(row['sif_probability']):.6f}")
        print(f"Prediction : " f"{row['prediction_at_0.5']}")
        print(f"Error type : " f"{row['error_type']}")
        print(f"Narrative:\n{row['description']}")

    # ---------------------------------------------------------
    # Model vocabulary / coefficients
    # ---------------------------------------------------------

    try:
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["classifier"]

        feature_names = np.asarray(vectorizer.get_feature_names_out())

        coefficients = classifier.coef_[0]

        top_positive = np.argsort(coefficients)[-25:][::-1]

        top_negative = np.argsort(coefficients)[:25]

        print("\nTOP FEATURES PUSHING TOWARD YES")
        print("-" * 70)

        for idx in top_positive:
            print(f"{feature_names[idx]:<35}" f"{coefficients[idx]: .6f}")

        print("\nTOP FEATURES PUSHING TOWARD NO")
        print("-" * 70)

        for idx in top_negative:
            print(f"{feature_names[idx]:<35}" f"{coefficients[idx]: .6f}")

    except Exception as exc:
        print("\nCould not extract model coefficients:" f" {exc}")

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
