from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "industrial_safety_canonical.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "quality"


def safe_value_counts(
    series: pd.Series,
    top_n: int = 20,
) -> dict[str, int]:
    counts = series.fillna("<NULL>").astype(str).value_counts().head(top_n)

    return {
        str(key): int(value)
        for key, value in counts.items()
    }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: {INPUT_PATH}"
        )

    df = pd.read_parquet(INPUT_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------
    # Text quality
    # -----------------------------

    text = df["description"].fillna("").astype(str).str.strip()
    text_lengths = text.str.len()

    empty_description_count = int((text == "").sum())

    duplicate_description_count = int(
        text.duplicated(keep=False).sum()
    )

    unique_description_count = int(
        text.nunique()
    )

    # -----------------------------
    # Timestamp quality
    # -----------------------------

    timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    invalid_timestamp_count = int(
        timestamps.isna().sum()
    )

    # -----------------------------
    # Identifier quality
    # -----------------------------

    duplicate_report_ids = int(
        df["report_id"].duplicated().sum()
    )

    # -----------------------------
    # Report
    # -----------------------------

    report = {
        "dataset": "industrial_safety_canonical",
        "records": int(len(df)),
        "columns": int(len(df.columns)),

        "text": {
            "empty_description_count": empty_description_count,
            "unique_description_count": unique_description_count,
            "duplicate_description_records": duplicate_description_count,
            "min_characters": int(text_lengths.min()),
            "max_characters": int(text_lengths.max()),
            "mean_characters": round(float(text_lengths.mean()), 2),
            "median_characters": round(
                float(text_lengths.median()),
                2,
            ),
        },

        "timestamps": {
            "invalid_count": invalid_timestamp_count,
            "minimum": (
                timestamps.min().isoformat()
                if not timestamps.isna().all()
                else None
            ),
            "maximum": (
                timestamps.max().isoformat()
                if not timestamps.isna().all()
                else None
            ),
        },

        "identifiers": {
            "duplicate_report_ids": duplicate_report_ids,
        },

        "distributions": {
            "actual_outcome": safe_value_counts(
                df["actual_outcome"]
            ),
            "potential_accident_level": safe_value_counts(
                df["potential_accident_level"]
            ),
            "site": safe_value_counts(
                df["site"]
            ),
            "location": safe_value_counts(
                df["location"]
            ),
            "department": safe_value_counts(
                df["department"]
            ),
            "hazards": safe_value_counts(
                df["hazards"].apply(
                    lambda x: (
                        ", ".join(x)
                        if isinstance(x, list)
                        else str(x)
                    )
                )
            ),
        },
    }

    output_path = OUTPUT_DIR / "industrial_safety_quality.json"

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -----------------------------
    # Console summary
    # -----------------------------

    print("=" * 60)
    print("RAKSHAK DATASET QUALITY REPORT")
    print("=" * 60)

    print(f"Records                : {len(df)}")
    print(f"Columns                : {len(df.columns)}")
    print()

    print("TEXT")
    print(f"Empty descriptions     : {empty_description_count}")
    print(f"Unique descriptions    : {unique_description_count}")
    print(f"Duplicate description  : {duplicate_description_count}")
    print(f"Minimum characters     : {text_lengths.min()}")
    print(f"Maximum characters     : {text_lengths.max()}")
    print(f"Mean characters        : {text_lengths.mean():.2f}")
    print(f"Median characters      : {text_lengths.median():.2f}")
    print()

    print("TIMESTAMPS")
    print(f"Invalid timestamps     : {invalid_timestamp_count}")
    print(f"Minimum timestamp      : {timestamps.min()}")
    print(f"Maximum timestamp      : {timestamps.max()}")
    print()

    print("IDENTIFIERS")
    print(f"Duplicate report IDs   : {duplicate_report_ids}")
    print()

    print("ACTUAL ACCIDENT LEVEL")
    print(df["actual_outcome"].value_counts(dropna=False).to_string())
    print()

    print("POTENTIAL ACCIDENT LEVEL")
    print(
        df["potential_accident_level"]
        .value_counts(dropna=False)
        .to_string()
    )
    print()

    print("INDUSTRY / DEPARTMENT")
    print(df["department"].value_counts(dropna=False).to_string())
    print()

    print("SITES")
    print(df["site"].value_counts(dropna=False).to_string())
    print()

    print("CRITICAL RISK")
    print(
        df["hazards"]
        .apply(
            lambda x: (
                ", ".join(x)
                if isinstance(x, list)
                else str(x)
            )
        )
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print(f"JSON report: {output_path}")


if __name__ == "__main__":
    main()