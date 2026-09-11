from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "industrial_safety_canonical.parquet"


def normalize_for_matching(text: str) -> str:
    """
    Normalize text only for duplicate detection.

    The original description is never modified.
    """
    text = str(text).lower()

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_parquet(INPUT_PATH)

    text = df["description"].fillna("").astype(str)

    normalized = text.map(normalize_for_matching)

    # Exact normalized duplicates
    duplicate_mask = normalized.duplicated(keep=False)

    duplicate_groups = (
        pd.DataFrame(
            {
                "normalized_description": normalized,
                "report_id": df["report_id"],
                "description": df["description"],
            }
        )
        .loc[duplicate_mask]
        .sort_values("normalized_description")
    )

    groups = (
        duplicate_groups.groupby("normalized_description")
        .agg(
            record_count=("report_id", "count"),
            report_ids=("report_id", list),
            descriptions=("description", list),
        )
        .reset_index()
        .sort_values(
            "record_count",
            ascending=False,
        )
    )

    print("=" * 70)
    print("RAKSHAK DATASET PROFILE")
    print("=" * 70)

    print(f"Total records             : {len(df)}")
    print(f"Unique raw descriptions   : " f"{text.nunique()}")
    print(f"Unique normalized texts   : " f"{normalized.nunique()}")
    print(f"Records in duplicate groups: " f"{duplicate_mask.sum()}")
    print(f"Duplicate groups          : " f"{len(groups)}")

    print()
    print("DUPLICATE GROUPS")
    print("-" * 70)

    if groups.empty:
        print("No duplicate groups found.")

    else:
        for _, row in groups.head(20).iterrows():
            print(f"\nRecords: {row['record_count']}")

            print("Report IDs: " + ", ".join(row["report_ids"]))

            for description in row["descriptions"]:
                print(f"  {description}")

    # Cross-tab potential level and actual level
    print()
    print("ACTUAL × POTENTIAL ACCIDENT LEVEL")
    print("-" * 70)

    cross = pd.crosstab(
        df["actual_outcome"],
        df["potential_accident_level"],
        margins=True,
    )

    print(cross.to_string())

    # Critical risk count
    print()
    print("TOP CRITICAL RISKS")
    print("-" * 70)

    risk_counts = (
        df["hazards"]
        .apply(lambda x: (", ".join(x) if isinstance(x, list) else str(x)))
        .value_counts()
        .head(20)
    )

    print(risk_counts.to_string())

    # Reports that have higher potential than actual outcome.
    level_order = {
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
    }

    comparison = pd.DataFrame(
        {
            "actual": df["actual_outcome"],
            "potential": df["potential_accident_level"],
        }
    )

    comparison["actual_num"] = comparison["actual"].map(level_order)

    comparison["potential_num"] = comparison["potential"].map(level_order)

    higher_potential = comparison[
        comparison["potential_num"] > comparison["actual_num"]
    ]

    print()
    print("Potential level > actual level")
    print("-" * 70)
    print(
        f"Records: {len(higher_potential)} "
        f"/ {len(df)} "
        f"({len(higher_potential) / len(df):.1%})"
    )


if __name__ == "__main__":
    main()
