from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "industrial_safety_canonical.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "industrial_safety_annotation_pool.csv"
)


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_parquet(INPUT_PATH).copy()

    df["normalized_description"] = (
        df["description"]
        .fillna("")
        .map(normalize_text)
    )

    # Every exact normalized narrative gets a stable group.
    group_map = {
        value: f"DUP_{index:04d}"
        for index, value in enumerate(
            sorted(df["normalized_description"].unique()),
            start=1,
        )
    }

    df["duplicate_group_id"] = (
        df["normalized_description"].map(group_map)
    )

    group_sizes = (
        df["duplicate_group_id"]
        .value_counts()
        .rename("duplicate_group_size")
    )

    df = df.join(
        group_sizes,
        on="duplicate_group_id",
    )

    df["is_duplicate"] = (
        df["duplicate_group_size"] > 1
    )

    # Annotation status starts explicitly unknown.
    df["sif_potential"] = pd.NA
    df["annotation_confidence"] = pd.NA
    df["annotation_version"] = pd.NA

    # Annotation priority:
    # higher potential severity gets examined earlier.
    level_order = {
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
    }

    df["potential_level_numeric"] = (
        df["potential_accident_level"]
        .map(level_order)
    )

    # Keep one representative record per exact duplicate
    # for the initial annotation pool.
    pool = (
        df.sort_values(
            [
                "potential_level_numeric",
                "duplicate_group_id",
                "report_id",
            ],
            ascending=[False, True, True],
        )
        .drop_duplicates(
            subset=["duplicate_group_id"],
            keep="first",
        )
        .copy()
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pool.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("=" * 60)
    print("RAKSHAK ANNOTATION POOL")
    print("=" * 60)

    print(f"Original records       : {len(df)}")
    print(f"Unique narrative groups: {len(pool)}")
    print(
        f"Exact duplicate groups : "
        f"{int((group_sizes > 1).sum())}"
    )

    print()
    print("Potential Accident Level")
    print(
        pool["potential_accident_level"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Industry")
    print(
        pool["department"]
        .value_counts()
        .to_string()
    )

    print()
    print("Pool output:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()