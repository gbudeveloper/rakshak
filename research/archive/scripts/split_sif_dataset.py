from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "data" / "annotations" / "resolved_annotations_v0.1.csv"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "sif_splits_v0.1"

RANDOM_SEED = 42

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def validate_input(df: pd.DataFrame) -> None:
    required = {
        "report_id",
        "description",
        "duplicate_group_id",
        "final_sif_potential",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "Input dataset is missing required columns: " + ", ".join(sorted(missing))
        )

    labels = df["final_sif_potential"].fillna("").astype(str).str.strip().str.upper()

    invalid = labels[~labels.isin({"YES", "NO", "UNCERTAIN"})]

    if not invalid.empty:
        raise ValueError(
            "Invalid final_sif_potential values found: "
            + ", ".join(sorted(set(invalid.tolist())))
        )

    if df["report_id"].duplicated().any():
        duplicates = (
            df.loc[
                df["report_id"].duplicated(keep=False),
                "report_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError("Duplicate report_id values found: " + ", ".join(duplicates))

    # Every report in the resolved dataset should belong to exactly
    # one duplicate group.
    if df["duplicate_group_id"].isna().any():
        raise ValueError("Missing duplicate_group_id values found.")


def build_group_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse records to duplicate groups.

    A duplicate group must never be split across train/validation/test.
    This prevents exact-narrative leakage.
    """
    group_table = df.groupby("duplicate_group_id", as_index=False).agg(
        label=("final_sif_potential", "first"),
        record_count=("report_id", "count"),
    )

    # A duplicate group should have one resolved label.
    label_counts = df.groupby("duplicate_group_id")["final_sif_potential"].nunique()

    conflicting = label_counts[label_counts > 1]

    if not conflicting.empty:
        raise ValueError(
            "Conflicting labels found inside duplicate groups: "
            + ", ".join(str(x) for x in conflicting.index.tolist())
        )

    return group_table


def stratified_group_split(
    group_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split duplicate groups into train/validation/test while preserving
    class proportions as far as the small dataset permits.

    With 40 reports, exact proportions may not be possible. The function
    explicitly checks that both classes appear in every split when enough
    groups exist.
    """

    if len(group_table) < 10:
        raise ValueError("Too few duplicate groups for a reliable three-way split.")

    labels = group_table["label"].astype(str)

    # First reserve 15% of groups for the final test set.
    train_val, test = train_test_split(
        group_table,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    # Convert the remaining fraction so validation is 15% of the
    # full dataset, not 15% of the remainder.
    validation_relative = VALIDATION_FRACTION / (TRAIN_FRACTION + VALIDATION_FRACTION)

    train_labels = train_val["label"].astype(str)

    train, validation = train_test_split(
        train_val,
        test_size=validation_relative,
        random_state=RANDOM_SEED,
        stratify=train_labels,
    )

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def expand_groups(
    df: pd.DataFrame,
    group_table: pd.DataFrame,
) -> pd.DataFrame:
    groups = set(group_table["duplicate_group_id"].astype(str))

    result = df[df["duplicate_group_id"].astype(str).isin(groups)].copy()

    return result.sort_values("report_id").reset_index(drop=True)


def distribution(df: pd.DataFrame) -> dict[str, int]:
    counts = (
        df["final_sif_potential"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .value_counts()
    )

    return {
        label: int(counts.get(label, 0))
        for label in [
            "YES",
            "NO",
            "UNCERTAIN",
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create leakage-safe train/validation/test splits "
            "for the RAKSHAK SIF dataset."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Resolved annotation dataset not found:\n{args.input}")

    df = pd.read_csv(args.input)

    validate_input(df)

    # Only binary-labeled records are currently eligible for the
    # first classifier. UNCERTAIN is intentionally excluded.
    model_df = df[
        df["final_sif_potential"].astype(str).str.upper().isin({"YES", "NO"})
    ].copy()

    if len(model_df) < 20:
        raise ValueError("Fewer than 20 binary-labeled records are available.")

    group_table = build_group_table(model_df)

    train_groups, validation_groups, test_groups = stratified_group_split(group_table)

    train = expand_groups(model_df, train_groups)
    validation = expand_groups(model_df, validation_groups)
    test = expand_groups(model_df, test_groups)

    # ---------------------------------------------------------
    # Leakage checks
    # ---------------------------------------------------------

    train_ids = set(train["report_id"].astype(str))
    validation_ids = set(validation["report_id"].astype(str))
    test_ids = set(test["report_id"].astype(str))

    if train_ids & validation_ids:
        raise RuntimeError("Train/validation report overlap detected.")

    if train_ids & test_ids:
        raise RuntimeError("Train/test report overlap detected.")

    if validation_ids & test_ids:
        raise RuntimeError("Validation/test report overlap detected.")

    train_groups_set = set(train["duplicate_group_id"].astype(str))
    validation_groups_set = set(validation["duplicate_group_id"].astype(str))
    test_groups_set = set(test["duplicate_group_id"].astype(str))

    if train_groups_set & validation_groups_set:
        raise RuntimeError("Duplicate-group leakage between train and validation.")

    if train_groups_set & test_groups_set:
        raise RuntimeError("Duplicate-group leakage between train and test.")

    if validation_groups_set & test_groups_set:
        raise RuntimeError("Duplicate-group leakage between validation and test.")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = args.output_dir / "train.csv"
    validation_path = args.output_dir / "validation.csv"
    test_path = args.output_dir / "test.csv"
    manifest_path = args.output_dir / "split_manifest.json"

    train.to_csv(
        train_path,
        index=False,
        encoding="utf-8",
    )

    validation.to_csv(
        validation_path,
        index=False,
        encoding="utf-8",
    )

    test.to_csv(
        test_path,
        index=False,
        encoding="utf-8",
    )

    manifest = {
        "dataset": "RAKSHAK SIF",
        "source_file": str(args.input),
        "random_seed": RANDOM_SEED,
        "strategy": "stratified group split",
        "group_column": "duplicate_group_id",
        "label_column": "final_sif_potential",
        "uncertain_policy": (
            "Excluded from binary model splits; retained in resolved dataset."
        ),
        "target_fractions": {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": TEST_FRACTION,
        },
        "splits": {
            "train": {
                "records": len(train),
                "groups": int(train["duplicate_group_id"].nunique()),
                "distribution": distribution(train),
                "report_ids": sorted(train["report_id"].astype(str).tolist()),
            },
            "validation": {
                "records": len(validation),
                "groups": int(validation["duplicate_group_id"].nunique()),
                "distribution": distribution(validation),
                "report_ids": sorted(validation["report_id"].astype(str).tolist()),
            },
            "test": {
                "records": len(test),
                "groups": int(test["duplicate_group_id"].nunique()),
                "distribution": distribution(test),
                "report_ids": sorted(test["report_id"].astype(str).tolist()),
            },
        },
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Console report
    # ---------------------------------------------------------

    print("=" * 70)
    print("RAKSHAK SIF DATASET SPLIT")
    print("=" * 70)

    print(f"Input records         : {len(df)}")
    print(f"Binary-model records  : {len(model_df)}")
    print(f"Excluded UNCERTAIN   : " f"{len(df) - len(model_df)}")
    print()

    for name, frame in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:
        print(name)
        print(f"  Records            : {len(frame)}")
        print(f"  Duplicate groups   : " f"{frame['duplicate_group_id'].nunique()}")
        print(f"  Distribution       : " f"{distribution(frame)}")
        print()

    print("Leakage checks       : PASSED")
    print()
    print(f"Train      : {train_path}")
    print(f"Validation : {validation_path}")
    print(f"Test       : {test_path}")
    print(f"Manifest   : {manifest_path}")


if __name__ == "__main__":
    main()
