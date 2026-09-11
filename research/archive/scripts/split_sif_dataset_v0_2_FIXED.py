from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "resolved_annotations_v0.2.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sif_splits_v0.2"
)

TRAIN_PATH = OUTPUT_DIR / "train.csv"
VALIDATION_PATH = OUTPUT_DIR / "validation.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"
MANIFEST_PATH = OUTPUT_DIR / "split_manifest.json"

RANDOM_STATE = 42

# Approximate target proportions.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15


def normalize_text(value: object) -> str:
    return (
        str(value)
        .lower()
        .replace("\u00a0", " ")
        .strip()
    )


def choose_column(
    df: pd.DataFrame,
    candidates: list[str],
    label: str,
) -> str:
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find {label}. "
        f"Tried {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def load_dataset() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Resolved v0.2 dataset not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    report_id_col = choose_column(
        df,
        ["report_id"],
        "report ID column",
    )

    description_col = choose_column(
        df,
        ["description", "Description"],
        "description column",
    )

    label_col = choose_column(
        df,
        [
            "final_sif_potential",
            "sif_potential",
            "resolved_sif_potential",
        ],
        "SIF label column",
    )

    df = df.copy()

    if report_id_col != "report_id":
        df = df.rename(columns={report_id_col: "report_id"})

    if description_col != "description":
        df = df.rename(columns={description_col: "description"})

    if label_col != "final_sif_potential":
        df = df.rename(
            columns={label_col: "final_sif_potential"}
        )

    df["report_id"] = (
        df["report_id"].astype(str).str.strip()
    )

    df["description"] = (
        df["description"].fillna("").astype(str).str.strip()
    )

    df["final_sif_potential"] = (
        df["final_sif_potential"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Binary-model dataset only.
    df = df[
        df["final_sif_potential"].isin(["YES", "NO"])
    ].copy()

    if df.empty:
        raise ValueError(
            "No YES/NO records found in resolved_annotations_v0.2.csv."
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
        raise ValueError(
            "Duplicate report IDs found: "
            + ", ".join(duplicates)
        )

    if df["description"].eq("").any():
        count = int(df["description"].eq("").sum())
        raise ValueError(
            f"{count} records have empty descriptions."
        )

    return df.reset_index(drop=True)


def build_duplicate_groups(
    df: pd.DataFrame,
) -> tuple[pd.Series, dict[str, int]]:
    """
    Use duplicate_group_id when available.

    Otherwise derive a deterministic group ID from normalized
    description. This prevents identical narratives from being
    placed in different splits.
    """
    if "duplicate_group_id" in df.columns:
        raw = (
            df["duplicate_group_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        # Fill missing group IDs from normalized descriptions.
        normalized = df["description"].map(normalize_text)

        result = raw.copy()

        missing = result.eq("")
        result.loc[missing] = normalized.loc[missing].map(
            lambda text: (
                "derived_"
                + hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest()[:16]
            )
        )

        return result, {
            "source": "duplicate_group_id + description fallback",
            "derived_count": int(missing.sum()),
        }

    normalized = df["description"].map(normalize_text)

    group_ids = normalized.map(
        lambda text: (
            "derived_"
            + hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()[:16]
        )
    )

    return group_ids, {
        "source": "normalized description",
        "derived_count": int(len(df)),
    }


def stratified_group_split(
    df: pd.DataFrame,
    group_ids: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split at duplicate-group level while approximately preserving
    YES/NO proportions.

    With 70 records and a small number of groups, exact 70/15/15
    is not always mathematically possible. We select groups using
    a deterministic greedy objective that balances:
      1) target record count
      2) target YES count
      3) target NO count
    """

    temp = df.copy()
    temp["_group_id"] = group_ids.values

    grouped = (
        temp.groupby(
            "_group_id",
            dropna=False,
            sort=True,
        )
        .agg(
            n=("report_id", "size"),
            yes=(
                "final_sif_potential",
                lambda s: int((s == "YES").sum()),
            ),
            no=(
                "final_sif_potential",
                lambda s: int((s == "NO").sum()),
            ),
        )
        .reset_index()
    )

    total_n = len(temp)
    total_yes = int(
        temp["final_sif_potential"].eq("YES").sum()
    )
    total_no = int(
        temp["final_sif_potential"].eq("NO").sum()
    )

    # Desired counts.
    desired = {
        "train": {
            "n": TRAIN_FRACTION * total_n,
            "yes": TRAIN_FRACTION * total_yes,
            "no": TRAIN_FRACTION * total_no,
        },
        "validation": {
            "n": VALIDATION_FRACTION * total_n,
            "yes": VALIDATION_FRACTION * total_yes,
            "no": VALIDATION_FRACTION * total_no,
        },
        "test": {
            "n": TEST_FRACTION * total_n,
            "yes": TEST_FRACTION * total_yes,
            "no": TEST_FRACTION * total_no,
        },
    }

    rng = random.Random(RANDOM_STATE)

    records = grouped.to_dict("records")
    rng.shuffle(records)

    # Larger groups first, then minority-heavy groups.
    records.sort(
        key=lambda row: (
            row["n"],
            abs(
                row["yes"] - row["no"]
            ),
        ),
        reverse=True,
    )

    assigned: dict[str, str] = {}

    totals = {
        "train": {"n": 0, "yes": 0, "no": 0},
        "validation": {"n": 0, "yes": 0, "no": 0},
        "test": {"n": 0, "yes": 0, "no": 0},
    }

    split_order = [
        "test",
        "validation",
        "train",
    ]

    def score(
        split_name: str,
        group: dict,
    ) -> float:
        current = totals[split_name]
        target = desired[split_name]

        new_n = current["n"] + group["n"]
        new_yes = current["yes"] + group["yes"]
        new_no = current["no"] + group["no"]

        # Relative target error, strongly emphasizing count balance.
        n_error = abs(new_n - target["n"]) / max(
            target["n"],
            1.0,
        )
        yes_error = abs(new_yes - target["yes"]) / max(
            target["yes"],
            1.0,
        )
        no_error = abs(new_no - target["no"]) / max(
            target["no"],
            1.0,
        )

        return (
            0.55 * n_error
            + 0.25 * yes_error
            + 0.20 * no_error
        )

    for group in records:
        candidates = []

        for split_name in split_order:
            # Avoid making validation/test empty.
            candidates.append(
                (
                    score(split_name, group),
                    split_name,
                )
            )

        candidates.sort(
            key=lambda item: item[0]
        )

        _, chosen = candidates[0]

        group_id = str(group["_group_id"])
        assigned[group_id] = chosen

        totals[chosen]["n"] += int(group["n"])
        totals[chosen]["yes"] += int(group["yes"])
        totals[chosen]["no"] += int(group["no"])

    # Rebalance if a split became empty.
    for required_split in ["train", "validation", "test"]:
        if any(
            assigned_group == required_split
            for assigned_group in assigned.values()
        ):
            continue

        donor = max(
            ["train", "validation", "test"],
            key=lambda name: totals[name]["n"],
        )

        donor_groups = [
            group_id
            for group_id, split_name in assigned.items()
            if split_name == donor
        ]

        if not donor_groups:
            raise RuntimeError(
                f"Could not populate {required_split}."
            )

        donor_group = donor_groups[-1]
        assigned[donor_group] = required_split

    temp["_split"] = temp["_group_id"].map(assigned)

    train = temp[temp["_split"].eq("train")].copy()
    validation = temp[temp["_split"].eq("validation")].copy()
    test = temp[temp["_split"].eq("test")].copy()

    train = train.drop(
        columns=["_group_id", "_split"]
    )
    validation = validation.drop(
        columns=["_group_id", "_split"]
    )
    test = test.drop(
        columns=["_group_id", "_split"]
    )

    return (
        train.sort_values("report_id").reset_index(drop=True),
        validation.sort_values("report_id").reset_index(drop=True),
        test.sort_values("report_id").reset_index(drop=True),
    )


def split_signature(df: pd.DataFrame) -> str:
    values = (
        df["report_id"]
        .astype(str)
        .sort_values()
        .tolist()
    )

    digest = hashlib.sha256(
        "\n".join(values).encode("utf-8")
    ).hexdigest()

    return digest


def profile(
    name: str,
    df: pd.DataFrame,
    group_ids: pd.Series,
) -> dict:
    ids = set(
        df["report_id"]
        .astype(str)
    )

    if "duplicate_group_id" in df.columns:
        local_groups = set(
            df["duplicate_group_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        local_groups = set(group_ids.loc[df.index].astype(str)) if len(df.index) else set()

    return {
        "records": int(len(df)),
        "yes": int(
            df["final_sif_potential"].eq("YES").sum()
        ),
        "no": int(
            df["final_sif_potential"].eq("NO").sum()
        ),
        "yes_rate": float(
            df["final_sif_potential"].eq("YES").mean()
        ) if len(df) else 0.0,
        "duplicate_groups": int(
            len(local_groups)
        ),
        "report_ids": sorted(ids),
        "signature": split_signature(df),
    }


def validate_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    group_ids: pd.Series,
) -> None:
    train_ids = set(train["report_id"].astype(str))
    validation_ids = set(validation["report_id"].astype(str))
    test_ids = set(test["report_id"].astype(str))

    if train_ids & validation_ids:
        raise RuntimeError("Train/validation report ID overlap.")
    if train_ids & test_ids:
        raise RuntimeError("Train/test report ID overlap.")
    if validation_ids & test_ids:
        raise RuntimeError("Validation/test report ID overlap.")

    # Group leakage check.
    # The split dataframes have already been reset to local indices,
    # so using the original group_ids Series by dataframe index would
    # misalign groups. Use the explicit duplicate_group_id column
    # carried in each split instead.
    if "duplicate_group_id" in train.columns:
        train_groups = set(
            train["duplicate_group_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        validation_groups = set(
            validation["duplicate_group_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        test_groups = set(
            test["duplicate_group_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        # Defensive fallback: derive groups from normalized descriptions.
        def fallback_groups(frame: pd.DataFrame) -> set[str]:
            values = (
                frame["description"]
                .fillna("")
                .astype(str)
                .map(normalize_text)
            )
            return {
                hashlib.sha256(
                    value.encode("utf-8")
                ).hexdigest()
                for value in values
            }

        train_groups = fallback_groups(train)
        validation_groups = fallback_groups(validation)
        test_groups = fallback_groups(test)

    if train_groups & validation_groups:
        raise RuntimeError("Duplicate-group leakage: train/validation.")
    if train_groups & test_groups:
        raise RuntimeError("Duplicate-group leakage: train/test.")
    if validation_groups & test_groups:
        raise RuntimeError("Duplicate-group leakage: validation/test.")

    if not len(train):
        raise RuntimeError("Training split is empty.")
    if not len(validation):
        raise RuntimeError("Validation split is empty.")
    if not len(test):
        raise RuntimeError("Test split is empty.")


def main() -> None:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    df = load_dataset()

    group_ids, group_info = build_duplicate_groups(df)

    train, validation, test = stratified_group_split(
        df,
        group_ids,
    )

    # NOTE:
    # The group IDs remain aligned to original df indices. Because
    # split dataframes preserve original index until reset, validation
    # can be performed before resetting index.
    validate_splits(
        train,
        validation,
        test,
        group_ids,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_csv(
        TRAIN_PATH,
        index=False,
        encoding="utf-8",
    )

    validation.to_csv(
        VALIDATION_PATH,
        index=False,
        encoding="utf-8",
    )

    test.to_csv(
        TEST_PATH,
        index=False,
        encoding="utf-8",
    )

    original_profile = {
        "records": int(len(df)),
        "yes": int(
            df["final_sif_potential"].eq("YES").sum()
        ),
        "no": int(
            df["final_sif_potential"].eq("NO").sum()
        ),
        "yes_rate": float(
            df["final_sif_potential"].eq("YES").mean()
        ),
        "duplicate_groups": int(
            group_ids.nunique()
        ),
    }

    manifest = {
        "version": "sif_splits_v0.2",
        "input": str(
            INPUT_PATH.relative_to(PROJECT_ROOT)
        ),
        "random_state": RANDOM_STATE,
        "target_fractions": {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": TEST_FRACTION,
        },
        "split_method": (
            "deterministic greedy duplicate-group-safe "
            "stratified split"
        ),
        "group_id_source": group_info,
        "dataset": original_profile,
        "splits": {
            "train": profile(
                "train",
                train,
                group_ids,
            ),
            "validation": profile(
                "validation",
                validation,
                group_ids,
            ),
            "test": profile(
                "test",
                test,
                group_ids,
            ),
        },
        "leakage_checks": {
            "report_id_overlap": False,
            "duplicate_group_overlap": False,
            "empty_split": False,
        },
        "note": (
            "Because duplicate groups are indivisible, exact "
            "70/15/15 record counts may not be mathematically "
            "possible. The splitter prioritizes approximate "
            "record and class balance while keeping groups intact."
        ),
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print("RAKSHAK LEAKAGE-SAFE SPLIT v0.2")
    print("=" * 78)
    print(
        f"Input records : {len(df)} "
        f"(YES={original_profile['yes']}, "
        f"NO={original_profile['no']})"
    )
    print(
        f"Duplicate groups: {original_profile['duplicate_groups']}"
    )
    print()

    for name, split in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:
        yes = int(
            split["final_sif_potential"].eq("YES").sum()
        )
        no = int(
            split["final_sif_potential"].eq("NO").sum()
        )
        print(
            f"{name:<12}: {len(split):>3} "
            f"(YES={yes:>2}, NO={no:>2}, "
            f"YES%={yes / len(split):.1%})"
        )

    print()
    print("LEAKAGE CHECKS")
    print("-" * 78)
    print("Report-ID overlap       : PASSED")
    print("Duplicate-group overlap : PASSED")
    print("Empty split             : PASSED")

    print()
    print("FILES")
    print("-" * 78)
    print(f"Train      : {TRAIN_PATH}")
    print(f"Validation : {VALIDATION_PATH}")
    print(f"Test       : {TEST_PATH}")
    print(f"Manifest   : {MANIFEST_PATH}")

    print()
    print("=" * 78)
    print("SPLIT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
