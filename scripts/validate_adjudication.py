from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALID_LABELS = {"YES", "NO", "UNCERTAIN"}
VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}
VALID_STATUS = {"pending", "completed"}

REQUIRED_COLUMNS = {
    "report_id",
    "description",
    "adjudication_status",
    "adjudication_label",
    "adjudication_reason",
    "adjudication_confidence",
    "adjudicator_id",
    "adjudicated_at_utc",
}


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def validate(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Adjudication file not found:\n{path}")

    df = pd.read_csv(path)

    print("=" * 70)
    print("RAKSHAK BOUNDARY ADJUDICATION VALIDATION")
    print("=" * 70)
    print(f"File: {path}")
    print(f"Records: {len(df)}")

    errors: list[str] = []
    warnings: list[str] = []

    # ---------------------------------------------------------
    # Schema
    # ---------------------------------------------------------

    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        errors.append("Missing columns: " + ", ".join(sorted(missing)))

    if errors:
        print()
        print("ERRORS")
        for error in errors:
            print(f"  ✗ {error}")
        print("=" * 70)
        return 1

    # ---------------------------------------------------------
    # Duplicate IDs
    # ---------------------------------------------------------

    duplicate_ids = df["report_id"].duplicated(keep=False)

    if duplicate_ids.any():
        ids = df.loc[duplicate_ids, "report_id"].astype(str).tolist()

        errors.append("Duplicate report IDs: " + ", ".join(ids))

    # ---------------------------------------------------------
    # Status
    # ---------------------------------------------------------

    status = df["adjudication_status"].fillna("").astype(str).str.strip().str.lower()

    invalid_status = ~status.isin(VALID_STATUS)

    if invalid_status.any():
        errors.append(
            f"Invalid adjudication status in " f"{int(invalid_status.sum())} record(s)."
        )

    completed = df.loc[status.eq("completed")].copy()
    pending = df.loc[status.eq("pending")].copy()

    print(f"Completed: {len(completed)}")
    print(f"Pending:   {len(pending)}")

    # ---------------------------------------------------------
    # Completed record validation
    # ---------------------------------------------------------

    for _, row in completed.iterrows():
        report_id = clean(row["report_id"])
        label = clean(row["adjudication_label"]).upper()
        reason = clean(row["adjudication_reason"])
        confidence = clean(row["adjudication_confidence"]).upper()
        adjudicator = clean(row["adjudicator_id"])
        timestamp = clean(row["adjudicated_at_utc"])

        if label not in VALID_LABELS:
            errors.append(f"{report_id}: invalid adjudication label " f"'{label}'.")

        if not reason:
            errors.append(f"{report_id}: missing adjudication reason.")

        if confidence not in VALID_CONFIDENCE:
            errors.append(f"{report_id}: invalid confidence " f"'{confidence}'.")

        if not adjudicator:
            warnings.append(f"{report_id}: adjudicator ID missing.")

        if not timestamp:
            warnings.append(f"{report_id}: adjudication timestamp missing.")

    # ---------------------------------------------------------
    # Label distribution
    # ---------------------------------------------------------

    print()
    print("ADJUDICATED LABEL DISTRIBUTION")

    if completed.empty:
        print("  No completed adjudications.")

    else:
        counts = (
            completed["adjudication_label"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            .value_counts()
        )

        for label in ["YES", "NO", "UNCERTAIN"]:
            print(f"  {label:<10}" f"{int(counts.get(label, 0)):>4}")

    # ---------------------------------------------------------
    # Confidence distribution
    # ---------------------------------------------------------

    if not completed.empty:
        print()
        print("CONFIDENCE DISTRIBUTION")

        counts = (
            completed["adjudication_confidence"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            .value_counts()
        )

        for level in ["HIGH", "MEDIUM", "LOW"]:
            print(f"  {level:<10}" f"{int(counts.get(level, 0)):>4}")

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print()
    print("=" * 70)

    if errors:
        print("RESULT: FAILED")
        print()
        print("ERRORS")

        for error in errors:
            print(f"  ✗ {error}")

        if warnings:
            print()
            print("WARNINGS")

            for warning in warnings:
                print(f"  ! {warning}")

        print("=" * 70)
        return 1

    print("RESULT: PASSED")

    if warnings:
        print()
        print("WARNINGS")

        for warning in warnings:
            print(f"  ! {warning}")

    print("=" * 70)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a RAKSHAK boundary adjudication CSV."
    )

    parser.add_argument(
        "file",
        type=Path,
        help="Path to the adjudication CSV.",
    )

    args = parser.parse_args()

    path = args.file

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    raise SystemExit(validate(path))


if __name__ == "__main__":
    main()
