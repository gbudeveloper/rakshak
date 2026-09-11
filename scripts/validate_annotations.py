from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


VALID_LABELS = {
    "YES",
    "NO",
    "UNCERTAIN",
}

VALID_CONFIDENCE = {
    "HIGH",
    "MEDIUM",
    "LOW",
}

VALID_STATUS = {
    "pending",
    "completed",
}


REQUIRED_COLUMNS = {
    "report_id",
    "description",
    "duplicate_group_id",
    "annotation_status",
    "sif_potential",
    "sif_reason",
    "evidence_text",
    "hazard_notes",
    "exposure_notes",
    "consequence_notes",
    "barrier_notes",
    "annotator_confidence",
    "annotator_id",
    "annotated_at_utc",
    "annotation_version",
}


def clean(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def validate(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found:\n{path}")

    df = pd.read_csv(path)

    print("=" * 70)
    print("RAKSHAK ANNOTATION VALIDATION")
    print("=" * 70)

    print(f"File: {path}")
    print(f"Records: {len(df)}")

    errors: list[str] = []
    warnings: list[str] = []

    # ---------------------------------------------------------
    # Schema
    # ---------------------------------------------------------

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)

    if missing_columns:
        errors.append("Missing columns: " + ", ".join(sorted(missing_columns)))

    if errors:
        print("\nERRORS")
        for error in errors:
            print(f"  ✗ {error}")

        return 1

    # ---------------------------------------------------------
    # Duplicate report IDs
    # ---------------------------------------------------------

    duplicate_ids = df["report_id"].duplicated(keep=False)

    if duplicate_ids.any():
        values = (
            df.loc[
                duplicate_ids,
                "report_id",
            ]
            .astype(str)
            .tolist()
        )

        errors.append("Duplicate report IDs: " + ", ".join(values))

    # ---------------------------------------------------------
    # Status
    # ---------------------------------------------------------

    status_series = (
        df["annotation_status"].fillna("").astype(str).str.strip().str.lower()
    )

    invalid_status = ~status_series.isin(VALID_STATUS)

    if invalid_status.any():
        errors.append(
            f"Invalid annotation status in " f"{int(invalid_status.sum())} record(s)."
        )

    completed = df.loc[status_series.eq("completed")].copy()

    pending = df.loc[status_series.eq("pending")].copy()

    print(f"Completed: {len(completed)}")
    print(f"Pending:   {len(pending)}")

    # ---------------------------------------------------------
    # Completed annotation validation
    # ---------------------------------------------------------

    for _, row in completed.iterrows():

        report_id = clean(row["report_id"])

        label = clean(row["sif_potential"]).upper()

        reason = clean(row["sif_reason"])

        evidence = clean(row["evidence_text"])

        confidence = clean(row["annotator_confidence"]).upper()

        annotator = clean(row["annotator_id"])

        timestamp = clean(row["annotated_at_utc"])

        if label not in VALID_LABELS:
            errors.append(f"{report_id}: invalid SIF label " f"'{label}'.")

        if not reason:
            errors.append(f"{report_id}: missing reasoning.")

        if label in {"YES", "UNCERTAIN"} and not evidence:
            errors.append(f"{report_id}: evidence required " f"for {label}.")

        if confidence not in VALID_CONFIDENCE:
            errors.append(f"{report_id}: invalid confidence " f"'{confidence}'.")

        if not annotator:
            warnings.append(f"{report_id}: annotator ID missing.")

        if not timestamp:
            warnings.append(f"{report_id}: annotation timestamp missing.")

    # ---------------------------------------------------------
    # Duplicate narrative groups
    # ---------------------------------------------------------

    completed_groups = completed.groupby("duplicate_group_id")

    for group_id, group in completed_groups:

        labels = (
            group["sif_potential"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            .unique()
        )

        if len(labels) > 1:

            ids = ", ".join(group["report_id"].astype(str).tolist())

            errors.append(
                f"Duplicate group {group_id} has " f"conflicting SIF labels: {ids}"
            )

    # ---------------------------------------------------------
    # Distribution
    # ---------------------------------------------------------

    print()
    print("SIF LABEL DISTRIBUTION")

    if completed.empty:
        print("  No completed annotations.")

    else:

        label_counts = (
            completed["sif_potential"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            .value_counts()
        )

        for label in [
            "YES",
            "NO",
            "UNCERTAIN",
        ]:

            count = int(
                label_counts.get(
                    label,
                    0,
                )
            )

            pct = count / len(completed)

            print(f"  {label:<10}" f"{count:>4} " f"({pct:.1%})")

    # ---------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------

    if not completed.empty:

        print()
        print("CONFIDENCE DISTRIBUTION")

        confidence_counts = (
            completed["annotator_confidence"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            .value_counts()
        )

        for level in [
            "HIGH",
            "MEDIUM",
            "LOW",
        ]:

            print(f"  {level:<10}" f"{int(confidence_counts.get(level, 0)):>4}")

    # ---------------------------------------------------------
    # Evidence coverage
    # ---------------------------------------------------------

    if not completed.empty:

        evidence_present = (
            completed["evidence_text"].fillna("").astype(str).str.strip().ne("")
        )

        print()
        print(
            "Evidence coverage: " f"{int(evidence_present.sum())}/" f"{len(completed)}"
        )

    # ---------------------------------------------------------
    # Final status
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
        description="Validate a RAKSHAK annotation dataset."
    )

    parser.add_argument(
        "file",
        type=Path,
        help=(
            "Annotation CSV path. " "Relative paths are resolved from the project root."
        ),
    )

    args = parser.parse_args()

    path = args.file

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    raise SystemExit(validate(path))


if __name__ == "__main__":
    main()
