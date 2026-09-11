from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLD_PATH = PROJECT_ROOT / "data" / "annotations" / "gold_set_v0.1.csv"


VALID_STATUSES = {
    "pending",
    "completed",
}

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


def main() -> None:
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"Gold Set not found:\n{GOLD_PATH}")

    df = pd.read_csv(GOLD_PATH)

    # ---------------------------------------------------------
    # Schema migration
    # ---------------------------------------------------------
    # Older Gold Set files may have been created before newer
    # annotation fields were introduced. Add them safely.
    schema_defaults = {
        "exposure_notes": "",
        "annotator_id": "",
        "annotated_at_utc": "",
        "annotation_version": "v0.1",
    }

    schema_changed = False

    for column, default in schema_defaults.items():
        if column not in df.columns:
            df[column] = default
            schema_changed = True

    if schema_changed:
        df.to_csv(
            GOLD_PATH,
            index=False,
            encoding="utf-8",
        )

        print(
            "Schema migration: missing annotation "
            "columns were added to the Gold Set."
        )

    print("=" * 70)
    print("RAKSHAK GOLD SET VALIDATION")
    print("=" * 70)

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

        raise SystemExit(1)

    print(f"Records: {len(df)}")

    # ---------------------------------------------------------
    # Duplicate report IDs
    # ---------------------------------------------------------

    duplicate_ids = df["report_id"].duplicated(keep=False)

    if duplicate_ids.any():
        ids = df.loc[duplicate_ids, "report_id"].astype(str).tolist()

        errors.append("Duplicate report IDs: " + ", ".join(ids))

    # ---------------------------------------------------------
    # Status validation
    # ---------------------------------------------------------

    invalid_status = (
        ~df["annotation_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(VALID_STATUSES)
    )

    if invalid_status.any():
        errors.append(
            f"Invalid annotation status in " f"{int(invalid_status.sum())} record(s)."
        )

    # ---------------------------------------------------------
    # Completed annotations
    # ---------------------------------------------------------

    completed_mask = (
        df["annotation_status"].fillna("").astype(str).str.lower().eq("completed")
    )

    completed = df.loc[completed_mask].copy()

    print(f"Completed: {len(completed)}")
    print(f"Pending:   {len(df) - len(completed)}")

    # ---------------------------------------------------------
    # Label validation
    # ---------------------------------------------------------

    for idx, row in completed.iterrows():

        report_id = clean(row["report_id"])

        label = clean(row["sif_potential"]).upper()

        reason = clean(row["sif_reason"])

        evidence = clean(row["evidence_text"])

        confidence = clean(row["annotator_confidence"]).upper()

        annotator = clean(row["annotator_id"])

        annotated_at = clean(row["annotated_at_utc"])

        if label not in VALID_LABELS:
            errors.append(f"{report_id}: invalid SIF label " f"'{label}'.")

        if not reason:
            errors.append(f"{report_id}: missing SIF reason.")

        if label in {"YES", "UNCERTAIN"} and not evidence:
            errors.append(f"{report_id}: evidence required for " f"{label}.")

        if confidence not in VALID_CONFIDENCE:
            errors.append(f"{report_id}: invalid confidence " f"'{confidence}'.")

        if not annotator:
            warnings.append(f"{report_id}: annotator ID is empty.")

        if not annotated_at:
            warnings.append(f"{report_id}: annotation timestamp is empty.")

    # ---------------------------------------------------------
    # Duplicate-group consistency
    # ---------------------------------------------------------

    grouped = df[completed_mask].groupby("duplicate_group_id")

    for group_id, group in grouped:

        labels = (
            group["sif_potential"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            .unique()
        )

        if len(labels) > 1:
            reports = ", ".join(group["report_id"].astype(str).tolist())

            errors.append(
                f"Duplicate group {group_id} has " f"conflicting labels: " f"{reports}"
            )

    # ---------------------------------------------------------
    # Annotation statistics
    # ---------------------------------------------------------

    print()
    print("SIF LABEL DISTRIBUTION")

    if completed.empty:
        print("  No completed annotations yet.")

    else:
        labels = (
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
            count = int(labels.get(label, 0))

            percentage = count / len(completed) if len(completed) else 0

            print(f"  {label:<10} " f"{count:>3} " f"({percentage:.1%})")

    print()
    print("CONFIDENCE DISTRIBUTION")

    if not completed.empty:
        confidence = (
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
            print(f"  {level:<10} " f"{int(confidence.get(level, 0)):>3}")

    # ---------------------------------------------------------
    # Evidence coverage
    # ---------------------------------------------------------

    if not completed.empty:

        evidence_present = (
            completed["evidence_text"].fillna("").astype(str).str.strip().ne("")
        )

        print()
        print(
            "Evidence supplied: " f"{int(evidence_present.sum())}/" f"{len(completed)}"
        )

    # ---------------------------------------------------------
    # Final result
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

        raise SystemExit(1)

    print("RESULT: PASSED")

    if warnings:
        print()
        print("WARNINGS")

        for warning in warnings:
            print(f"  ! {warning}")

    print("=" * 70)


if __name__ == "__main__":
    main()
