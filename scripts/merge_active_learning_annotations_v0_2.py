from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESOLVED_PATH = PROJECT_ROOT / "data" / "annotations" / "resolved_annotations_v0.1.csv"

ACTIVE_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "active_learning_v0.1"
    / "active_learning_batch_01_annotated.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "annotations"

OUTPUT_PATH = OUTPUT_DIR / "resolved_annotations_v0.2.csv"

REPORT_PATH = OUTPUT_DIR / "resolved_annotations_v0.2_report.json"


def find_label_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        f"No label column found. Tried {candidates}. " f"Available: {list(df.columns)}"
    )


def find_required(df: pd.DataFrame, name: str) -> str:
    if name not in df.columns:
        raise ValueError(
            f"Required column '{name}' is missing. " f"Available: {list(df.columns)}"
        )
    return name


def main() -> None:
    for path in [RESOLVED_PATH, ACTIVE_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file:\n{path}")

    old = pd.read_csv(RESOLVED_PATH)
    new = pd.read_csv(ACTIVE_PATH)

    find_required(old, "report_id")
    find_required(new, "report_id")

    old_label = find_label_column(
        old,
        ["final_sif_potential", "sif_potential", "resolved_sif_potential"],
    )
    new_label = find_label_column(
        new,
        ["final_sif_potential", "sif_potential", "resolved_sif_potential"],
    )

    old = old.copy()
    new = new.copy()

    old["report_id"] = old["report_id"].astype(str).str.strip()
    new["report_id"] = new["report_id"].astype(str).str.strip()

    old[old_label] = old[old_label].astype(str).str.strip().str.upper()
    new[new_label] = new[new_label].astype(str).str.strip().str.upper()

    # Only binary, completed labels are eligible for this v0.2
    # resolved training set.
    new_completed = new[
        new[new_label].isin(["YES", "NO"])
        & new.get(
            "annotation_status",
            pd.Series("completed", index=new.index),
        )
        .fillna("completed")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("completed")
    ].copy()

    if new_completed.empty:
        raise ValueError("No completed binary labels found in active-learning batch.")

    # Strong integrity checks before concatenation.
    old_ids = set(old["report_id"])
    new_ids = set(new_completed["report_id"])

    overlap = old_ids & new_ids
    if overlap:
        raise ValueError(
            "Overlap detected between existing resolved set and active batch: "
            + ", ".join(sorted(overlap))
        )

    # Preserve old resolved schema first.
    # Add any new annotation columns that did not exist previously.
    combined = pd.concat(
        [old, new_completed],
        ignore_index=True,
        sort=False,
    )

    # Normalize the canonical label name.
    if old_label != "final_sif_potential":
        combined["final_sif_potential"] = combined[old_label]

    if new_label != "final_sif_potential" and new_label in combined.columns:
        combined["final_sif_potential"] = combined["final_sif_potential"].fillna(
            combined[new_label]
        )

    combined["final_sif_potential"] = (
        combined["final_sif_potential"].astype(str).str.strip().str.upper()
    )

    combined = combined[combined["final_sif_potential"].isin(["YES", "NO"])].copy()

    # Deterministic ordering by report ID.
    combined = combined.sort_values("report_id").reset_index(drop=True)

    # Duplicate integrity.
    if combined["report_id"].duplicated().any():
        dupes = (
            combined.loc[
                combined["report_id"].duplicated(keep=False),
                "report_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError("Duplicate report IDs after merge: " + ", ".join(dupes))

    yes_count = int(combined["final_sif_potential"].eq("YES").sum())
    no_count = int(combined["final_sif_potential"].eq("NO").sum())

    old_yes = int(old[old_label].isin(["YES"]).sum())
    old_no = int(old[old_label].isin(["NO"]).sum())

    new_yes = int(new_completed[new_label].eq("YES").sum())
    new_no = int(new_completed[new_label].eq("NO").sum())

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    report = {
        "version": "resolved_annotations_v0.2",
        "previous_resolved_records": int(len(old)),
        "previous_yes": old_yes,
        "previous_no": old_no,
        "active_learning_completed_records": int(len(new_completed)),
        "active_learning_yes": new_yes,
        "active_learning_no": new_no,
        "final_resolved_records": int(len(combined)),
        "final_yes": yes_count,
        "final_no": no_count,
        "duplicate_report_ids": 0,
        "source_files": {
            "previous": str(RESOLVED_PATH.relative_to(PROJECT_ROOT)),
            "active_learning": str(ACTIVE_PATH.relative_to(PROJECT_ROOT)),
        },
        "eligible_for_binary_training": True,
        "note": (
            "v0.2 combines the original 40 resolved binary "
            "annotations with the 30 completed active-learning "
            "annotations. Existing labels are preserved."
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print("RAKSHAK RESOLVED ANNOTATIONS MERGE v0.2")
    print("=" * 78)
    print(f"Previous resolved : {len(old)}")
    print(f"Active batch      : {len(new_completed)}")
    print(f"Final resolved    : {len(combined)}")
    print(f"YES               : {yes_count}")
    print(f"NO                : {no_count}")
    print(f"Duplicates        : 0")
    print()
    print(f"Output : {OUTPUT_PATH}")
    print(f"Report : {REPORT_PATH}")
    print("=" * 78)


if __name__ == "__main__":
    main()
