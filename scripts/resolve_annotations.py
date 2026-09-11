from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PILOT_FILES = [
    PROJECT_ROOT / "data" / "annotations" / "gold_set_v0.1.csv",
    PROJECT_ROOT / "data" / "annotations" / "calibration_batch_02_annotated.csv",
    PROJECT_ROOT / "data" / "annotations" / "calibration_batch_03_annotated.csv",
]

ADJUDICATION_FILE = (
    PROJECT_ROOT / "data" / "annotations" / "boundary_adjudication_v0.3.csv"
)

OUTPUT_FILE = PROJECT_ROOT / "data" / "annotations" / "resolved_annotations_v0.1.csv"


def clean(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def main() -> None:

    frames: list[pd.DataFrame] = []

    for path in PILOT_FILES:

        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        frame = pd.read_csv(path)

        frame = frame[
            frame["annotation_status"]
            .fillna("")
            .astype(str)
            .str.lower()
            .eq("completed")
        ].copy()

        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise RuntimeError("No completed pilot annotations found.")

    pilot = pd.concat(
        frames,
        ignore_index=True,
    )

    # One report should appear only once.
    pilot = pilot.drop_duplicates(
        subset=["report_id"],
        keep="last",
    )

    pilot["final_sif_potential"] = (
        pilot["sif_potential"].fillna("").astype(str).str.strip().str.upper()
    )

    pilot["label_source"] = "pilot"

    # ---------------------------------------------------------
    # Apply v0.3 adjudications
    # ---------------------------------------------------------

    if ADJUDICATION_FILE.exists():

        adjudicated = pd.read_csv(ADJUDICATION_FILE)

        adjudicated = adjudicated[
            adjudicated["adjudication_status"]
            .fillna("")
            .astype(str)
            .str.lower()
            .eq("completed")
        ].copy()

        adjudicated["adjudication_label"] = (
            adjudicated["adjudication_label"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        adjudication_map = dict(
            zip(
                adjudicated["report_id"].astype(str),
                adjudicated["adjudication_label"],
            )
        )

        adjudicated_ids: set[str] = set()

        for idx, row in pilot.iterrows():

            report_id = str(row["report_id"])

            if report_id in adjudication_map:

                label = adjudication_map[report_id]

                if label not in {
                    "YES",
                    "NO",
                    "UNCERTAIN",
                }:
                    raise ValueError(
                        f"Invalid adjudication label " f"for {report_id}: {label}"
                    )

                pilot.loc[
                    idx,
                    "final_sif_potential",
                ] = label

                pilot.loc[
                    idx,
                    "label_source",
                ] = "adjudicated"

                adjudicated_ids.add(report_id)

        print(f"Adjudications applied: " f"{len(adjudicated_ids)}")

    # ---------------------------------------------------------
    # Validate resolved labels
    # ---------------------------------------------------------

    valid_labels = {
        "YES",
        "NO",
        "UNCERTAIN",
    }

    invalid = pilot[~pilot["final_sif_potential"].isin(valid_labels)]

    if not invalid.empty:

        print("WARNING: unresolved labels:")

        print(
            invalid[
                [
                    "report_id",
                    "final_sif_potential",
                ]
            ].to_string(index=False)
        )

    resolved = pilot[pilot["final_sif_potential"].isin(valid_labels)].copy()

    if resolved.empty:
        raise RuntimeError("No resolved annotations available.")

    # ---------------------------------------------------------
    # Add model-training flags
    # ---------------------------------------------------------

    resolved["is_uncertain"] = resolved["final_sif_potential"] == "UNCERTAIN"

    resolved["usable_for_binary_sif_model"] = resolved["final_sif_potential"].isin(
        ["YES", "NO"]
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print("=" * 70)
    print("RAKSHAK RESOLVED ANNOTATIONS")
    print("=" * 70)

    print(f"Pilot records: " f"{len(pilot)}")

    print(f"Resolved records: " f"{len(resolved)}")

    print()
    print("Final SIF labels:")

    print(
        resolved["final_sif_potential"]
        .value_counts()
        .reindex(
            ["YES", "NO", "UNCERTAIN"],
            fill_value=0,
        )
        .to_string()
    )

    print()
    print("Label source:")

    print(resolved["label_source"].value_counts().to_string())

    print()
    print("Binary-model eligible:")

    print(int(resolved["usable_for_binary_sif_model"].sum()))

    print()
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
