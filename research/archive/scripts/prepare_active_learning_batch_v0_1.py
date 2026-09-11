from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SELECTED_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "active_learning_v0.1"
    / "active_learning_batch_01.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "active_learning_v0.1"
    / "active_learning_batch_01_annotation.csv"
)


DEFAULTS = {
    "annotation_status": "pending",
    "sif_potential": "",
    "sif_reason": "",
    "evidence_text": "",
    "hazard_notes": "",
    "exposure_notes": "",
    "consequence_notes": "",
    "barrier_notes": "",
    "annotator_confidence": "",
    "annotation_version": "v0.3",
    "annotation_round": "active_learning_01",
    "annotator_id": "",
    "annotated_at_utc": "",
}


REQUIRED_SOURCE_COLUMNS = {
    "report_id",
    "description",
}


def main() -> None:
    if not SELECTED_PATH.exists():
        raise FileNotFoundError(
            f"Active-learning selector output not found:\n{SELECTED_PATH}\n\n"
            "Run select_active_learning_batch_v0_1.py first."
        )

    df = pd.read_csv(SELECTED_PATH)

    missing = REQUIRED_SOURCE_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"Selected batch is missing required columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    # Preserve selector metadata for reproducibility, but the annotation
    # UI will not display the model scores.
    if "batch_id" not in df.columns:
        df["batch_id"] = "active_learning_batch_01"

    if "active_learning_rank" not in df.columns:
        df["active_learning_rank"] = range(1, len(df) + 1)

    # Add annotation fields without overwriting existing values.
    for column, default in DEFAULTS.items():
        if column not in df.columns:
            df[column] = default

    # Force this batch to the intended protocol/round.
    df["annotation_version"] = "v0.3"
    df["annotation_round"] = "active_learning_01"

    # Ensure fresh annotation state.
    df["annotation_status"] = (
        df["annotation_status"]
        .fillna("pending")
        .astype(str)
        .str.strip()
        .replace({"": "pending"})
    )

    # Keep the exact same core schema used by the existing annotation app.
    # The model-ranking columns remain in the file but are hidden by the UI.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("=" * 70)
    print("RAKSHAK ACTIVE LEARNING ANNOTATION BATCH PREPARATION")
    print("=" * 70)
    print(f"Selected records : {len(df)}")
    print(f"Output           : {OUTPUT_PATH}")
    print()
    print(
        df[
            [
                "active_learning_rank",
                "report_id",
            ]
        ].head(30).to_string(index=False)
    )
    print()
    print(
        "Model-selection scores are retained in the CSV for "
        "reproducibility but are not shown in the annotation UI."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
