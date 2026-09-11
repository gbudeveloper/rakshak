from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT / "data" / "annotations" / "industrial_safety_annotation_pool.csv"
)

CURRENT_GOLD_PATH = PROJECT_ROOT / "data" / "annotations" / "gold_set_v0.1.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "annotations" / "calibration_batch_02.csv"


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def main() -> None:

    pool = pd.read_csv(INPUT_PATH)
    gold = pd.read_csv(CURRENT_GOLD_PATH)

    # Reports already selected for annotation.
    used_ids = set(gold["report_id"].astype(str))

    pool = pool[~pool["report_id"].astype(str).isin(used_ids)].copy()

    # ---------------------------------------------------------
    # Narrative length
    # ---------------------------------------------------------

    pool["text_length"] = pool["description"].fillna("").astype(str).str.len()

    pool["risk"] = pool["hazards"].apply(clean)

    # ---------------------------------------------------------
    # Create broad source strata.
    #
    # IMPORTANT:
    # These are sampling strata only.
    # They are NOT SIF labels.
    # ---------------------------------------------------------

    low_potential = pool[pool["potential_accident_level"].isin(["I", "II"])].copy()

    medium = pool[pool["potential_accident_level"].eq("III")].copy()

    high = pool[pool["potential_accident_level"].isin(["IV", "V", "VI"])].copy()

    selected: list[pd.DataFrame] = []

    # ---------------------------------------------------------
    # 4 lower-potential examples
    #
    # Prefer short / simple narratives.
    # ---------------------------------------------------------

    low_potential = low_potential.sort_values(
        [
            "text_length",
            "department",
            "report_id",
        ]
    )

    selected.append(low_potential.head(4))

    # ---------------------------------------------------------
    # 3 medium / potentially ambiguous examples
    #
    # Prefer middle-length narratives with varied risks.
    # ---------------------------------------------------------

    medium["length_distance"] = (medium["text_length"] - 350).abs()

    medium = medium.sort_values(
        [
            "length_distance",
            "risk",
            "report_id",
        ]
    )

    selected.append(medium.head(3))

    # ---------------------------------------------------------
    # 3 higher-risk examples
    #
    # Avoid simply selecting the highest levels.
    # Pick varied narratives.
    # ---------------------------------------------------------

    high = high.sort_values(
        [
            "department",
            "risk",
            "text_length",
            "report_id",
        ]
    )

    selected.append(high.head(3))

    result = pd.concat(
        selected,
        ignore_index=True,
    )

    # Guard against insufficient candidates.
    if len(result) != 10:
        raise RuntimeError(f"Expected 10 calibration records, got {len(result)}.")

    result["annotation_round"] = "pilot_02"
    result["annotation_status"] = "pending"

    result["sif_potential"] = ""
    result["sif_reason"] = ""
    result["evidence_text"] = ""
    result["hazard_notes"] = ""
    result["exposure_notes"] = ""
    result["consequence_notes"] = ""
    result["barrier_notes"] = ""
    result["annotator_confidence"] = ""
    result["annotator_id"] = ""
    result["annotated_at_utc"] = ""
    result["annotation_version"] = "v0.1"

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("=" * 65)
    print("RAKSHAK CALIBRATION BATCH 02")
    print("=" * 65)

    print(f"Batch size: {len(result)}")

    print()
    print("Sampling strata:")
    print(result["potential_accident_level"].value_counts().sort_index().to_string())

    print()
    print("Industry:")
    print(result["department"].value_counts().to_string())

    print()
    print("Reports:")
    for report_id in result["report_id"]:
        print(f"  - {report_id}")

    print()
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
