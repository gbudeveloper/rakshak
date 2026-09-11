from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

POOL_PATH = (
    PROJECT_ROOT / "data" / "annotations" / "industrial_safety_annotation_pool.csv"
)

GOLD_PATH = PROJECT_ROOT / "data" / "annotations" / "gold_set_v0.1.csv"

BATCH02_PATH = PROJECT_ROOT / "data" / "annotations" / "calibration_batch_02.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "annotations" / "calibration_batch_03.csv"


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def main() -> None:

    pool = pd.read_csv(POOL_PATH)

    used_ids: set[str] = set()

    for path in [
        GOLD_PATH,
        BATCH02_PATH,
    ]:
        if path.exists():
            data = pd.read_csv(path)

            if "report_id" in data.columns:
                used_ids.update(data["report_id"].astype(str))

    pool = pool[~pool["report_id"].astype(str).isin(used_ids)].copy()

    if len(pool) < 20:
        raise RuntimeError(f"Only {len(pool)} unused records remain.")

    pool["text_length"] = pool["description"].fillna("").astype(str).str.len()

    pool["risk"] = pool["hazards"].map(clean)

    pool["department"] = pool["department"].fillna("Unknown").astype(str)

    # ---------------------------------------------------------
    # Construct four sampling strata.
    # These are NOT SIF labels.
    # ---------------------------------------------------------

    groups = {
        "lower": pool[pool["potential_accident_level"].isin(["I", "II"])].copy(),
        "medium": pool[pool["potential_accident_level"].eq("III")].copy(),
        "high": pool[pool["potential_accident_level"].eq("IV")].copy(),
        "very_high": pool[pool["potential_accident_level"].isin(["V", "VI"])].copy(),
    }

    targets = {
        "lower": 5,
        "medium": 5,
        "high": 5,
        "very_high": 5,
    }

    selected_parts: list[pd.DataFrame] = []

    for name, target in targets.items():

        group = groups[name]

        if len(group) < target:
            raise RuntimeError(
                f"Not enough records in {name}: " f"need {target}, have {len(group)}."
            )

        # Encourage diversity across departments and risks.
        group = group.sort_values(
            [
                "department",
                "risk",
                "text_length",
                "report_id",
            ]
        )

        chosen: list[int] = []
        seen_departments: set[str] = set()
        seen_risks: set[str] = set()

        # First pass: department diversity.
        for idx, row in group.iterrows():

            department = clean(row["department"])

            if department not in seen_departments:
                chosen.append(idx)
                seen_departments.add(department)

            if len(chosen) == target:
                break

        # Second pass: risk diversity.
        if len(chosen) < target:

            for idx, row in group.iterrows():

                if idx in chosen:
                    continue

                risk = clean(row["risk"])

                if risk not in seen_risks:
                    chosen.append(idx)
                    seen_risks.add(risk)

                if len(chosen) == target:
                    break

        # Final fill.
        if len(chosen) < target:

            for idx in group.index:

                if idx not in chosen:
                    chosen.append(idx)

                if len(chosen) == target:
                    break

        selected = group.loc[chosen[:target]].copy()

        selected["annotation_round"] = "v0.2_calibration"

        selected_parts.append(selected)

    result = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    if len(result) != 20:
        raise RuntimeError(f"Expected 20 records, got {len(result)}.")

    # Fresh annotation fields.
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
    result["annotation_version"] = "v0.2"

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
    print("RAKSHAK CALIBRATION BATCH 03")
    print("=" * 65)

    print(f"Batch size: {len(result)}")

    print()
    print("Potential Accident Level:")
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
