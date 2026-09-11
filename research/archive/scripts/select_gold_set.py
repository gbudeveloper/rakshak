from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "industrial_safety_annotation_pool.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "gold_set_candidates.csv"
)


LEVEL_ORDER = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
}


def clean(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)

    # ---------------------------------------------------------
    # Derived fields
    # ---------------------------------------------------------

    df["potential_numeric"] = (
        df["potential_accident_level"]
        .map(LEVEL_ORDER)
        .fillna(0)
    )

    df["actual_numeric"] = (
        df["actual_outcome"]
        .map(LEVEL_ORDER)
        .fillna(0)
    )

    df["description_length"] = (
        df["description"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    df["department"] = df["department"].fillna("Unknown")
    df["site"] = df["site"].fillna("Unknown")
    df["location"] = df["location"].fillna("Unknown")

    # Canonical risk string.
    def risk_value(value: object) -> str:
        if isinstance(value, list):
            return " | ".join(str(x) for x in value)

        return clean(value)

    df["risk"] = df["hazards"].map(risk_value)

    # ---------------------------------------------------------
    # Potential severity bands
    # ---------------------------------------------------------

    df["selection_band"] = "lower"

    df.loc[
        df["potential_accident_level"].eq("III"),
        "selection_band",
    ] = "medium"

    df.loc[
        df["potential_accident_level"].eq("IV"),
        "selection_band",
    ] = "medium_high"

    df.loc[
        df["potential_accident_level"].isin(["V", "VI"]),
        "selection_band",
    ] = "high"

    targets = {
        "high": 10,
        "medium_high": 15,
        "medium": 15,
        "lower": 10,
    }

    # ---------------------------------------------------------
    # Deterministic greedy diversity selection
    # ---------------------------------------------------------

    selected_indices: list[int] = []

    for band, target in targets.items():

        candidates = df[
            df["selection_band"].eq(band)
        ].copy()

        if candidates.empty:
            continue

        # We want representation from all available departments.
        department_counts = (
            candidates["department"]
            .value_counts()
            .to_dict()
        )

        # Start with rare combinations first.
        candidates["combo"] = (
            candidates["department"].astype(str)
            + "||"
            + candidates["risk"].astype(str)
        )

        combo_counts = (
            candidates["combo"]
            .value_counts()
            .to_dict()
        )

        candidates["department_frequency"] = (
            candidates["department"]
            .map(department_counts)
        )

        candidates["combo_frequency"] = (
            candidates["combo"]
            .map(combo_counts)
        )

        # Prefer:
        # 1. rare department
        # 2. rare risk/department combination
        # 3. diverse actual outcome
        # 4. reproducible report ID
        candidates = candidates.sort_values(
            [
                "department_frequency",
                "combo_frequency",
                "actual_numeric",
                "description_length",
                "report_id",
            ],
            ascending=[
                True,
                True,
                False,
                True,
                True,
            ],
        )

        # First pass: guarantee each available department gets
        # representation whenever possible.
        chosen: list[int] = []
        departments_seen: set[str] = set()

        for idx, row in candidates.iterrows():
            department = str(row["department"])

            if department not in departments_seen:
                chosen.append(idx)
                departments_seen.add(department)

            if len(chosen) >= target:
                break

        # Second pass: fill remaining slots.
        if len(chosen) < target:
            for idx in candidates.index:
                if idx not in chosen:
                    chosen.append(idx)

                if len(chosen) >= target:
                    break

        selected_indices.extend(chosen[:target])

    result = df.loc[selected_indices].copy()

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    if len(result) != 50:
        raise RuntimeError(
            f"Expected 50 candidates, got {len(result)}."
        )

    # Ensure one record per duplicate group.
    if result["duplicate_group_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate narrative groups detected in Gold Set."
        )

    # ---------------------------------------------------------
    # Annotation fields
    # ---------------------------------------------------------

    result["annotation_status"] = "pending"
    result["sif_potential"] = ""
    result["sif_reason"] = ""
    result["evidence_text"] = ""
    result["hazard_notes"] = ""
    result["consequence_notes"] = ""
    result["barrier_notes"] = ""
    result["annotator_confidence"] = ""
    result["annotation_version"] = "v0.1"

    # Stable output ordering.
    band_order = {
        "high": 1,
        "medium_high": 2,
        "medium": 3,
        "lower": 4,
    }

    result["band_order"] = (
        result["selection_band"].map(band_order)
    )

    result = result.sort_values(
        ["band_order", "report_id"]
    ).drop(
        columns=[
            "band_order",
            "combo",
            "department_frequency",
            "combo_frequency",
        ],
        errors="ignore",
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print("=" * 60)
    print("RAKSHAK GOLD SET CANDIDATES")
    print("=" * 60)

    print(f"Total candidates: {len(result)}")

    print()
    print("Selection bands:")
    print(
        result["selection_band"]
        .value_counts()
        .to_string()
    )

    print()
    print("Potential Accident Level:")
    print(
        result["potential_accident_level"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Industry:")
    print(
        result["department"]
        .value_counts()
        .to_string()
    )

    print()
    print("Actual Accident Level:")
    print(
        result["actual_outcome"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Output:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()