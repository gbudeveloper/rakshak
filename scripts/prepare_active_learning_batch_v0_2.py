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

POOL_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "industrial_safety_annotation_pool.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "active_learning_v0.1"
    / "active_learning_batch_01_annotation.csv"
)

REQUIRED_APP_COLUMNS = [
    "report_id",
    "description",
    "department",
    "site",
    "location",
    "hazards",
]

ANNOTATION_DEFAULTS = {
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


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def main() -> None:
    if not SELECTED_PATH.exists():
        raise FileNotFoundError(
            f"Active-learning selection file not found:\n{SELECTED_PATH}\n\n"
            "Run select_active_learning_batch_v0_1.py first."
        )

    if not POOL_PATH.exists():
        raise FileNotFoundError(
            f"Annotation pool not found:\n{POOL_PATH}"
        )

    selected = pd.read_csv(SELECTED_PATH)
    pool = pd.read_csv(POOL_PATH)

    require_columns(
        selected,
        ["report_id", "description"],
        "Selected batch",
    )

    require_columns(
        pool,
        REQUIRED_APP_COLUMNS,
        "Canonical annotation pool",
    )

    selected = selected.copy()
    pool = pool.copy()

    selected["report_id"] = normalize_id(selected["report_id"])
    pool["report_id"] = normalize_id(pool["report_id"])

    # Keep the selector's order/rank.
    if "active_learning_rank" in selected.columns:
        selected["active_learning_rank"] = pd.to_numeric(
            selected["active_learning_rank"],
            errors="coerce",
        )
    else:
        selected["active_learning_rank"] = range(1, len(selected) + 1)

    # Pull the complete canonical fields from the original annotation
    # pool. This fixes the missing department/site/location/hazards issue
    # without manufacturing any metadata.
    pool_columns = [
        "report_id",
        "source",
        "report_type",
        "timestamp",
        "site",
        "location",
        "department",
        "activity",
        "asset",
        "description",
        "unsafe_act",
        "unsafe_condition",
        "near_miss",
        "hi_po",
        "hazards",
        "exposure",
        "potential_consequences",
        "barriers",
        "barrier_status",
        "life_saving_rules",
        "actual_outcome",
        "potential_accident_level",
        "source_gender",
        "source_actor_type",
    ]

    pool_columns = [
        column
        for column in pool_columns
        if column in pool.columns
    ]

    canonical = pool[pool_columns].drop_duplicates(
        subset=["report_id"],
        keep="first",
    )

    output = selected[
        [
            column
            for column in [
                "batch_id",
                "active_learning_rank",
                "selection_reason",
                "tfidf_probability",
                "transformer_probability",
                "hybrid_probability",
                "model_disagreement",
                "boundary_uncertainty",
                "ranking_score",
                "report_id",
            ]
            if column in selected.columns
        ]
    ].copy()

    output = output.merge(
        canonical,
        on="report_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_pool"),
    )

    # The canonical pool's description is authoritative.
    if "description_pool" in output.columns:
        output["description"] = output["description_pool"]
        output = output.drop(columns=["description_pool"])

    # Initialize/retain annotation columns.
    for column, default in ANNOTATION_DEFAULTS.items():
        if column not in output.columns:
            output[column] = default

    output["annotation_status"] = (
        output["annotation_status"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    output.loc[
        output["annotation_status"].eq(""),
        "annotation_status",
    ] = "pending"

    output["annotation_version"] = "v0.3"
    output["annotation_round"] = "active_learning_01"

    # Ensure the fields expected by the Streamlit app are present.
    require_columns(
        output,
        REQUIRED_APP_COLUMNS,
        "Prepared annotation batch",
    )

    if output[REQUIRED_APP_COLUMNS].isnull().all().any():
        bad = [
            column
            for column in REQUIRED_APP_COLUMNS
            if output[column].isnull().all()
        ]
        raise ValueError(
            f"Prepared batch contains completely empty required columns: {bad}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("=" * 78)
    print("RAKSHAK ACTIVE LEARNING ANNOTATION BATCH PREPARATION v0.2")
    print("=" * 78)
    print(f"Selected records      : {len(selected)}")
    print(f"Prepared records      : {len(output)}")
    print(f"Output                : {OUTPUT_PATH}")
    print()
    print("Required app columns:")
    for column in REQUIRED_APP_COLUMNS:
        print(f"  {column:<14}: OK")
    print()
    print("Metadata source: industrial_safety_annotation_pool.csv")
    print("Model scores retained only for reproducibility.")
    print("Model scores are not intended to influence human labels.")
    print("=" * 78)


if __name__ == "__main__":
    main()
