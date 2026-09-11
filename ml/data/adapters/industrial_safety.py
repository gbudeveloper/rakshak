from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "Unnamed: 0",
    "Data",
    "Countries",
    "Local",
    "Industry Sector",
    "Accident Level",
    "Potential Accident Level",
    "Genre",
    "Employee or Third Party",
    "Critical Risk",
    "Description",
}


def _clean(value: Any) -> str | None:
    """Convert a dataframe value into a clean string or None."""
    if pd.isna(value):
        return None

    value = str(value).strip()

    return value if value else None


def _split_semicolon(value: Any) -> list[str]:
    """Convert a simple source field into a list."""
    value = _clean(value)

    if not value:
        return []

    return [item.strip() for item in value.split(";") if item.strip()]


def validate_source_columns(df: pd.DataFrame) -> None:
    """Validate the raw Industrial Safety dataset schema."""
    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(
            "Industrial Safety dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )


def load_industrial_safety(path: str | Path) -> pd.DataFrame:
    """Load the original Industrial Safety CSV without modifying it."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    validate_source_columns(df)

    return df


def convert_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the Industrial Safety dataset into the RAKSHAK canonical schema.

    Important:
    Potential Accident Level is preserved as source metadata.
    It is NOT converted into sif_potential.
    """
    validate_source_columns(df)

    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        report_id = f"industrial_safety_{int(row['Unnamed: 0']):06d}"

        timestamp = pd.to_datetime(
            row["Data"],
            errors="coerce",
        )

        records.append(
            {
                "report_id": report_id,
                "source": "industrial_safety_analytics_database",
                "report_type": "incident",

                "timestamp": (
                    timestamp.isoformat()
                    if not pd.isna(timestamp)
                    else None
                ),

                "site": _clean(row["Countries"]),
                "location": _clean(row["Local"]),
                "department": _clean(row["Industry Sector"]),
                "activity": None,
                "asset": None,

                "description": _clean(row["Description"]),

                "unsafe_act": None,
                "unsafe_condition": None,
                "near_miss": None,
                "hi_po": None,

                "hazards": _split_semicolon(row["Critical Risk"]),
                "exposure": [],
                "potential_consequences": [],

                "barriers": [],
                "barrier_status": [],

                "life_saving_rules": [],

                "actual_outcome": _clean(row["Accident Level"]),

                "potential_accident_level": _clean(
                    row["Potential Accident Level"]
                ),

                # Deliberately unlabelled.
                "sif_potential": None,

                "annotation_confidence": None,
                "annotation_version": None,

                # Preserve useful source metadata.
                "source_gender": _clean(row["Genre"]),
                "source_actor_type": _clean(
                    row["Employee or Third Party"]
                ),
            }
        )

    return pd.DataFrame(records)


def load_and_convert(path: str | Path) -> pd.DataFrame:
    """Load and convert the dataset in one operation."""
    raw_df = load_industrial_safety(path)
    return convert_to_canonical(raw_df)