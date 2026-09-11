from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data" / "annotations" / "industrial_safety_annotation_pool.csv"
PRED = ROOT / "experiments" / "industrial_safety_predictions_v1.4.csv"
EXTRACT = (
    ROOT
    / "experiments"
    / "safety_information_extraction_v0.5"
    / "safety_extraction_flat.csv"
)


def test_canonical_pool_integrity():
    df = pd.read_csv(POOL)
    assert len(df) == 411
    assert {"report_id", "description"}.issubset(df.columns)
    assert df["report_id"].astype(str).is_unique
    assert df["description"].fillna("").astype(str).str.strip().ne("").all()


def test_prediction_artifact_integrity():
    df = pd.read_csv(PRED)
    assert len(df) == 411
    assert df["report_id"].astype(str).is_unique
    assert df["sif_precursor_probability"].between(0, 1).all()
    assert set(df["review_priority"].dropna().unique()) <= {
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
    }

    counts = (
        df["review_priority"]
        .value_counts()
        .reindex(["P1", "P2", "P3", "P4", "P5"], fill_value=0)
    )
    assert counts.to_dict() == {"P1": 120, "P2": 67, "P3": 80, "P4": 45, "P5": 99}


def test_activity_metadata_integrity():
    df = pd.read_csv(EXTRACT)
    assert len(df) == 411
    assert df["report_id"].astype(str).is_unique
    assert df["activities"].fillna("").astype(str).str.strip().ne("").sum() > 0
