from __future__ import annotations

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANNOTATED_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "active_learning_v0.1"
    / "active_learning_batch_01_annotated.csv"
)

POOL_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "industrial_safety_annotation_pool.csv"
)

OUTPUT_PATH = ANNOTATED_PATH

def main() -> None:
    if not ANNOTATED_PATH.exists():
        raise FileNotFoundError(f"Missing annotated file:\n{ANNOTATED_PATH}")
    if not POOL_PATH.exists():
        raise FileNotFoundError(f"Missing canonical pool:\n{POOL_PATH}")

    annotated = pd.read_csv(ANNOTATED_PATH)
    pool = pd.read_csv(POOL_PATH)

    for name, df in [("annotated file", annotated), ("canonical pool", pool)]:
        if "report_id" not in df.columns:
            raise ValueError(f"{name} is missing report_id")

    annotated["report_id"] = annotated["report_id"].astype(str).str.strip()
    pool["report_id"] = pool["report_id"].astype(str).str.strip()

    if "duplicate_group_id" not in pool.columns:
        # Canonical pool in this project can lack the validator's
        # helper column. Build it deterministically from normalized
        # narrative text, so identical narratives share a group.
        if "description" not in pool.columns:
            raise ValueError("Canonical pool has neither duplicate_group_id nor description")
        normalized = (
            pool["description"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        codes, _ = pd.factorize(normalized, sort=True)
        pool["duplicate_group_id"] = [
            f"dup_{int(code):04d}" if text else ""
            for code, text in zip(codes, normalized)
        ]

    pool_meta = pool[
        ["report_id", "duplicate_group_id"]
    ].drop_duplicates("report_id")

    # Do not overwrite any annotation content. Only add missing validator metadata.
    merged = annotated.drop(
        columns=["duplicate_group_id"],
        errors="ignore"
    ).merge(
        pool_meta,
        on="report_id",
        how="left",
        validate="many_to_one",
    )

    missing = merged["duplicate_group_id"].isna()
    if missing.any():
        bad_ids = merged.loc[missing, "report_id"].tolist()
        raise ValueError(
            "Could not resolve duplicate_group_id for report IDs: "
            + ", ".join(bad_ids)
        )

    # Keep deterministic string IDs.
    merged["duplicate_group_id"] = (
        merged["duplicate_group_id"].fillna("").astype(str).str.strip()
    )

    merged.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("=" * 78)
    print("RAKSHAK ACTIVE LEARNING — VALIDATOR COLUMN REPAIR v0.1")
    print("=" * 78)
    print(f"Records repaired : {len(merged)}")
    print(f"Output           : {OUTPUT_PATH}")
    print("Added/normalized : duplicate_group_id")
    print("Annotation fields were not modified.")
    print("=" * 78)

if __name__ == "__main__":
    main()
