import os
import re
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.integration


def normalize_report_id(value) -> str:
    """Normalize report IDs such as industrial_safety_000422 -> 000422."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""

    # Canonical source IDs are commonly prefixed, e.g.
    # industrial_safety_000422. Keep only the trailing numeric token.
    match = re.search(r"(\d+)(?:\.0)?$", text)
    if match:
        return match.group(1).zfill(6)

    return text


@pytest.mark.skipif(
    os.getenv("RAKSHAK_RUN_MODEL_TESTS") != "1",
    reason="Set RAKSHAK_RUN_MODEL_TESTS=1 to run frozen-model integration tests",
)
def test_api_feature_builder_matches_frozen_prediction_artifact():
    from fastapi.testclient import TestClient
    from scripts import rakshak_api as api

    root = Path(__file__).resolve().parents[1]
    pool = pd.read_csv(
        root / "data" / "annotations" / "industrial_safety_annotation_pool.csv"
    )
    pred = pd.read_csv(root / "experiments" / "industrial_safety_predictions_v1.4.csv")

    pool["_report_id"] = pool["report_id"].map(normalize_report_id)
    pred["_report_id"] = pred["report_id"].map(normalize_report_id)

    ids = ["000422", "000364", "000145"]

    missing_pool = [rid for rid in ids if rid not in set(pool["_report_id"])]
    assert not missing_pool, f"Missing report IDs in annotation pool: {missing_pool}"

    missing_pred = [rid for rid in ids if rid not in set(pred["_report_id"])]
    assert (
        not missing_pred
    ), f"Missing report IDs in prediction artifact: {missing_pred}"

    descriptions = {
        rid: pool.loc[pool["_report_id"] == rid, "description"].iloc[0] for rid in ids
    }

    expected = {
        rid: float(
            pred.loc[
                pred["_report_id"] == rid,
                "sif_precursor_probability",
            ].iloc[0]
        )
        for rid in ids
    }

    with TestClient(api.app) as client:
        for rid in ids:
            response = client.post(
                "/predict",
                json={"report_id": rid, "description": descriptions[rid]},
            )
            assert response.status_code == 200, response.text
            actual = float(response.json()["sif_precursor_probability"])
            assert abs(actual - expected[rid]) < 1e-5, (
                f"Prediction mismatch for {rid}: API={actual:.12f}, "
                f"artifact={expected[rid]:.12f}"
            )
