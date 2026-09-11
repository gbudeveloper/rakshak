import os
from fastapi.testclient import TestClient


def test_queue_handles_nullable_metadata(monkeypatch):
    monkeypatch.setenv("RAKSHAK_LAZY_LOAD", "1")
    from scripts import rakshak_api as api

    original = api.meta_df
    original_reviews = api.latest_review_map
    import pandas as pd

    frame = pd.DataFrame([{
        "report_id": "industrial_safety_000422",
        "description": "test description",
        "sif_precursor_probability": 0.9,
        "review_priority": "P1",
        "activity": pd.NA,
        "site": pd.NA,
        "location": pd.NA,
        "department": pd.NA,
    }])
    api.meta_df = lambda: frame.copy()
    api.latest_review_map = lambda: {}
    try:
        with TestClient(api.app) as client:
            response = client.get("/queue?limit=10")
            assert response.status_code == 200, response.text
            item = response.json()["items"][0]
            assert item["activity"] == ""
            assert item["site"] == ""
            assert item["location"] == ""
            assert item["department"] == ""
    finally:
        api.meta_df = original
        api.latest_review_map = original_reviews
