import os
from pathlib import Path
import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RAKSHAK_RUN_MODEL_TESTS") != "1",
    reason="Set RAKSHAK_RUN_MODEL_TESTS=1 to run end-to-end acceptance tests",
)
def test_full_hsse_review_workflow(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from scripts import rakshak_api as api

    # Isolate acceptance reviews from the developer/demo database.
    monkeypatch.setattr(api, "REVIEW_DIR", tmp_path)
    monkeypatch.setattr(api, "DB_FILE", tmp_path / "acceptance_reviews.db")
    api._POOL_CACHE = None
    api._EXTRACT_CACHE = None
    api._META_CACHE = None

    with TestClient(api.app) as client:
        root = client.get("/", follow_redirects=False)
        assert root.status_code == 307
        assert root.headers["location"] == "/dashboard/"

        dashboard = client.get("/dashboard/")
        assert dashboard.status_code == 200
        assert "SIF-Insight" in dashboard.text

        health = client.get("/health")
        assert health.status_code == 200
        health_json = health.json()
        assert health_json["status"] == "ok"
        assert health_json["human_in_loop"] is True
        assert health_json["model_input_dimension"] == 422

        queue = client.get("/queue", params={"limit": 5})
        assert queue.status_code == 200
        queue_json = queue.json()
        assert queue_json["items"]
        report_id = queue_json["items"][0]["report_id"]

        report = client.get(f"/reports/{report_id}")
        assert report.status_code == 200
        report_json = report.json()
        assert report_json["report_id"] == report_id
        assert 0.0 <= report_json["sif_precursor_probability"] <= 1.0
        assert report_json["human_review_required"] if "human_review_required" in report_json else True
        assert report_json["governance"]["requires_hsse_review"] is True
        assert report_json["governance"]["autonomous_sif_decision"] is False

        review_payload = {
            "report_id": report_id,
            "decision": "NEEDS_MORE_INFO",
            "reviewer": "acceptance-test",
            "note": "End-to-end acceptance test; synthetic review metadata only.",
        }
        saved = client.post("/reviews", json=review_payload)
        assert saved.status_code == 200
        assert saved.json()["status"] == "saved"

        report_after = client.get(f"/reports/{report_id}")
        assert report_after.status_code == 200
        reviews = report_after.json()["reviews"]
        assert reviews[0]["decision"] == "NEEDS_MORE_INFO"
        assert reviews[0]["reviewer"] == "acceptance-test"

        stats = client.get("/review-stats")
        assert stats.status_code == 200
        stats_json = stats.json()
        assert stats_json["total_reviews"] == 1
        assert any(
            row["decision"] == "NEEDS_MORE_INFO" and row["n"] == 1
            for row in stats_json["by_decision"]
        )

        analytics_paths = [
            "/analytics/overview",
            "/analytics/hazards",
            "/analytics/exposures",
            "/analytics/pathways",
            "/analytics/lsr",
            "/analytics/activities",
            "/analytics/locations",
        ]
        for path in analytics_paths:
            response = client.get(path, params={"limit": 5} if path != "/analytics/overview" else None)
            assert response.status_code == 200, path
