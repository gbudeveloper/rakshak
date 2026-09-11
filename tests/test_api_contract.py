from pathlib import Path
import sqlite3

import pandas as pd
from fastapi.testclient import TestClient

from scripts import rakshak_api as api


def _fake_resources():
    class FakeModel:
        n_features_in_ = 422

    pkg = {"reference_threshold": 0.5}
    return pkg, FakeModel(), object(), "cpu"


def _fake_predict_records(records):
    return [
        {
            "report_id": r.report_id,
            "description": r.description.strip(),
            "sif_precursor_probability": 0.72,
            "prediction_at_0_50": True,
            "review": {
                "priority": "P3",
                "reason": "MODERATE_MODEL_SCORE",
                "human_review_required": True,
            },
            "evidence": {
                "hazards": "vehicle",
                "exposures": "",
                "pathways": "",
                "hazard_signal": 1,
                "exposure_signal": 0,
                "pathway_signal": 0,
                "complete_pathway": 0,
                "sif_context_status": "UNKNOWN",
            },
            "lsr_candidates": [],
            "governance": {
                "model_input": "Narrative description only",
                "autonomous_sif_decision": False,
                "requires_hsse_review": True,
            },
        }
        for r in records
    ]


def test_health_and_model_info_contract(monkeypatch):
    monkeypatch.setattr(api, "load_resources", _fake_resources)
    with TestClient(api.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "ok"
        assert body["model_input_dimension"] == 422
        assert body["model_version"] == "rakshak_final_v1.2"
        assert body["review_policy"] == "v1.4"
        assert body["evidence_engine"] == "v1.5"

        info = client.get("/model-info")
        assert info.status_code == 200
        info_body = info.json()
        assert info_body["raw_features"] == 422
        assert info_body["embedding_dimension"] == 384
        assert info_body["pathway_features"] == 38


def test_predict_contract_without_loading_model(monkeypatch):
    monkeypatch.setattr(api, "predict_records", _fake_predict_records)
    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            json={"report_id": "TEST-001", "description": "Vehicle moved near worker."},
        )
        assert response.status_code == 200
        body = response.json()
        assert 0 <= body["sif_precursor_probability"] <= 1
        assert body["review"]["priority"] in {"P1", "P2", "P3", "P4", "P5"}
        assert body["governance"]["autonomous_sif_decision"] is False


def test_reviews_use_a_temporary_database(monkeypatch, tmp_path):
    db_path = tmp_path / "reviews.db"

    monkeypatch.setattr(api, "REVIEW_DIR", tmp_path)
    monkeypatch.setattr(api, "DB_FILE", db_path)

    # This test is about temporary SQLite persistence, not
    # report-existence validation.
    monkeypatch.setattr(api, "report_exists", lambda report_id: True)

    with TestClient(api.app) as client:
        saved = client.post(
            "/reviews",
            json={
                "report_id": "TEST-001",
                "decision": "CONFIRMED",
                "reviewer": "pytest",
                "note": "test review",
            },
        )
        assert saved.status_code == 200
        assert saved.json()["status"] == "saved"

        detail = client.get("/reviews/TEST-001")
        assert detail.status_code == 200
        assert detail.json()["reviews"][0]["decision"] == "CONFIRMED"

        stats = client.get("/review-stats")
        assert stats.status_code == 200
        assert stats.json()["total_reviews"] == 1

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 1


def test_analytics_contract():
    with TestClient(api.app) as client:
        overview = client.get("/analytics/overview")
        assert overview.status_code == 200
        body = overview.json()
        assert body["records"] == 411
        assert body["priority_distribution"] == {
            "P1": 120,
            "P2": 67,
            "P3": 80,
            "P4": 45,
            "P5": 99,
        }
        assert body["activity_metadata_available"] is True
        assert "not ground-truth SIF prevalence" in body["warning"]

        activities = client.get("/analytics/activities")
        assert activities.status_code == 200
        activity_body = activities.json()
        assert activity_body["status"] == "OK"
        assert activity_body["authoritative"] is False
        assert len(activity_body["items"]) > 0
