import os
from pathlib import Path
import sqlite3
from fastapi.testclient import TestClient
from scripts import rakshak_api as api


def test_reviews_require_known_report_and_use_temp_db(monkeypatch, tmp_path):
    known = api.meta_df()["report_id"].astype(str).iloc[0]
    db_path = tmp_path / "reviews.db"
    monkeypatch.setattr(api, "REVIEW_DIR", tmp_path)
    monkeypatch.setattr(api, "DB_FILE", db_path)
    with TestClient(api.app) as client:
        saved = client.post("/reviews", json={"report_id": known, "decision": "CONFIRMED", "reviewer": "pytest", "note": "test review"})
        assert saved.status_code == 200
        assert client.post("/reviews", json={"report_id": "TEST-UNKNOWN", "decision": "CONFIRMED"}).status_code == 404
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 1


def test_health_failure_is_generic(monkeypatch):
    old_lazy = os.environ.get("RAKSHAK_LAZY_LOAD")
    os.environ["RAKSHAK_LAZY_LOAD"] = "1"
    def fail():
        raise RuntimeError("SECRET_INTERNAL_PATH C:/private/model.bin")
    monkeypatch.setattr(api, "load_resources", fail)
    try:
        with TestClient(api.app) as client:
            r = client.get("/health")
            assert r.status_code == 503
            assert "SECRET_INTERNAL_PATH" not in r.text
            assert "C:/private/model.bin" not in r.text
    finally:
        if old_lazy is None: os.environ.pop("RAKSHAK_LAZY_LOAD", None)
        else: os.environ["RAKSHAK_LAZY_LOAD"] = old_lazy
