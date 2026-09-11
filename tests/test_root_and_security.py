from fastapi.testclient import TestClient


def test_root_redirects_to_dashboard():
    from scripts.rakshak_api import app

    with TestClient(app) as client:
        r = client.get("/", follow_redirects=False)
        assert r.status_code == 307
        assert r.headers["location"] == "/dashboard/"


def test_invalid_review_decision_is_rejected():
    from scripts.rakshak_api import app

    with TestClient(app) as client:
        r = client.post(
            "/reviews",
            json={
                "report_id": "industrial_safety_test",
                "decision": "MAYBE",
                "reviewer": "test",
                "note": "x",
            },
        )
        assert r.status_code == 422
