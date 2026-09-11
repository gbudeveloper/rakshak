from fastapi.testclient import TestClient


def test_unknown_report_review_is_rejected():
    from scripts import rakshak_api as api
    with TestClient(api.app) as client:
        r = client.post("/reviews", json={
            "report_id": "industrial_safety_DOES_NOT_EXIST",
            "decision": "CONFIRMED",
            "reviewer": "tester",
            "note": "should not be stored",
        })
    assert r.status_code == 404


def test_unknown_report_endpoint_is_404():
    from scripts import rakshak_api as api
    with TestClient(api.app) as client:
        r = client.get("/reports/industrial_safety_DOES_NOT_EXIST")
    assert r.status_code == 404


def test_health_failure_does_not_expose_internal_error(monkeypatch):
    from scripts import rakshak_api as api
    def fail():
        raise RuntimeError("SECRET_INTERNAL_PATH C:/private/model.bin")
    monkeypatch.setattr(api, "load_resources", fail)
    with TestClient(api.app) as client:
        r = client.get("/health")
    assert r.status_code == 503
    assert "SECRET_INTERNAL_PATH" not in r.text
    assert "private/model.bin" not in r.text
