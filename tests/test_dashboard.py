from fastapi.testclient import TestClient

from scripts import rakshak_api as api


def test_dashboard_is_served():
    with TestClient(api.app) as client:
        response = client.get("/dashboard/")
    assert response.status_code == 200
    assert "SIF-Insight" in response.text
    assert "Analyze report" in response.text
