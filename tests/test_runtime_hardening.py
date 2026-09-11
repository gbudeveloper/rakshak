from fastapi.testclient import TestClient
from scripts import rakshak_api as api

def test_root_redirects_to_dashboard():
    with TestClient(api.app) as c:
        r=c.get("/",follow_redirects=False)
    assert r.status_code==307 and r.headers["location"]=="/dashboard/"

def test_favicon_is_served():
    with TestClient(api.app) as c: r=c.get("/favicon.ico")
    assert r.status_code==200 and "svg" in r.headers.get("content-type","")

def test_resource_object_is_reused():
    with TestClient(api.app) as c:
        c.get("/health"); first=api._ENCODER; c.get("/model-info"); second=api._ENCODER
    assert first is second
