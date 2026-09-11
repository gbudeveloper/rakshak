import importlib


def test_offline_mode_defaults_on(monkeypatch):
    monkeypatch.delenv("RAKSHAK_HF_LOCAL_ONLY", raising=False)
    import scripts.rakshak_api as api

    importlib.reload(api)
    assert api._offline_mode_enabled() is True


def test_offline_mode_can_be_disabled(monkeypatch):
    monkeypatch.setenv("RAKSHAK_HF_LOCAL_ONLY", "0")
    import scripts.rakshak_api as api

    importlib.reload(api)
    assert api._offline_mode_enabled() is False


def test_cors_is_not_wildcard_by_default(monkeypatch):
    monkeypatch.delenv("RAKSHAK_CORS_ORIGINS", raising=False)
    import scripts.rakshak_api as api

    importlib.reload(api)
    assert "*" not in api._CORS_ORIGINS
    assert "http://127.0.0.1:8000" in api._CORS_ORIGINS
