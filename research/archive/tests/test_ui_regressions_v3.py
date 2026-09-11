from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_new_analysis_has_explicit_fresh_context_reset():
    html = (ROOT / "apps" / "web" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "apps" / "web" / "app.js").read_text(encoding="utf-8")
    assert 'id="startNewAnalysis"' in html
    assert 'id="analysisContext"' in html
    assert 'function startNewAnalysis' in js
    assert 'state.analysisContext = "new"' in js
    assert "Previous report data is cleared" in html + js


def test_queue_search_has_dedicated_clear_and_live_hint():
    html = (ROOT / "apps" / "web" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "apps" / "web" / "app.js").read_text(encoding="utf-8")
    css = (ROOT / "apps" / "web" / "styles.css").read_text(encoding="utf-8")
    for token in ["queueSearch", "clearQueueSearch", "clearQueueFilters", "queueFilterHint"]:
        assert token in html + js
    assert 'addEventListener("click"' in js
    assert ".search-control" in css


def test_mobile_breakpoint_is_present():
    css = (ROOT / "apps" / "web" / "styles.css").read_text(encoding="utf-8")
    assert "@media (max-width: 640px)" in css
