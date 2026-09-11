from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fresh_analysis_and_analytics_details_ui_contract():
    html = (ROOT / "apps" / "web" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "apps" / "web" / "app.js").read_text(encoding="utf-8")
    css = (ROOT / "apps" / "web" / "styles.css").read_text(encoding="utf-8")

    for token in [
        'id="startNewAnalysis"',
        'id="startNewAnalysisInline"',
        'id="analysisMode"',
        'id="analysisContext"',
        'id="descriptionCount"',
        'function startNewAnalysis',
        'state.analysisContext = "new"',
        'Previous report data is cleared',
        'id="expandAnalytics"',
        'id="hazardsDetails"',
        'id="activitiesDetails"',
        'function renderAnalyticsSummary',
        'function renderCountDetails',
    ]:
        assert token in html + js

    assert "analytics-summary" in css
    assert "analytics-details" in css
    assert "@media (max-width: 640px)" in css


def test_fresh_analysis_does_not_render_persistent_review_controls_by_default():
    js = (ROOT / "apps" / "web" / "app.js").read_text(encoding="utf-8")
    assert 'state.analysisContext === "existing" && d.report_id' in js
    assert "renderAdHocNotice" in js


def test_queue_search_clear_control_exists():
    html = (ROOT / "apps" / "web" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "apps" / "web" / "app.js").read_text(encoding="utf-8")
    assert 'id="queueSearch"' in html
    assert 'id="clearQueueSearch"' in html
    assert 'queueSearch' in js and 'clearQueueSearch' in js
