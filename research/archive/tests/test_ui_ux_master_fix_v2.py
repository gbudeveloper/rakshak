from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return (ROOT / "apps" / "web" / name).read_text(encoding="utf-8")


def test_fresh_analysis_and_queue_controls_are_present():
    html, js, css = read("index.html"), read("app.js"), read("styles.css")
    for token in [
        'id="startNewAnalysis"', 'id="startNewAnalysisInline"', 'id="analysisMode"',
        'id="analysisContext"', 'id="descriptionCount"', 'function startNewAnalysis',
        'state.analysisContext = "new"', 'Previous report data is cleared',
        'clearQueueSearch', 'clearQueueFilters', 'queueFilterHint', 'queue-toolbar',
        '.search-control', 'event.key === "/"', 'for="reportId"', 'for="description"',
    ]:
        assert token in html + js + css
    assert 'class="queue-toolbar queue-tools"' in html
    assert "context-banner" in css


def test_analytics_details_contract_is_present():
    html, js = read("index.html"), read("app.js")
    for token in [
        'id="expandAnalytics"', 'id="hazardsDetails"', 'id="activitiesDetails"',
        "function renderAnalyticsSummary", "function renderCountDetails",
        'limit=50',
    ]:
        assert token in html + js


def test_fresh_analysis_blocks_persistent_review_for_anonymous_narrative():
    js = read("app.js")
    assert 'state.analysisContext === "existing" && d.report_id' in js
    assert 'Persistent review is available after this narrative is linked to an existing queue report.' in js
