from pathlib import Path


def test_new_analysis_clears_existing_context_controls():
    root = Path(__file__).resolve().parents[1]
    html = (root / "apps" / "web" / "index.html").read_text(encoding="utf-8")
    js = (root / "apps" / "web" / "app.js").read_text(encoding="utf-8")
    css = (root / "apps" / "web" / "styles.css").read_text(encoding="utf-8")

    for needle in [
        'id="startNewAnalysis"',
        'id="startNewAnalysisInline"',
        'id="analysisMode"',
        'id="analysisContext"',
        'id="descriptionCount"',
        'function startNewAnalysis',
        'state.analysisContext = "new"',
        'Previous report data is cleared',
    ]:
        assert needle in (html + js)
    assert "context-banner" in css
    assert "queue-tools" in css


def test_dashboard_has_accessible_analysis_fields_and_keyboard_shortcuts():
    root = Path(__file__).resolve().parents[1]
    html = (root / "apps" / "web" / "index.html").read_text(encoding="utf-8")
    js = (root / "apps" / "web" / "app.js").read_text(encoding="utf-8")
    assert 'for="reportId"' in html and 'for="description"' in html
    assert 'aria-live="polite"' in html
    assert 'event.key === "/"' in js
    assert 'event.key === "Escape"' in js
    assert 'event.ctrlKey || event.metaKey' in js


def test_review_controls_are_blocked_for_fresh_narratives():
    root = Path(__file__).resolve().parents[1]
    js = (root / "apps" / "web" / "app.js").read_text(encoding="utf-8")
    assert 'state.analysisContext === "existing"' in js
    assert 'Persistent review is available after this narrative is linked to an existing queue report.' in js
