from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_queue_search_ui_has_clearable_search_and_filters():
    html = (ROOT / "apps" / "web" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "apps" / "web" / "app.js").read_text(encoding="utf-8")
    css = (ROOT / "apps" / "web" / "styles.css").read_text(encoding="utf-8")

    for token in ["queueSearch", "clearQueueSearch", "clearQueueFilters", "queueFilterHint"]:
        assert token in html + js
    assert "queue-toolbar" in css
    assert "@media (max-width: 640px)" in css


def test_queue_search_filters_priority_status_and_narrative():
    js = (ROOT / "apps" / "web" / "app.js").read_text(encoding="utf-8")
    assert "x.priority" in js or "review_priority" in js
    assert "x.review_status" in js
    assert "x.description" in js
