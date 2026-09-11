from pathlib import Path


def test_dashboard_contains_review_controls_and_core_sections():
    root = Path(__file__).resolve().parents[1]
    html = (root / "apps" / "web" / "index.html").read_text(encoding="utf-8")
    js = (root / "apps" / "web" / "app.js").read_text(encoding="utf-8")
    css = (root / "apps" / "web" / "styles.css").read_text(encoding="utf-8")

    for text in [
        "overview",
        "queue",
        "analytics",
        "analyze",
        "reviewDecision",
        "saveReview",
    ]:
        assert text in (html + js)
    assert "POST /reviews" in js or "'/reviews'" in js or '"/reviews"' in js
    css_compact = "".join(css.split())
    assert "@media(max-width:640px)" in css_compact
    assert "Human HSSE review" in (html + js)


def test_dashboard_does_not_use_inline_event_handlers_for_queue_rows():
    root = Path(__file__).resolve().parents[1]
    js = (root / "apps" / "web" / "app.js").read_text(encoding="utf-8")
    assert "onclick=" not in js
