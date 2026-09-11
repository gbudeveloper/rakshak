from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_queue_toolbar_semantic_class_is_present():
    html = (ROOT / "apps" / "web" / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "apps" / "web" / "styles.css").read_text(encoding="utf-8")
    assert 'class="queue-toolbar queue-tools"' in html
    assert ".queue-toolbar" in css
