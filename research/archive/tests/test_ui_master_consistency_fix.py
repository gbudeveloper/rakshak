from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_ui_master_consistency_tokens():
    html=(ROOT/"apps"/"web"/"index.html").read_text(encoding="utf-8")
    js=(ROOT/"apps"/"web"/"app.js").read_text(encoding="utf-8")
    css=(ROOT/"apps"/"web"/"styles.css").read_text(encoding="utf-8")
    assert "Analyze report" in html
    assert 'id="descriptionCount"' in html
    assert 'function renderAdHocNotice' in js
    assert 'event.ctrlKey || event.metaKey' in js
    assert 'class="queue-toolbar queue-tools"' in html
    assert ".queue-tools" in css
    assert ".search-control" in css
    assert "@media (max-width: 640px)" in css
