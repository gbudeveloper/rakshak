from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_analytics_runtime_v2_uses_real_visual_sections():
    html = (ROOT / "apps" / "web" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "apps" / "web" / "app.js").read_text(encoding="utf-8")
    css = (ROOT / "apps" / "web" / "styles.css").read_text(encoding="utf-8")
    for token in [
        'id="priorityChart"',
        'id="signalFunnel"',
        'id="hazardsChart"',
        'id="exposuresChart"',
        'id="pathwaysChart"',
        'id="lsrChart"',
        'id="activitiesChart"',
        'id="locationsChart"',
        "function safeNode",
        "Promise.allSettled",
        "model_yes_ge_0_50",
    ]:
        assert token in html + js
    assert ".horizontal-chart" in css
    assert ".priority-chart" in css
    assert ".hotspot-chart" in css


def test_analytics_runtime_v2_removes_visible_detail_controls():
    html = (ROOT / "apps" / "web" / "index.html").read_text(encoding="utf-8")
    assert 'class="ghost compact detail-toggle"' not in html
    assert 'aria-hidden="true" tabindex="-1"' in html
