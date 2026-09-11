from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_analytics_uses_visual_sections_not_per_card_details_buttons():
    html=(ROOT/"apps/web/index.html").read_text(encoding="utf-8")
    js=(ROOT/"apps/web/app.js").read_text(encoding="utf-8")
    css=(ROOT/"apps/web/styles.css").read_text(encoding="utf-8")
    for token in ["priorityChart","signalFunnel","hazardsChart","pathwaysChart","locationsChart","renderHorizontalChart","renderHotspots"]:
        assert token in html+js
    assert "analytics-grid-main" in css and "horizontal-chart" in css and "hotspot-chart" in css
    assert 'class="detail-toggle' not in html

def test_analytics_keeps_existing_contract_hooks():
    html=(ROOT/"apps/web/index.html").read_text(encoding="utf-8")
    js=(ROOT/"apps/web/app.js").read_text(encoding="utf-8")
    for token in ['id="expandAnalytics"','id="hazardsDetails"','id="activitiesDetails"','function renderAnalyticsSummary','function renderCountDetails']:
        assert token in html+js
