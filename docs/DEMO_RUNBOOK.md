# RAKSHAK / SIF-Insight — SIH 2026 Demo Runbook

## Start

```powershell
.\.venv\Scripts\Activate.ps1
$env:RAKSHAK_HF_LOCAL_ONLY="1"
uvicorn scripts.rakshak_api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`.

## Demo flow

1. **Overview** — show dataset scale, triage mix, and human-review posture.
2. **Review Queue** — search/filter a P1/P2 report.
3. **Incident Analysis** — inspect probability, evidence, pathway, and Life-Saving Rule candidates.
4. **Human Review** — submit a decision and reopen the report to show persistence.
5. **Analytics** — show recurring hazards, exposure/pathway signals, LSR candidates, and operational hotspots.
6. **Fresh Analysis** — clear prior report state, enter a new narrative, and run an ad-hoc analysis.

## Demo reset

```powershell
python scripts\reset_demo_reviews.py
```

The runtime database is intentionally excluded from the source release.

## Safety posture

The model produces a triage/ranking signal. Human HSSE review remains required before accepting a safety conclusion. Evidence and Life-Saving Rule outputs are reviewer-support signals rather than authoritative OIL classifications.
