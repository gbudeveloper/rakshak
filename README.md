# SIH26165 — SIF-Insight

**Team:** RAKSHAK  
**Problem:** SIH26165 — AI/NLP Engine to Detect Serious Injury & Fatality (SIF) Precursors in OIL Unsafe-Act/Unsafe-Condition and Near-Miss Reports  
**Organization:** Oil India Limited  
**Category:** Software  
**Theme:** Miscellaneous

## Project

SIF-Insight is a human-in-the-loop safety intelligence prototype that turns safety narratives into a triage signal, evidence-backed reviewer support, and recurring-pattern analytics. It does **not** make an autonomous SIF determination.

## Current release stack

| Component | Release |
|---|---|
| API | 1.6.0 |
| Frozen classifier | RAKSHAK final v1.2 |
| Reviewer priority policy | v1.4 |
| Evidence engine | v1.5 |
| Activity metadata | v0.5 |
| Prediction artifact | v1.4 |
| Model input | 422 features (384-d embedding + 38 pathway features) |
| Review store | SQLite, created at runtime |
| UI | Static HTML/CSS/JavaScript dashboard served by FastAPI |

## System flow

```text
Safety narrative
      ↓
MiniLM embedding (384)
      +
38 pathway features
      ↓
422-feature frozen classifier
      ↓
SIF precursor probability
      ↓
P1–P5 reviewer prioritization
      +
Evidence / pathway signals
      +
IOGP Life-Saving Rule candidates
      ↓
Human HSSE review
      ↓
Persistent review + pattern analytics
```

## Quick start — Windows + NVIDIA GPU

Use Python 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project was validated with the NVIDIA CUDA build of PyTorch. When a fresh environment installs the CPU wheel, install the tested CUDA 12.6 build before running the model:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
```

Verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

For the final demo, keep the encoder local-only:

```powershell
$env:RAKSHAK_HF_LOCAL_ONLY="1"
uvicorn scripts.rakshak_api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`.

## Validation

Run the deterministic suite:

```powershell
pytest -q
```

Run the frozen-model integration suite:

```powershell
$env:RAKSHAK_RUN_MODEL_TESTS="1"
pytest -q -m integration
```

The final working development environment was validated with the full suite and frozen-model integration tests before release packaging.

## Repository structure

```text
apps/web/                 Judge-facing dashboard
apps/annotation/          Human annotation/research tooling
data/                     Public/development datasets and annotation artifacts
experiments/              Current frozen model + runtime prediction artifacts
scripts/                  Canonical runtime and reproducibility scripts
research/archive/         Historical experiments and superseded implementations
tests/                    Automated regression + integration tests
docs/                     Protocols, runbooks, acceptance documentation
```

The `research/archive/` directory is intentionally separated from the active runtime path so the submission remains readable while preserving the project's research history.

## Human-in-the-loop safety posture

The model output is a ranking/triage signal. Human HSSE review remains required before a safety conclusion is accepted. Evidence and Life-Saving Rule outputs are reviewer-support signals, not authoritative OIL classifications.

## Research limitations

The labeled research set is small. Reported model metrics are prototype validation results and should not be presented as production-grade generalization. The current activity metadata is narrative-derived heuristic metadata and is not authoritative OIL activity labeling.

## Demo reset

The runtime review database is intentionally excluded from source control. To reset a local demo database:

```powershell
python scripts\reset_demo_reviews.py
```

## Developer ownership

Developed as the SIH 2026 RAKSHAK team project. The repository is organized around one canonical runtime path; historical experiments are preserved separately for transparency.
