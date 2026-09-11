# Final SIH Acceptance Checklist

## Automated

```powershell
pytest -q
$env:RAKSHAK_RUN_MODEL_TESTS="1"
pytest -q -m integration
```

## Runtime

- [ ] `/health` returns `status=ok`.
- [ ] `/` redirects to `/dashboard/`.
- [ ] Encoder loads locally with `RAKSHAK_HF_LOCAL_ONLY=1`.
- [ ] CUDA is used on the tested NVIDIA setup.
- [ ] `/queue` returns data.
- [ ] `/predict` returns a triage result.
- [ ] `/reviews` persists and retrieves human decisions.
- [ ] `/analytics/*` endpoints load successfully.

## UI

- [ ] Review Queue search and filters work.
- [ ] Existing report opens correctly.
- [ ] Fresh Analysis clears prior report context.
- [ ] Analytics renders charts/rankings without Details buttons.
- [ ] Mobile layout is usable at approximately 390px width.

## Release hygiene

- [ ] Runtime review database is empty/reset before the demo.
- [ ] No `.git` directory is shipped in the release archive.
- [ ] No Python cache files are shipped.
- [ ] No secrets are committed.
