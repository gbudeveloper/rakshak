# Development notes

The submission uses one canonical runtime entrypoint: `scripts/rakshak_api.py`.

The model, reviewer policy, evidence engine, and activity metadata are frozen for the SIH 2026 prototype. Historical training/evaluation scripts are retained under `research/archive/` and are not part of the runtime path.

The `apps/web` dashboard is the judge-facing interface. The annotation application and superseded implementations are preserved only for research traceability.
