# V2 Release Checklist

Use this checklist before promoting any V2 recommender bundle.

## Required Evidence

- Live E2E validation passes:
  - `overall_passed: true`
  - `feedback_events_persisted: true`
  - `control_plane_jobs: true`
- Python service is healthy and compatible.
- Node gateway is healthy and not stuck in an unexplained fallback state.
- Outcome attribution, drift monitor, experiment analysis, and retrain controller artifacts are produced.
- The current bundle id and request ids are recorded in the release evidence.

## Required Commands

```bash
PYTHONPATH=. python3 scripts/run_live_e2e_validation.py \
  --launch-python \
  --launch-node \
  --run-control-plane \
  --db-url "$DATABASE_URL" \
  --output-json artifacts/control_plane/live_e2e_validation_report.json

PYTHONPATH=. python3 scripts/run_outcome_attribution.py \
  --db-url "$DATABASE_URL" \
  --output-json artifacts/control_plane/outcome_attribution_report.json

PYTHONPATH=. python3 scripts/run_drift_monitor.py \
  --db-url "$DATABASE_URL" \
  --output-json artifacts/control_plane/drift_report.json

PYTHONPATH=. python3 scripts/run_experiment_analysis.py \
  --db-url "$DATABASE_URL" \
  --output-json artifacts/control_plane/experiment_report.json

PYTHONPATH=. python3 scripts/run_retrain_controller.py \
  --db-url "$DATABASE_URL" \
  --drift-report-json artifacts/control_plane/drift_report.json \
  --output-json artifacts/control_plane/retrain_decision.json
```

## Pass Criteria

- `GET /v1/health` returns `ok: true`.
- `GET /v1/compatibility` returns `ok: true`.
- Gateway `p95` total latency stays within budget.
- Fallback reasons are expected and explained.
- Control-plane reports are either `insufficient_data` by design or pass the matured evidence thresholds.

## Matured Evidence Gate

- `experiment_report.objectives[*].comparison.evidence_sufficient == true`
- `matured_primary_24h_samples >= 20` per objective and variant
- `matured_stability_96h_samples >= 10` per objective and variant
- `fallback_rate_treatment_minus_control <= 0.02`
- `policy_violation_rate_treatment_minus_control <= 0.005`
- `latency_p95_ms_treatment_minus_control <= 25`

## Rollback Gate

Do not promote if:

- Compatibility mismatches are unexplained.
- Breakers are opening repeatedly.
- Any control-plane job fails.
- Request-scoped validation is failing closed unexpectedly.
- Derived-only governance is violated.

## Evidence Bundle

Attach the following to the release record:

- `artifacts/control_plane/live_e2e_validation_report.json`
- `artifacts/control_plane/outcome_attribution_report.json`
- `artifacts/control_plane/drift_report.json`
- `artifacts/control_plane/experiment_report.json`
- `artifacts/control_plane/retrain_decision.json`
- Current bundle id and previous bundle id

## Owner Sign-Off

- Recommender owner
- Node gateway owner
- Data/control-plane owner
- On-call reviewer
