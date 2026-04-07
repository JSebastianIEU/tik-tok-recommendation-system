# Recommender SLO and SLA

## Measurement Sources

- Python service health, compatibility, and metrics:
  - `GET /v1/health`
  - `GET /v1/compatibility`
  - `GET /v1/metrics`
- Node gateway metrics:
  - `GET /recommender-gateway-metrics`
- Control-plane validation:
  - `scripts/run_live_e2e_validation.py`
  - `scripts/run_outcome_attribution.py`
  - `scripts/run_drift_monitor.py`
  - `scripts/run_experiment_analysis.py`
  - `scripts/run_retrain_controller.py`

## SLOs

### Availability

- Node recommendation endpoint availability: `>= 99.5%` monthly.
- Python recommender availability when enabled: `>= 99.5%` monthly.
- Control-plane job success rate: `100%` for scheduled daily runs.

### Latency

- Node recommendation `p95` total latency: `<= 250 ms`.
- Python roundtrip `p95` latency: `<= 150 ms`.
- Compatibility check `p95` latency: `<= 10 ms`.

### Quality

- Fallback rate excluding injected failures: `<= 2%` over a 7-day rolling window.
- Compatibility mismatch rate: `<= 0.5%` over a 7-day rolling window.
- Breaker open rate: `<= 1%` of requests over a 7-day rolling window.
- Feedback persistence success: `>= 99%` of served requests when DB is enabled.

### Validation

- Live E2E validation must pass with `overall_passed: true`.
- Request plan coverage must include each objective/variant pair when run-scoped.
- Control-plane outputs must be produced for outcome attribution, drift monitoring, experiment analysis, and retrain decisions.

## SLA

Internal service owners will investigate within one business day for SLO breaches and within one hour for repeated compatibility or breaker incidents.

## Example Threshold Check

```bash
curl -s http://127.0.0.1:5174/recommender-gateway-metrics
curl -s http://127.0.0.1:8081/v1/metrics
PYTHONPATH=. python3 scripts/run_live_e2e_validation.py \
  --launch-python \
  --launch-node \
  --run-control-plane \
  --db-url "$DATABASE_URL" \
  --output-json artifacts/control_plane/live_e2e_validation_report.json
```

Treat any report with elevated fallback, repeated compatibility mismatches, or `overall_passed: false` as an SLO breach until explained.
