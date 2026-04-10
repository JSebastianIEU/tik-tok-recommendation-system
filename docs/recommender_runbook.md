# Recommender Runbook

## Scope

This runbook covers the Node gateway, Python recommender service, and the control-plane jobs that validate and promote bundles.

## Healthy State

- Node gateway is ready at `GET /recommender-gateway-metrics`.
- Python service is ready at `GET /v1/health`.
- Python compatibility is healthy at `GET /v1/compatibility`.
- Python request metrics are visible at `GET /v1/metrics`.
- Live E2E validation passes with `overall_passed: true`.

## First Checks

```bash
curl -s http://127.0.0.1:8081/v1/health
curl -s http://127.0.0.1:8081/v1/compatibility
curl -s http://127.0.0.1:8081/v1/metrics
curl -s http://127.0.0.1:5174/recommender-gateway-metrics
```

Look for:

- `status: "ready"` from Python health.
- `ok: true` from compatibility.
- Stable latency summaries.
- `fallback_by_reason` containing only expected injected failures during validation.

## Triage Order

1. Check whether Node is serving fallback because Python is down, incompatible, or timing out.
2. Check Python health and compatibility.
3. Check gateway metrics for breaker transitions and fallback spikes.
4. Check recent live E2E report and control-plane reports.
5. Verify feedback persistence if DB is attached.

## Common Failure Modes

- `recommender_disabled`: Node is intentionally bypassing Python.
- `fetch failed`: Python is unreachable.
- `incompatible_artifact` or `required_compat_mismatch`: bundle/runtime mismatch.
- `breaker open`: gateway is protecting downstream by serving fallback.
- `fallback_reason` changes without an expected deployment: investigate before promoting.

## Rollback

Rollback means restoring the last known good bundle and reducing live routing pressure.

```bash
# Force Node to stop calling Python and serve fallback only
export RECOMMENDER_ENABLED=false

# Or point the gateway at a previous healthy bundle / stable Python endpoint
export RECOMMENDER_BASE_URL="http://127.0.0.1:8081"
export RECOMMENDER_FALLBACK_BUNDLE_DIR="artifacts/recommender_fallback"
```

If a newly promoted bundle is the cause, revert `RECOMMENDER_BUNDLE_DIR` to the previous bundle and restart Python.

## Escalation

Escalate immediately if any of the following are true:

- Compatibility mismatches persist after one restart.
- Breaker opens repeatedly within the same incident window.
- Python health remains degraded.
- Control-plane jobs stop producing artifacts.
- Feedback DB persistence drops below expected counts.

## Incident Notes

- Keep the request id, bundle id, and fallback reason for every incident.
- Attach the latest live E2E report and the gateway metrics snapshot to the incident record.
- Prefer deterministic fallback over partial or ambiguous service behavior.
