import assert from "node:assert/strict";
import test from "node:test";

import { RecommenderCircuitBreakers } from "./resilience";

test("circuit breaker opens on consecutive failures and blocks requests", () => {
  const breaker = new RecommenderCircuitBreakers({
    minRequests: 20,
    errorRateThreshold: 0.6,
    consecutiveFailureThreshold: 5,
    windowMs: 60000,
    openMs: 30000,
    halfOpenMaxProbes: 3,
    halfOpenSuccessToClose: 2
  });
  const key = "/v1/recommendations::engagement";
  const base = 1000;
  for (let index = 0; index < 5; index += 1) {
    const allowed = breaker.shouldAllow(key, base + index);
    assert.equal(allowed.allow, true);
    breaker.recordFailure(key, base + index);
  }
  const blocked = breaker.shouldAllow(key, base + 10);
  assert.equal(blocked.allow, false);
  assert.equal(blocked.state, "open");
});

test("circuit breaker transitions half-open to closed on successful probes", () => {
  const breaker = new RecommenderCircuitBreakers({
    minRequests: 20,
    errorRateThreshold: 0.6,
    consecutiveFailureThreshold: 1,
    windowMs: 60000,
    openMs: 10,
    halfOpenMaxProbes: 3,
    halfOpenSuccessToClose: 2
  });
  const key = "/v1/recommendations::engagement";
  breaker.recordFailure(key, 1000);
  const blocked = breaker.shouldAllow(key, 1001);
  assert.equal(blocked.allow, false);
  const probe1 = breaker.shouldAllow(key, 1012);
  assert.equal(probe1.allow, true);
  assert.equal(probe1.state, "half_open");
  breaker.recordSuccess(key, 1012);
  const probe2 = breaker.shouldAllow(key, 1013);
  assert.equal(probe2.allow, true);
  breaker.recordSuccess(key, 1013);
  const closed = breaker.shouldAllow(key, 1014);
  assert.equal(closed.state, "closed");
  assert.equal(closed.allow, true);
});
