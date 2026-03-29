import assert from "node:assert/strict";
import test from "node:test";

import { buildRoutingEnvelope } from "./routing";

test("buildRoutingEnvelope maps community to engagement and applies defaults", () => {
  const route = buildRoutingEnvelope({
    objectiveRequested: "community",
    requestId: "018f0f57-21cb-7f81-8d17-6efec2b5f2be",
    experiment: { id: "rec_v2", variant: "control", unit_hash: "abc" },
    stageBudgetsMs: { retrieval: 260, ranking: 260, explainability: 300 }
  });
  assert.equal(route.objective_requested, "community");
  assert.equal(route.objective_effective, "engagement");
  assert.equal(route.track, "post_publication");
  assert.equal(route.allow_fallback, true);
  assert.equal(route.required_compat.component, "recommender-learning-v1");
  assert.equal(route.model_family, "ranker_engagement");
  assert.equal(route.request_id, "018f0f57-21cb-7f81-8d17-6efec2b5f2be");
  assert.equal(route.experiment?.id, "rec_v2");
});

test("buildRoutingEnvelope keeps explicit track and required_compat override", () => {
  const route = buildRoutingEnvelope({
    objectiveRequested: "reach",
    track: "pre_publication",
    allowFallback: false,
    requiredCompat: { feature_schema_hash: "abc123" },
    modelFamilySuffix: "treatment",
    stageBudgetsMs: { retrieval: 1, ranking: 2, explainability: 3 }
  });
  assert.equal(route.objective_effective, "reach");
  assert.equal(route.track, "pre_publication");
  assert.equal(route.allow_fallback, false);
  assert.equal(route.required_compat.feature_schema_hash, "abc123");
  assert.equal(route.stage_budgets_ms.ranking, 2);
  assert.equal(route.model_family, "ranker_reach_treatment");
});
