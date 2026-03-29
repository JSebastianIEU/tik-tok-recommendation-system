import test from "node:test";
import assert from "node:assert/strict";

import { parseRecommendationsRequest } from "./parseRecommendationsRequest";

test("parseRecommendationsRequest accepts valid payload", () => {
  const result = parseRecommendationsRequest({
    description: "Test",
    hashtags: ["#growth"],
    mentions: ["@user"],
    objective: "community",
    language: "en",
    as_of_time: "2026-03-22T00:00:00Z",
    candidate_ids: ["v1", "v2"],
    policy_overrides: {
      strict_language: true,
      max_items_per_author: 2
    },
    portfolio: {
      enabled: true,
      weights: {
        reach: 0.5,
        conversion: 0.3,
        durability: 0.2
      },
      risk_aversion: 0.15,
      candidate_pool_cap: 120
    },
    graph_controls: {
      enable_graph_branch: true
    },
    trajectory_controls: {
      enabled: true
    },
    explainability: {
      enabled: true,
      top_features: 4,
      neighbor_k: 2,
      run_counterfactuals: true
    },
    routing: {
      track: "post_publication",
      allow_fallback: true,
      required_compat: {
        component: "recommender-learning-v1"
      }
    },
    experiment: {
      id: "rec_v2_launch",
      force_variant: "treatment"
    },
    top_k: 15,
    retrieve_k: 300,
    debug: true
  });
  assert.equal(result.ok, true);
  if (!result.ok) {
    return;
  }
  assert.equal(result.value.top_k, 15);
  assert.equal(result.value.retrieve_k, 300);
  assert.equal(result.value.language, "en");
  assert.deepEqual(result.value.candidate_ids, ["v1", "v2"]);
  assert.equal(result.value.policy_overrides.strict_language, true);
  assert.equal(result.value.policy_overrides.max_items_per_author, 2);
  assert.equal(result.value.portfolio.enabled, true);
  assert.equal(result.value.portfolio.weights?.reach, 0.5);
  assert.equal(result.value.portfolio.risk_aversion, 0.15);
  assert.equal(result.value.portfolio.candidate_pool_cap, 120);
  assert.equal(result.value.graph_controls.enable_graph_branch, true);
  assert.equal(result.value.trajectory_controls.enabled, true);
  assert.equal(result.value.explainability.enabled, true);
  assert.equal(result.value.explainability.top_features, 4);
  assert.equal(result.value.explainability.neighbor_k, 2);
  assert.equal(result.value.explainability.run_counterfactuals, true);
  assert.equal(result.value.routing.track, "post_publication");
  assert.equal(result.value.routing.allow_fallback, true);
  assert.equal(result.value.routing.required_compat.component, "recommender-learning-v1");
  assert.equal(result.value.experiment.id, "rec_v2_launch");
  assert.equal(result.value.experiment.force_variant, "treatment");
  assert.equal(result.value.debug, true);
});

test("parseRecommendationsRequest rejects invalid top_k", () => {
  const result = parseRecommendationsRequest({
    description: "Test",
    top_k: 0
  });
  assert.equal(result.ok, false);
});

test("parseRecommendationsRequest rejects invalid candidate_ids type", () => {
  const result = parseRecommendationsRequest({
    description: "Test",
    candidate_ids: "bad"
  });
  assert.equal(result.ok, false);
});

test("parseRecommendationsRequest rejects invalid policy_overrides", () => {
  const result = parseRecommendationsRequest({
    description: "Test",
    policy_overrides: {
      strict_locale: "yes"
    }
  });
  assert.equal(result.ok, false);
});

test("parseRecommendationsRequest rejects invalid portfolio", () => {
  const result = parseRecommendationsRequest({
    description: "Test",
    portfolio: {
      enabled: true,
      weights: {
        reach: -0.2
      }
    }
  });
  assert.equal(result.ok, false);
});

test("parseRecommendationsRequest rejects invalid explainability", () => {
  const result = parseRecommendationsRequest({
    description: "Test",
    explainability: {
      top_features: 0
    }
  });
  assert.equal(result.ok, false);
});

test("parseRecommendationsRequest rejects invalid graph_controls", () => {
  const result = parseRecommendationsRequest({
    description: "Test",
    graph_controls: {
      enable_graph_branch: "yes"
    }
  });
  assert.equal(result.ok, false);
});

test("parseRecommendationsRequest rejects invalid trajectory_controls", () => {
  const result = parseRecommendationsRequest({
    description: "Test",
    trajectory_controls: {
      enabled: "yes"
    }
  });
  assert.equal(result.ok, false);
});

test("parseRecommendationsRequest rejects invalid routing", () => {
  const result = parseRecommendationsRequest({
    description: "Test",
    routing: {
      track: "bad"
    }
  });
  assert.equal(result.ok, false);
});

test("parseRecommendationsRequest rejects invalid experiment", () => {
  const result = parseRecommendationsRequest({
    description: "Test",
    experiment: {
      force_variant: "bad"
    }
  });
  assert.equal(result.ok, false);
});
