import test from "node:test";
import assert from "node:assert/strict";

import { parseRecommendationsRequest } from "./parseRecommendationsRequest";

test("parseRecommendationsRequest accepts valid payload", () => {
  const result = parseRecommendationsRequest({
    description: "Test",
    hashtags: ["#growth"],
    mentions: ["@user"],
    objective: "community",
    as_of_time: "2026-03-22T00:00:00Z",
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
  assert.equal(result.value.debug, true);
});

test("parseRecommendationsRequest rejects invalid top_k", () => {
  const result = parseRecommendationsRequest({
    description: "Test",
    top_k: 0
  });
  assert.equal(result.ok, false);
});

