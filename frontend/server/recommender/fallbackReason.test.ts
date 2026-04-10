import test from "node:test";
import assert from "node:assert/strict";

import { classifyRecommenderFailure } from "./fallbackReason";

test("classifyRecommenderFailure detects required compatibility mismatch", () => {
  const classified = classifyRecommenderFailure(
    "HTTP 422: {'detail': {'reason': 'required_compat_mismatch'}}"
  );
  assert.equal(classified.reason, "required_compat_mismatch");
  assert.equal(classified.compatibilityMismatch, true);
});

test("classifyRecommenderFailure detects incompatible artifact", () => {
  const classified = classifyRecommenderFailure("incompatible_artifact: feature schema mismatch");
  assert.equal(classified.reason, "incompatible_artifact");
  assert.equal(classified.compatibilityMismatch, true);
});

test("classifyRecommenderFailure detects stage timeout", () => {
  const classified = classifyRecommenderFailure("ranking_stage_timeout elapsed");
  assert.equal(classified.reason, "core_stage_timeout");
  assert.equal(classified.compatibilityMismatch, false);
});

test("classifyRecommenderFailure falls back to python_request_failed", () => {
  const classified = classifyRecommenderFailure("fetch failed");
  assert.equal(classified.reason, "python_request_failed");
  assert.equal(classified.compatibilityMismatch, false);
});
