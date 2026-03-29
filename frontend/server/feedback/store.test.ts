import assert from "node:assert/strict";
import test from "node:test";

import { FeedbackEventStore, buildRequestHash } from "./store";

test("buildRequestHash is deterministic", () => {
  const first = buildRequestHash({
    objectiveRequested: "community",
    objectiveEffective: "engagement",
    requestId: "018f0f57-21cb-7f81-8d17-6efec2b5f2be",
    assignmentUnitHash: "abc"
  });
  const second = buildRequestHash({
    objectiveRequested: "community",
    objectiveEffective: "engagement",
    requestId: "018f0f57-21cb-7f81-8d17-6efec2b5f2be",
    assignmentUnitHash: "abc"
  });
  assert.equal(first, second);
});

test("FeedbackEventStore remains disabled cleanly without db url", async () => {
  const store = new FeedbackEventStore({
    enabled: true,
    dbUrl: ""
  });
  await store.init();
  const status = store.status();
  assert.equal(status.ready, false);
  assert.equal(status.error, "feedback_store_disabled");
});

