import assert from "node:assert/strict";
import test from "node:test";

import { parseReportFeedbackRequest } from "./parseReportFeedbackRequest";

test("parseReportFeedbackRequest accepts valid comparable feedback payload", () => {
  const result = parseReportFeedbackRequest({
    request_id: "018f0f57-21cb-7f81-8d17-6efec2b5f2be",
    event_name: "comparable_marked_relevant",
    entity_type: "comparable",
    entity_id: "cand-1",
    section: "comparables",
    rank: 2,
    objective_effective: "engagement",
    experiment_id: "rec_v2",
    variant: "control",
    signal_strength: "strong",
    label_direction: "positive",
    metadata: {
      source: "report-panel"
    }
  });

  assert.equal(result.ok, true);
  if (result.ok) {
    assert.equal(result.value.event_name, "comparable_marked_relevant");
    assert.equal(result.value.rank, 2);
    assert.equal(result.value.signal_strength, "strong");
  }
});

test("parseReportFeedbackRequest rejects invalid signal strength", () => {
  const result = parseReportFeedbackRequest({
    request_id: "018f0f57-21cb-7f81-8d17-6efec2b5f2be",
    event_name: "report_viewed",
    entity_type: "report",
    section: "header",
    objective_effective: "engagement",
    signal_strength: "loud",
    label_direction: "context"
  });

  assert.equal(result.ok, false);
});

test("parseReportFeedbackRequest accepts no-good-options session feedback", () => {
  const result = parseReportFeedbackRequest({
    request_id: "018f0f57-21cb-7f81-8d17-6efec2b5f2be",
    event_name: "comparable_no_good_options",
    entity_type: "report",
    section: "comparables",
    objective_effective: "engagement",
    signal_strength: "strong",
    label_direction: "negative"
  });

  assert.equal(result.ok, true);
  if (result.ok) {
    assert.equal(result.value.event_name, "comparable_no_good_options");
    assert.equal(result.value.entity_type, "report");
    assert.equal(result.value.entity_id, undefined);
  }
});
