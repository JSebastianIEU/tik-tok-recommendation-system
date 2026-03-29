import assert from "node:assert/strict";
import test from "node:test";

import {
  buildExperimentAssignment,
  buildExperimentRoutePolicy,
  stableHashHex
} from "./experimentation";

const BASE_CONFIG = {
  defaultExperimentId: "rec_v2_default",
  treatmentRatio: 0.5,
  salt: "unit-test-salt",
  controlBundleId: "",
  treatmentBundleId: ""
};

test("buildExperimentAssignment is deterministic for the same unit hash", () => {
  const first = buildExperimentAssignment({
    objectiveRequested: "community",
    assignmentUnit: "seed-video::v123",
    config: BASE_CONFIG
  });
  const second = buildExperimentAssignment({
    objectiveRequested: "community",
    assignmentUnit: "seed-video::v123",
    config: BASE_CONFIG
  });
  assert.equal(first.assignment_key, second.assignment_key);
  assert.equal(first.variant, second.variant);
  assert.equal(first.objective_effective, "engagement");
  assert.equal(first.unit_hash, stableHashHex("seed-video::v123"));
});

test("buildExperimentAssignment honors forced variant", () => {
  const assignment = buildExperimentAssignment({
    objectiveRequested: "reach",
    assignmentUnit: "qhash::123",
    config: BASE_CONFIG,
    experiment: { id: "spring_launch", force_variant: "control" }
  });
  assert.equal(assignment.variant, "control");
  assert.equal(assignment.experiment_id, "spring_launch");
});

test("buildExperimentRoutePolicy maps variant to required compat and bundle pin", () => {
  const assignment = buildExperimentAssignment({
    objectiveRequested: "engagement",
    assignmentUnit: "qhash::444",
    config: {
      ...BASE_CONFIG,
      treatmentRatio: 1
    }
  });
  const policy = buildExperimentRoutePolicy({
    assignment,
    config: {
      ...BASE_CONFIG,
      treatmentBundleId: "bundle_treat_42"
    }
  });
  assert.equal(policy.modelFamilySuffix, "treatment");
  assert.equal(policy.requiredCompat.bundle_id, "bundle_treat_42");
});
