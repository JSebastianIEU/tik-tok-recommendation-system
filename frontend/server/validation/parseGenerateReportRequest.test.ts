import assert from "node:assert/strict";
import test from "node:test";

import { parseGenerateReportRequest } from "./parseGenerateReportRequest";

test("parseGenerateReportRequest accepts valid payload with signal hints", () => {
  const result = parseGenerateReportRequest({
    seed_video_id: "s123",
    description: "How to prep meals in 10 minutes",
    mentions: ["@creator"],
    hashtags: ["#mealprep", "fitness"],
    objective: "conversion",
    audience: "busy students",
    content_type: "tutorial",
    primary_cta: "save",
    locale: "en-US",
    language: "en",
    signal_hints: {
      duration_seconds: 42,
      transcript_text: "Step one cook pasta",
      estimated_scene_cuts: 16,
      visual_motion_score: 0.5,
      speech_seconds: 25,
      music_seconds: 17
    }
  });

  assert.equal(result.ok, true);
  if (!result.ok) {
    return;
  }
  assert.equal(result.value.seed_video_id, "s123");
  assert.equal(result.value.objective, "conversion");
  assert.equal(result.value.content_type, "tutorial");
  assert.equal(result.value.primary_cta, "save");
  assert.equal(result.value.language, "en");
  assert.equal(result.value.signal_hints?.duration_seconds, 42);
});

test("parseGenerateReportRequest rejects invalid enum values", () => {
  const result = parseGenerateReportRequest({
    description: "x",
    mentions: [],
    hashtags: [],
    objective: "invalid_objective"
  });

  assert.equal(result.ok, false);
});

test("parseGenerateReportRequest rejects malformed arrays", () => {
  const result = parseGenerateReportRequest({
    description: "x",
    mentions: "not-array",
    hashtags: []
  });

  assert.equal(result.ok, false);
});

test("parseGenerateReportRequest rejects invalid signal hints", () => {
  const result = parseGenerateReportRequest({
    description: "x",
    mentions: [],
    hashtags: [],
    signal_hints: {
      tempo_bpm: "fast"
    }
  });

  assert.equal(result.ok, false);
});

test("parseGenerateReportRequest applies defaults when optional fields are missing", () => {
  const result = parseGenerateReportRequest({
    description: "Simple draft",
    mentions: [],
    hashtags: []
  });

  assert.equal(result.ok, true);
  if (!result.ok) {
    return;
  }

  assert.equal(result.value.seed_video_id, "s001");
  assert.equal(result.value.audience, "");
  assert.equal(result.value.objective, undefined);
  assert.equal(result.value.content_type, undefined);
  assert.equal(result.value.primary_cta, undefined);
});

test("parseGenerateReportRequest accepts structured audience objects", () => {
  const result = parseGenerateReportRequest({
    description: "Draft",
    mentions: [],
    hashtags: [],
    audience: {
      label: "small business owners",
      segments: ["small_business", "marketing"],
      expertise_level: "beginner"
    }
  });

  assert.equal(result.ok, true);
  if (!result.ok) {
    return;
  }

  assert.deepEqual(result.value.audience, {
    label: "small business owners",
    segments: ["small_business", "marketing"],
    expertise_level: "beginner"
  });
});
