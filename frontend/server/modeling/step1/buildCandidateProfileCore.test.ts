import assert from "node:assert/strict";
import test from "node:test";

import { buildCandidateProfileCore } from "./buildCandidateProfileCore";

test("buildCandidateProfileCore applies defaults and keeps stable shape", () => {
  const profile = buildCandidateProfileCore({
    description: "Quick recipe for students",
    hashtags: ["mealprep"],
    mentions: []
  });

  assert.equal(profile.profile_version, "core.v1");
  assert.equal(profile.intent.objective, "engagement");
  assert.equal(profile.intent.content_type, "showcase");
  assert.equal(profile.intent.primary_cta, "none");
  assert.equal(profile.locale, "en");
  assert.ok(profile.created_at.length > 0);
});

test("normalizes and dedupes hashtags and mentions", () => {
  const profile = buildCandidateProfileCore({
    description: "Desc",
    hashtags: ["#MealPrep", "mealprep", "#MEALPREP", "#HighProteinMeals"],
    mentions: ["@Creator.Name", "creator.name", "@creator_name"]
  });

  assert.deepEqual(profile.normalized.hashtags, ["mealprep", "highproteinmeals"]);
  assert.deepEqual(profile.normalized.mentions, ["creator.name", "creator_name"]);
});

test("key feature counters are computed correctly", () => {
  const profile = buildCandidateProfileCore({
    description: "Top 3 tips for growth! Comment below?",
    hashtags: ["growth", "creator"],
    mentions: ["team"],
    audience: "creators"
  });

  assert.equal(profile.features.word_count > 0, true);
  assert.equal(profile.features.hashtag_count, 2);
  assert.equal(profile.features.mention_count, 1);
  assert.equal(profile.features.question_mark_present, true);
  assert.equal(profile.features.exclamation_present, true);
  assert.equal(profile.features.number_present, true);
});

test("marks low signal and missing audience in quality flags", () => {
  const profile = buildCandidateProfileCore({
    description: "",
    hashtags: [],
    mentions: []
  });

  assert.equal(profile.quality.flags.includes("missing_description"), true);
  assert.equal(profile.quality.flags.includes("missing_audience"), true);
  assert.equal(profile.quality.confidence <= 0.5, true);
});

test("supports multilingual text and produces retrieval text", () => {
  const profile = buildCandidateProfileCore({
    description: "Receta rápida para estudiantes con proteína",
    hashtags: ["comida", "mealprep"],
    mentions: [],
    locale: "es-ES",
    audience: "estudiantes"
  });

  assert.equal(profile.locale, "es-es");
  assert.equal(profile.tokens.description_tokens.length > 0, true);
  assert.equal(profile.retrieval.text.length > 0, true);
});

test("normalizes invalid locale to fallback", () => {
  const profile = buildCandidateProfileCore({
    description: "Sample",
    hashtags: ["sample"],
    mentions: [],
    locale: "not-a-locale!!!"
  });

  assert.equal(profile.locale, "en");
});

