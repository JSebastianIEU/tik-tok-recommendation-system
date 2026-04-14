import assert from "node:assert/strict";
import test from "node:test";

import {
  buildKnowledgeBaseStore,
  searchKnowledgeBase,
  validateKnowledgeBaseDataset,
  type KnowledgeBaseDataset
} from "./knowledgeBase";

function buildDataset(): KnowledgeBaseDataset {
  return {
    version: "tiktok_kb.v1",
    entries: [
      {
        id: "creators-1",
        title: "Top creators repeat recognizable formats",
        category: "creators",
        content: ["Top creators often keep a repeatable format signature."],
        action_hint: "Define and repeat one recognizable post structure.",
        impact_area: "format_consistency",
        keywords: ["top creators", "creator strategy", "format"],
        objective_tags: ["reach", "engagement"],
        updated_at: "2026-04-14T00:00:00Z",
        confidence: "medium",
        active: true
      },
      {
        id: "hashtags-1",
        title: "Hashtag intent balance improves relevance",
        category: "hashtags",
        content: ["Balanced hashtag sets usually improve qualified discovery."],
        action_hint: "Use 3-5 hashtags with one broad and two niche tags.",
        impact_area: "qualified_discovery",
        keywords: ["hashtags", "discoverability", "niche"],
        objective_tags: ["reach", "engagement"],
        updated_at: "2026-04-14T00:00:00Z",
        confidence: "high",
        active: true
      },
      {
        id: "algo-1",
        title: "Early hold influences deeper distribution",
        category: "algorithm",
        content: ["Strong first-second clarity helps distribution depth."],
        action_hint: "Put the value proposition in the first second.",
        impact_area: "early_hold",
        keywords: ["algorithm", "fyp", "distribution", "first second"],
        objective_tags: ["reach", "engagement"],
        updated_at: "2026-04-14T00:00:00Z",
        confidence: "high",
        active: true
      }
    ]
  };
}

test("validateKnowledgeBaseDataset accepts a valid curated dataset", () => {
  const dataset = validateKnowledgeBaseDataset(buildDataset());
  assert.equal(dataset.version, "tiktok_kb.v1");
  assert.equal(dataset.entries.length, 3);
});

test("validateKnowledgeBaseDataset rejects malformed entries", () => {
  const malformed = {
    version: "tiktok_kb.v1",
    entries: [
      {
        id: "bad-1",
        title: "Missing category",
        content: ["bad"],
        action_hint: "bad",
        impact_area: "bad",
        keywords: ["bad"],
        updated_at: "2026-04-14T00:00:00Z",
        confidence: "high",
        active: true
      }
    ]
  };

  assert.throws(
    () => validateKnowledgeBaseDataset(malformed),
    /invalid 'category'/i
  );
});

test("validateKnowledgeBaseDataset rejects inactive-only datasets", () => {
  const inactiveOnly = {
    version: "tiktok_kb.v1",
    entries: [
      {
        id: "inactive-1",
        title: "Inactive",
        category: "algorithm",
        content: ["inactive"],
        action_hint: "inactive",
        impact_area: "inactive",
        keywords: ["inactive"],
        updated_at: "2026-04-14T00:00:00Z",
        confidence: "medium",
        active: false
      }
    ]
  };

  assert.throws(
    () => validateKnowledgeBaseDataset(inactiveOnly),
    /no active entries/i
  );
});

test("searchKnowledgeBase ranks creators category first for creator-focused queries", () => {
  const store = buildKnowledgeBaseStore(buildDataset());
  const result = searchKnowledgeBase(store, {
    question: "What do top creators do that performs well?",
    objective: "engagement",
    priority: "high",
    maxResults: 3
  });

  assert.ok(result.entries.length > 0);
  assert.equal(result.entries[0]?.category, "creators");
});

test("searchKnowledgeBase surfaces hashtag guidance for hashtag engagement queries", () => {
  const store = buildKnowledgeBaseStore(buildDataset());
  const result = searchKnowledgeBase(store, {
    question: "Best hashtags for engagement on TikTok?",
    objective: "engagement",
    priority: "high",
    maxResults: 3
  });

  assert.ok(result.entries.length > 0);
  assert.ok(result.entries.some((entry) => entry.category === "hashtags"));
});
